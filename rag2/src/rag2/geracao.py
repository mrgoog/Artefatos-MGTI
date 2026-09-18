"""Harness de geração (frente B e confirmação): contexto congelado → agentes → juiz → auditor → resumo, tudo retomável e registrar-uma-vez."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from ensemble_llm.agentes.contratos import ClienteLLM
from ensemble_llm.agentes.estruturado import invocar_estruturado
from ensemble_llm.esquemas import BlocoRecuperado, ItemDataset
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.artefatos import ler_resumo
from rag2.auditoria import eh_recusa, parse_auditor, prompt_auditor
from rag2.bootstrap import ic_percentil
from rag2.config import RAIZ, ConfigRag2
from rag2.constantes import B_BOOTSTRAP, MODELO_JUIZ, RUN_PRODUCAO, TOLERANCIA_CANONICO
from rag2.dados import amostra_estratificada, contar_palavras, itens_com_dispositivo
from rag2.escada import GoldenDivergente
from rag2.prompts import VariantePrompt
from rag2.recuperadores.base import Recuperador

DIR_PROMPTS = RAIZ / "runs" / "prompts"
"""runs/prompts/<variante>/ — artefatos versionados de cada variante (arch §5)."""
TAMANHO_SMOKE = 20
"""Itens do smoke test de M4.1 (mecanismo, não a frente B pré-registrada de 60); estratificado sobre os 715, semente 42."""
DIR_CACHE_JUIZ = RAIZ / "runs" / "_shared" / "judge_labels"
"""Cache de rótulos do juiz, compartilhado entre variantes e pela confirmação; pesado, gitignored; pré-carregado da produção."""

COLUNAS_RESPOSTAS = (
    "item_id", "agent_id", "prompt_hash", "response_text", "confidence_raw", "valid_json",
    "n_retries", "raw_output", "cost_usd", "retrieved_chunk_ids", "timestamp",
)
"""Schema de responses.parquet — o mesmo da origem, para que M5.2 leia produção e novo com o mesmo código."""
COLUNAS_CONTEXTO = ("item_id", "combined_rank", "chunk_id", "source_doc", "text")
"""Contexto servido, com texto: congela o que o agente viu sem depender do índice na geração nem na auditoria."""
COLUNAS_ROTULOS = ("item_id", "agent_id", "z_agent", "prompt_hash_juiz")
"""z_agent ∈ {0,1}; prompt_hash_juiz = chave no cache do juiz ('' quando z=0 por JSON inválido, sem chamada)."""
COLUNAS_AUDITORIA = ("item_id", "agent_id", "veredicto_recusa", "raciocinio_auditor", "cost_usd")
"""Schema de auditoria_recusas.parquet — o mesmo da origem; veredicto 2/1/0, -1 = parse do auditor falhou."""


def agora_iso() -> str:
    """Timestamp UTC ISO 8601, como a origem grava em responses.parquet."""
    return datetime.now(UTC).isoformat()


def amostra_smoke(itens: Sequence[ItemDataset], n: int = TAMANHO_SMOKE) -> list[ItemDataset]:
    """Os n itens do smoke test (estratificada por has_doc_ref sobre os 715, semente 42), ordenados por item_id — ordem fixa dos artefatos.

    Não é a frente B pré-registrada (60) nem o piloto de 200: é a amostra de mecanismo de M4.1 (decisão 2026-09-13).
    """
    return sorted(amostra_estratificada(list(itens), n), key=lambda it: it.item_id)


def _parquet_atomico(df: pd.DataFrame, caminho: Path) -> None:
    """tmp + os.replace, como os caches deste repositório."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, caminho)


def _ler_ou_vazio(caminho: Path, colunas: Sequence[str]) -> pd.DataFrame:
    return pd.read_parquet(caminho) if caminho.exists() else pd.DataFrame(columns=list(colunas))


def blocos_de(contexto: pd.DataFrame, item_id: str) -> list[BlocoRecuperado]:
    """Reconstrói os BlocoRecuperado servidos a um item a partir de contexto.parquet (ordem de combined_rank)."""
    sub = contexto[contexto["item_id"] == item_id].sort_values("combined_rank")
    return [
        BlocoRecuperado(chunk_id=r.chunk_id, text=r.text, source_doc=r.source_doc, combined_rank=int(r.combined_rank))
        for r in sub.itertuples(index=False)
    ]


def recuperar_contexto(
    recuperador: Recuperador, itens: Sequence[ItemDataset], caminho: Path, *, log: Callable[[str], None] = print
) -> pd.DataFrame:
    """Recupera (pré-aquecendo em lote) só os itens ainda sem contexto no parquet e o regrava; devolve as linhas dos itens dados.

    Chama `recuperar(pergunta)` do degrau diretamente — nunca `executar_degrau`, que é o executor dos 556 e exige
    ItemAvaliado; aqui há itens sem dispositivo (5 dos 20). Os caches de pool/reranker do degrau só crescem.
    """
    atual = _ler_ou_vazio(caminho, COLUNAS_CONTEXTO)
    feitos = set(atual["item_id"])
    faltam = [it for it in itens if it.item_id not in feitos]
    if faltam:
        if hasattr(recuperador, "preaquecer"):
            recuperador.preaquecer([it.pergunta for it in faltam])
        linhas = [
            {"item_id": it.item_id, "combined_rank": b.combined_rank, "chunk_id": b.chunk_id, "source_doc": b.source_doc, "text": b.text}
            for it in faltam
            for b in recuperador.recuperar(it.pergunta)
        ]
        atual = pd.concat([atual, pd.DataFrame(linhas, columns=list(COLUNAS_CONTEXTO))], ignore_index=True)
        _parquet_atomico(atual, caminho)
    ids = [it.item_id for it in itens]
    saida = atual[atual["item_id"].isin(ids)]
    palavras = saida.groupby("item_id")["text"].apply(lambda s: sum(contar_palavras(t) for t in s))
    log(
        f"contexto: {caminho} | recuperados={len(faltam)} em_cache={len(itens) - len(faltam)} "
        f"chunks={len(saida)} palavras/item={palavras.reindex(ids).fillna(0).mean():.1f}"
    )
    return saida


def gerar_respostas(
    variante: VariantePrompt,
    cliente: ClienteLLM,
    agent_id: str,
    itens: Sequence[ItemDataset],
    contexto: pd.DataFrame,
    caminho: Path,
    *,
    log: Callable[[str], None] = print,
    flush_every: int = 10,
) -> pd.DataFrame:
    """Uma invocação estruturada por item faltante (k_max=3, como a origem), persistindo a cada flush_every; devolve as linhas dos itens dados.

    Reimplementa o laço de `_executar_agentes` da origem (privado e acoplado ao layout de runs/): mesmas colunas,
    mesma regra de parse (response_text/confidence_raw = None quando valid_json=False).
    """
    atual = _ler_ou_vazio(caminho, COLUNAS_RESPOSTAS)
    feitos = set(atual["item_id"])
    faltam = [it for it in itens if it.item_id not in feitos]
    linhas: list[dict] = atual.to_dict("records")
    n_falhas = 0
    for i, it in enumerate(faltam, 1):
        blocos = blocos_de(contexto, it.item_id)
        system, user = variante.construir(it.pergunta, blocos)
        r = invocar_estruturado(cliente, user, variante.schema, system_prompt=system, schema_hint=variante.schema_hint, k_max=3)
        ok = bool(r.valid_json) and r.parsed is not None
        n_falhas += 0 if ok else 1
        linhas.append(
            {
                "item_id": it.item_id,
                "agent_id": agent_id,
                "prompt_hash": r.prompt_hash,
                "response_text": r.parsed.resposta if ok else None,   # type: ignore[union-attr]
                "confidence_raw": float(r.parsed.confianca) if ok else None,   # type: ignore[union-attr]
                "valid_json": bool(r.valid_json),
                "n_retries": int(r.n_retries),
                "raw_output": r.raw_output,
                "cost_usd": float(r.cost_usd),
                "retrieved_chunk_ids": [b.chunk_id for b in blocos],
                "timestamp": agora_iso(),
            }
        )
        if i % flush_every == 0 or i == len(faltam):
            _parquet_atomico(pd.DataFrame(linhas, columns=list(COLUNAS_RESPOSTAS)), caminho)
    log(
        f"agente {agent_id}: {caminho} | geradas={len(faltam)} em_cache={len(itens) - len(faltam)} "
        f"falhas_parse={n_falhas} modelo={cliente.model}"
    )
    df = pd.DataFrame(linhas, columns=list(COLUNAS_RESPOSTAS))
    return df[df["item_id"].isin([it.item_id for it in itens])]


def preparar_cache_juiz(cfg: ConfigRag2, dir_cache: Path = DIR_CACHE_JUIZ, *, log: Callable[[str], None] = print) -> CacheRotulosJuiz:
    """Cache do juiz deste repositório; na 1ª vez copia o cache.parquet da produção (2.129 rótulos) — mesmo cache, arch §4."""
    nome = MODELO_JUIZ.replace("/", "_")
    destino = Path(dir_cache) / nome / "cache.parquet"
    origem = cfg.dir_run(RUN_PRODUCAO) / "judge_labels" / nome / "cache.parquet"
    if not destino.exists() and origem.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origem, destino)
        log(f"cache do juiz: pré-carregado de {origem}")
    return CacheRotulosJuiz(cache_dir=Path(dir_cache), judge_model=MODELO_JUIZ)


def julgar(
    itens: Sequence[ItemDataset],
    respostas: Mapping[str, pd.DataFrame],
    executor: ExecutorJuiz,
    caminho: Path,
    *,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Rótulo z por (item, agente): JSON inválido ou resposta vazia → z=0 sem chamar o juiz (regra da origem); resto via ExecutorJuiz (cache por prompt_hash)."""
    por_id = {it.item_id: it for it in itens}
    atual = _ler_ou_vazio(caminho, COLUNAS_ROTULOS)
    feitos = set(zip(atual["item_id"], atual["agent_id"], strict=True))
    linhas: list[dict] = atual.to_dict("records")
    n_cache_antes, custo_antes = len(executor.cache), executor.cache.custo_total()
    n_novos = n_z0 = 0
    for agent_id, df in respostas.items():
        for r in df.itertuples(index=False):
            if (r.item_id, agent_id) in feitos:
                continue
            if not bool(r.valid_json) or not r.response_text:
                linhas.append({"item_id": r.item_id, "agent_id": agent_id, "z_agent": 0, "prompt_hash_juiz": ""})
                n_z0 += 1
                continue
            rot = executor.julgar(item=por_id[r.item_id], resposta_candidata=str(r.response_text), fonte_resposta=agent_id)
            linhas.append({"item_id": r.item_id, "agent_id": agent_id, "z_agent": int(rot.z), "prompt_hash_juiz": rot.prompt_hash})
            n_novos += 1
    executor.cache.flush()
    df = pd.DataFrame(linhas, columns=list(COLUNAS_ROTULOS)).astype({"z_agent": "int64"})
    _parquet_atomico(df, caminho)
    chamadas = len(executor.cache) - n_cache_antes
    log(
        f"juiz: {executor.cache.cache_file} | rotuladas={n_novos} em_cache={len(feitos)} z0_por_parse={n_z0} "
        f"chamadas={chamadas} custo=US$ {executor.cache.custo_total() - custo_antes:.2f} modelo={executor.client.model}"
    )
    return df


def auditar(
    itens: Sequence[ItemDataset],
    respostas: Mapping[str, pd.DataFrame],
    variante: VariantePrompt,
    contexto: pd.DataFrame,
    cliente: ClienteLLM,
    caminho: Path,
    *,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Audita cada recusa textual (PADRAO_RECUSA sobre response_text) com o prompt da origem sobre o contexto que o agente viu; retomável por (item, agente)."""
    por_id = {it.item_id: it for it in itens}
    atual = _ler_ou_vazio(caminho, COLUNAS_AUDITORIA)
    feitos = set(zip(atual["item_id"], atual["agent_id"], strict=True))
    linhas: list[dict] = atual.to_dict("records")
    recusas = [(aid, r) for aid, df in respostas.items() for r in df.itertuples(index=False) if eh_recusa(r.response_text)]
    n_novas, custo = 0, 0.0
    for agent_id, r in recusas:
        if (r.item_id, agent_id) in feitos:
            continue
        it = por_id[r.item_id]
        system, user = prompt_auditor(it.pergunta, variante.contexto(blocos_de(contexto, r.item_id)), it.ground_truth, str(r.response_text))
        res = cliente.invoke(user, system_prompt=system)
        raciocinio, veredicto = parse_auditor(res.raw_output)
        linhas.append({"item_id": r.item_id, "agent_id": agent_id, "veredicto_recusa": int(veredicto), "raciocinio_auditor": raciocinio, "cost_usd": float(res.cost_usd)})
        n_novas += 1
        custo += float(res.cost_usd)
        if n_novas % 10 == 0:
            _parquet_atomico(pd.DataFrame(linhas, columns=list(COLUNAS_AUDITORIA)), caminho)
    df = pd.DataFrame(linhas, columns=list(COLUNAS_AUDITORIA)).astype({"veredicto_recusa": "int64"})
    _parquet_atomico(df, caminho)
    log(f"auditor: {caminho} | recusas={len(recusas)} auditadas={n_novas} em_cache={len(recusas) - n_novas} custo=US$ {custo:.2f} modelo={cliente.model}")
    return df


def por_item(
    itens: Sequence[ItemDataset],
    contexto: pd.DataFrame,
    respostas: Mapping[str, pd.DataFrame],
    rotulos: pd.DataFrame,
    auditoria: pd.DataFrame,
) -> pd.DataFrame:
    """Uma linha por item (ordem de `itens`): has_doc_ref, n_chunks, palavras, e por agente z_/recusa_/evasiva_/parse_; médias entre agentes.

    evasiva = recusa com veredicto do auditor 1; parse = valid_json. Item sem linha em `respostas` para um agente
    conta z=0, recusa=0, evasiva=0, parse=0 (não acontece na run completa; documenta o comportamento em --limite).
    """
    agentes = list(respostas)
    z = {(r.item_id, r.agent_id): int(r.z_agent) for r in rotulos.itertuples(index=False)}
    ev = {(r.item_id, r.agent_id) for r in auditoria.itertuples(index=False) if int(r.veredicto_recusa) == 1}
    resp = {(r.item_id, aid): r for aid, df in respostas.items() for r in df.itertuples(index=False)}
    ctx = contexto.groupby("item_id")["text"]
    n_chunks = ctx.size()
    palavras = ctx.apply(lambda s: sum(contar_palavras(t) for t in s))
    linhas = []
    for it in itens:
        d: dict = {
            "item_id": it.item_id,
            "has_doc_ref": bool(it.has_doc_ref),
            "n_chunks": int(n_chunks.get(it.item_id, 0)),
            "palavras": int(palavras.get(it.item_id, 0)),
        }
        for aid in agentes:
            r = resp.get((it.item_id, aid))
            d[f"z_{aid}"] = z.get((it.item_id, aid), 0)
            d[f"recusa_{aid}"] = int(r is not None and eh_recusa(r.response_text))
            d[f"evasiva_{aid}"] = int((it.item_id, aid) in ev)
            d[f"parse_{aid}"] = int(r is not None and bool(r.valid_json))
        for nome in ("z", "recusa", "evasiva", "parse"):
            d[f"{nome}_media"] = sum(d[f"{nome}_{aid}"] for aid in agentes) / len(agentes)
        linhas.append(d)
    return pd.DataFrame(linhas)


def resumo_variante(df: pd.DataFrame, agentes: Sequence[str], *, b: int = B_BOOTSTRAP) -> dict:
    """Adequação/recusa/evasiva/parse por agente e média entre agentes com IC percentil estratificado por has_doc_ref (B, semente 42)."""
    saida: dict = {
        "n_itens": int(len(df)),
        "n_respostas": int(len(df) * len(agentes)),
        "agentes": list(agentes),
        "n_chunks_medio": float(df["n_chunks"].mean()),
        "palavras_medias": float(df["palavras"].mean()),
        "por_agente": {
            aid: {
                "adequacao": float(df[f"z_{aid}"].mean()),
                "n_adequadas": int(df[f"z_{aid}"].sum()),
                "n_recusas": int(df[f"recusa_{aid}"].sum()),
                "n_evasivas": int(df[f"evasiva_{aid}"].sum()),
                "n_falhas_parse": int((1 - df[f"parse_{aid}"]).sum()),
            }
            for aid in agentes
        },
        "b_bootstrap": int(b),
    }
    for chave, coluna in (("adequacao_media", "z_media"), ("taxa_recusa", "recusa_media"), ("taxa_evasiva", "evasiva_media")):
        saida[chave] = float(df[coluna].mean())
        ic = ic_percentil(df, lambda d, c=coluna: float(d[c].mean()), b=b, coluna_estrato="has_doc_ref")
        saida[f"ic_{chave}"] = [float(ic.ci_lo), float(ic.ci_hi)]
    saida["taxa_falha_parse"] = float(1.0 - df["parse_media"].mean())
    return saida


CABECALHO_PROMPTS = (
    "| variante | degrau | adequação média [IC95] | qwen35_9b | granite41_8b | gemma4_e4b | recusa | evasiva | parse inválido | palavras/item |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def linha_prompts(nome: str, r: Mapping) -> str:
    """Linha da tabela P0…P3 (adequação por agente na ordem de AGENTES; '-' se o agente não consta)."""
    lo, hi = r["ic_adequacao_media"]
    pa = r["por_agente"]
    col = lambda aid: f"{pa[aid]['adequacao']:.4f}" if aid in pa else "-"   # noqa: E731
    return (
        f"| {nome} | {r['degrau']} | {r['adequacao_media']:.4f} [{lo:.4f}, {hi:.4f}] | {col('qwen35_9b')} | {col('granite41_8b')} "
        f"| {col('gemma4_e4b')} | {r['taxa_recusa']:.4f} | {r['taxa_evasiva']:.4f} | {r['taxa_falha_parse']:.4f} | {r['palavras_medias']:.0f} |"
    )


def gravar_variante(diretorio: Path, df: pd.DataFrame, res: dict, nome: str) -> Path:
    """por_item.parquet, resumo.json (indentado, UTF-8, chaves ordenadas) e resumo.md; devolve o diretório."""
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    df.to_parquet(diretorio / "por_item.parquet", index=False)
    (diretorio / "resumo.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (diretorio / "resumo.md").write_text(f"# {nome}\n\n{CABECALHO_PROMPTS}\n{linha_prompts(nome, res)}\n", encoding="utf-8")
    return diretorio


def tabela_prompts(dir_prompts: Path = DIR_PROMPTS) -> str:
    """Tabela markdown com uma linha por runs/prompts/<variante>/resumo.json, em ordem alfabética (P0, P1, …)."""
    if not Path(dir_prompts).exists():
        return CABECALHO_PROMPTS
    resumos = {p.name: ler_resumo(p) for p in sorted(Path(dir_prompts).iterdir()) if (p / "resumo.json").exists()}
    return "\n".join([CABECALHO_PROMPTS] + [linha_prompts(n, r) for n, r in resumos.items()])


def executar_variante(
    nome: str,
    degrau: str,
    recuperador: Recuperador,
    itens: Sequence[ItemDataset],
    variante: VariantePrompt,
    clientes_agentes: Mapping[str, ClienteLLM],
    executor_juiz: ExecutorJuiz,
    cliente_auditor: ClienteLLM,
    *,
    saida: Path | None = None,
    b: int = B_BOOTSTRAP,
    limite: int | None = None,
    sobrescrever: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Contexto → agentes (na ordem de clientes_agentes) → juiz → auditor → por_item/resumo em `saida` (padrão runs/prompts/<nome>).

    Registrar uma vez: se `saida/resumo.json` existe e não há `sobrescrever`, o resumo é recomputado dos artefatos
    congelados (nenhum modelo é chamado) e comparado ao golden pela adequação média — GoldenDivergente se
    |Δ| > TOLERANCIA_CANONICO; nada é regravado. Um golden registrado sobre outro degrau é erro, não Δ.
    """
    itens = list(itens)[:limite] if limite else list(itens)
    saida = Path(saida) if saida is not None else DIR_PROMPTS / nome
    golden = ler_resumo(saida)
    if golden is not None and not sobrescrever and golden.get("degrau") != degrau:
        raise GoldenDivergente(f"{nome}: golden registrado sobre {golden.get('degrau')}, pedido {degrau}; use --sobrescrever com decisão registrada")
    com_disp = {it.item_id for it in itens_com_dispositivo(itens)}
    log(f"amostra: {len(itens)} itens (has_doc_ref={sum(it.has_doc_ref for it in itens)}, com dispositivo={len(com_disp)}) | variante={nome} degrau={degrau}")
    saida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"item_id": it.item_id, "has_doc_ref": bool(it.has_doc_ref), "com_dispositivo": it.item_id in com_disp} for it in itens]
    ).to_parquet(saida / "itens.parquet", index=False)
    contexto = recuperar_contexto(recuperador, itens, saida / "contexto.parquet", log=log)
    respostas = {
        aid: gerar_respostas(variante, cli, aid, itens, contexto, saida / "agent_responses" / aid / "responses.parquet", log=log)
        for aid, cli in clientes_agentes.items()
    }
    rotulos = julgar(itens, respostas, executor_juiz, saida / "rotulos.parquet", log=log)
    auditoria = auditar(itens, respostas, variante, contexto, cliente_auditor, saida / "auditoria_recusas.parquet", log=log)
    df = por_item(itens, contexto, respostas, rotulos, auditoria)
    res = resumo_variante(df, list(clientes_agentes), b=b)
    res["variante"], res["degrau"] = nome, degrau
    if golden is not None and not sobrescrever:
        delta = res["adequacao_media"] - golden["adequacao_media"]
        res["delta_vs_golden"] = delta
        if abs(delta) > TOLERANCIA_CANONICO:
            raise GoldenDivergente(f"{nome}: adequacao_media={res['adequacao_media']:.4f} vs golden {golden['adequacao_media']:.4f}")
        return res
    gravar_variante(saida, df, res, nome)
    return res
