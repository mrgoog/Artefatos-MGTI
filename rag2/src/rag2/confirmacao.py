"""Confirmação em N=200 (DESIGN §2.3): etapas do harness de geração nos 200 do piloto + juiz de suficiência + custo por etapa, em runs/confirmacao/novo/ (retomável, registrar uma vez). O baseline não é reexecutado — M5.2 o lê da origem."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
from ensemble_llm.agentes.contratos import ClienteLLM
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.artefatos import ler_resumo
from rag2.bootstrap import ic_percentil
from rag2.config import RAIZ, ConfigRag2
from rag2.constantes import B_BOOTSTRAP, TAMANHO_AMOSTRA_CONFIRMACAO, TOLERANCIA_CANONICO
from rag2.dados import ids_amostra_200, itens_com_dispositivo
from rag2.escada import GoldenDivergente
from rag2.geracao import (
    CABECALHO_PROMPTS,
    agora_iso,
    auditar,
    gerar_respostas,
    julgar,
    linha_prompts,
    por_item,
    recuperar_contexto,
    resumo_variante,
)
from rag2.prompts import VariantePrompt
from rag2.recuperadores.base import Recuperador
from rag2.suficiencia import julgar_suficiencia

DIR_CONFIRMACAO = RAIZ / "runs" / "confirmacao"
"""runs/confirmacao/<nome>/ — artefatos versionados da confirmação (arch §5); M5.2 grava comparacao.* e custo.md ao lado."""
NOME_NOVO = "novo"
"""Subdiretório do pipeline novo; o baseline não é copiado para cá (é lido da origem por M5.2)."""

CABECALHO_SUFICIENCIA = "| confirmação | suficiente [IC95] | parcial | insuficiente | falha de parse |\n|---|---:|---:|---:|---:|"


def itens_confirmacao(cfg: ConfigRag2, itens: Sequence[ItemDataset]) -> list[ItemDataset]:
    """Os 200 do piloto (ids_amostra_200), ordenados por item_id — as mesmas questões dos runs do pipeline antigo (pré-registro §2)."""
    ids = set(ids_amostra_200(cfg))
    sel = sorted((it for it in itens if it.item_id in ids), key=lambda it: it.item_id)
    if len(sel) != TAMANHO_AMOSTRA_CONFIRMACAO:
        raise ValueError(f"amostra da confirmação: {len(sel)} itens casaram os {len(ids)} ids do piloto (esperados {TAMANHO_AMOSTRA_CONFIRMACAO})")
    return sel


def _n_faltantes(caminho: Path, ids: Sequence[str]) -> int:
    """Quantos dos ids ainda não têm linha no parquet (todos, se o arquivo não existe) — contagem de chamadas desta execução."""
    if not caminho.exists():
        return len(ids)
    return len(set(ids) - set(pd.read_parquet(caminho, columns=["item_id"])["item_id"]))


def _n_linhas(caminho: Path) -> int:
    return len(pd.read_parquet(caminho, columns=["item_id"])) if caminho.exists() else 0


def resumo_suficiencia(df: pd.DataFrame, *, b: int = B_BOOTSTRAP) -> dict:
    """Contagens do veredicto de suficiência (2/1/0/-1) e frações 'suficiente' (= 2) e 'insuficiente' (= 0) com IC percentil estratificado por has_doc_ref."""
    v = df["suficiencia"]
    saida: dict = {
        "n_suficiente": int((v == 2).sum()),
        "n_parcial": int((v == 1).sum()),
        "n_insuficiente": int((v == 0).sum()),
        "n_falha_parse": int((v == -1).sum()),
    }
    d = df.assign(suf=(v == 2).astype(float), ins=(v == 0).astype(float))
    for chave, col in (("frac_suficiente", "suf"), ("frac_insuficiente", "ins")):
        saida[chave] = float(d[col].mean())
        ic = ic_percentil(d, lambda x, c=col: float(x[c].mean()), b=b, coluna_estrato="has_doc_ref")
        saida[f"ic_{chave}"] = [float(ic.ci_lo), float(ic.ci_hi)]
    return saida


def linha_suficiencia(nome: str, s: Mapping) -> str:
    """Linha da tabela de suficiência do resumo.md."""
    lo, hi = s["ic_frac_suficiente"]
    return f"| {nome} | {s['frac_suficiente']:.4f} [{lo:.4f}, {hi:.4f}] | {s['n_parcial']} | {s['n_insuficiente']} | {s['n_falha_parse']} |"


def gravar_confirmacao(diretorio: Path, df: pd.DataFrame, res: dict, custo: dict, nome: str) -> Path:
    """por_item.parquet, resumo.json (sem o custo), resumo.md (tabela P e tabela de suficiência) e custo.json (etapas na ordem de execução); devolve o diretório."""
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    df.to_parquet(diretorio / "por_item.parquet", index=False)
    (diretorio / "resumo.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (diretorio / "resumo.md").write_text(
        f"# {nome} ({res['variante']} sobre {res['degrau']}, N={res['n_itens']})\n\n{CABECALHO_PROMPTS}\n{linha_prompts(nome, res)}\n\n"
        f"{CABECALHO_SUFICIENCIA}\n{linha_suficiencia(nome, res['suficiencia'])}\n",
        encoding="utf-8",
    )
    (diretorio / "custo.json").write_text(json.dumps(custo, ensure_ascii=False, indent=2), encoding="utf-8")   # sem sort_keys: etapas na ordem de execução
    return diretorio


def executar_confirmacao(
    degrau: str,
    recuperador: Recuperador,
    itens: Sequence[ItemDataset],
    variante: VariantePrompt,
    clientes_agentes: Mapping[str, ClienteLLM],
    executor_juiz: ExecutorJuiz,
    cliente_auditor: ClienteLLM,
    cliente_suficiencia: ClienteLLM,
    *,
    saida: Path | None = None,
    b: int = B_BOOTSTRAP,
    limite: int | None = None,
    sobrescrever: bool = False,
    ambiente: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Contexto → agentes → juiz → auditor → suficiência → por_item/resumo/custo em `saida` (padrão runs/confirmacao/novo).

    Registrar uma vez: se `saida/resumo.json` existe e não há `sobrescrever`, tudo é recomputado dos artefatos
    congelados (nenhum modelo é chamado) e comparado ao golden — GoldenDivergente se |Δ adequacao_media| ou
    |Δ frac_suficiente| > TOLERANCIA_CANONICO, ou se o golden é de outro degrau/variante; nada é regravado
    (custo.json inclusive: é a medição da 1ª execução). `res["custo"]` sempre descreve a execução atual.
    """
    itens = list(itens)[:limite] if limite else list(itens)
    saida = Path(saida) if saida is not None else DIR_CONFIRMACAO / NOME_NOVO
    golden = ler_resumo(saida)
    if golden is not None and not sobrescrever and (golden.get("degrau"), golden.get("variante")) != (degrau, variante.nome):
        raise GoldenDivergente(
            f"{NOME_NOVO}: golden registrado sobre {golden.get('variante')}/{golden.get('degrau')}, pedido {variante.nome}/{degrau}; use --sobrescrever com decisão registrada"
        )
    ids = [it.item_id for it in itens]
    com_disp = {it.item_id for it in itens_com_dispositivo(itens)}
    n_ref = sum(it.has_doc_ref for it in itens)
    log(f"confirmacao: {len(itens)} itens (has_doc_ref={n_ref}, com dispositivo={len(com_disp)}) | variante={variante.nome} degrau={degrau}")
    saida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"item_id": it.item_id, "has_doc_ref": bool(it.has_doc_ref), "com_dispositivo": it.item_id in com_disp} for it in itens]
    ).to_parquet(saida / "itens.parquet", index=False)
    etapas: dict[str, dict] = {}

    def medir(nome: str, fn: Callable[[], object], **contagens: object) -> object:
        t0 = time.perf_counter()
        r = fn()
        etapas[nome] = {**contagens, "segundos": round(time.perf_counter() - t0, 1)}
        return r

    p_ctx = saida / "contexto.parquet"
    contexto: pd.DataFrame = medir(   # type: ignore[assignment]
        "contexto", lambda: recuperar_contexto(recuperador, itens, p_ctx, log=log), recuperados=_n_faltantes(p_ctx, ids)
    )
    respostas: dict[str, pd.DataFrame] = {}
    for aid, cli in clientes_agentes.items():
        p = saida / "agent_responses" / aid / "responses.parquet"
        geradas = _n_faltantes(p, ids)
        df_a: pd.DataFrame = medir(   # type: ignore[assignment]
            f"agente_{aid}", lambda cli=cli, aid=aid, p=p: gerar_respostas(variante, cli, aid, itens, contexto, p, log=log), geradas=geradas
        )
        novas = df_a.iloc[len(df_a) - geradas:] if geradas else df_a.iloc[0:0]
        etapas[f"agente_{aid}"].update(
            retries=int(novas["n_retries"].sum()), falhas_parse=int((~novas["valid_json"].astype(bool)).sum()),
            custo_usd=round(float(novas["cost_usd"].sum()), 4),   # 0 nos locais (sem cobrança de API); geração de API entra no total
        )
        respostas[aid] = df_a
    n0, c0 = len(executor_juiz.cache), executor_juiz.cache.custo_total()
    rotulos: pd.DataFrame = medir("juiz", lambda: julgar(itens, respostas, executor_juiz, saida / "rotulos.parquet", log=log))   # type: ignore[assignment]
    etapas["juiz"].update(chamadas=len(executor_juiz.cache) - n0, custo_usd=round(executor_juiz.cache.custo_total() - c0, 4))
    p_aud = saida / "auditoria_recusas.parquet"
    n_aud = _n_linhas(p_aud)
    auditoria: pd.DataFrame = medir(   # type: ignore[assignment]
        "auditor", lambda: auditar(itens, respostas, variante, contexto, cliente_auditor, p_aud, log=log)
    )
    novas = auditoria.iloc[n_aud:]
    etapas["auditor"].update(auditadas=len(novas), custo_usd=round(float(novas["cost_usd"].sum()), 4))
    p_suf = saida / "suficiencia_contexto.parquet"
    n_suf = _n_linhas(p_suf)
    suficiencia: pd.DataFrame = medir(   # type: ignore[assignment]
        "suficiencia", lambda: julgar_suficiencia(itens, contexto, cliente_suficiencia, p_suf, log=log)
    )
    novas = suficiencia.iloc[n_suf:]
    etapas["suficiencia"].update(
        julgados=int((~novas["contexto_vazio"].astype(bool)).sum()), vazios=int(novas["contexto_vazio"].astype(bool).sum()),
        custo_usd=round(float(novas["cost_usd"].sum()), 4),
    )

    df = por_item(itens, contexto, respostas, rotulos, auditoria)
    suf = suficiencia[["item_id", "veredicto_suficiencia"]].rename(columns={"veredicto_suficiencia": "suficiencia"})
    df = df.merge(suf, on="item_id", how="left")
    df["suficiencia"] = df["suficiencia"].fillna(-1).astype("int64")
    res = resumo_variante(df, list(clientes_agentes), b=b)
    res.update(variante=variante.nome, degrau=degrau, n_has_doc_ref=int(n_ref), n_com_dispositivo=len(com_disp))
    res["suficiencia"] = resumo_suficiencia(df, b=b)
    custo = {
        "n_itens": len(itens),
        "executado_em": agora_iso(),
        "etapas": etapas,
        "custo_usd_total": round(sum(float(e.get("custo_usd", 0.0)) for e in etapas.values()), 4),
        "palavras_medias": res["palavras_medias"],
        "n_chunks_medio": res["n_chunks_medio"],
        "ambiente": dict(ambiente or {}),
    }
    if golden is not None and not sobrescrever:
        res["delta_vs_golden"] = res["adequacao_media"] - golden["adequacao_media"]
        res["delta_vs_golden_suficiencia"] = res["suficiencia"]["frac_suficiente"] - golden["suficiencia"]["frac_suficiente"]
        if abs(res["delta_vs_golden"]) > TOLERANCIA_CANONICO or abs(res["delta_vs_golden_suficiencia"]) > TOLERANCIA_CANONICO:
            raise GoldenDivergente(
                f"{NOME_NOVO}: adequacao_media={res['adequacao_media']:.4f} vs golden {golden['adequacao_media']:.4f}; "
                f"frac_suficiente={res['suficiencia']['frac_suficiente']:.4f} vs golden {golden['suficiencia']['frac_suficiente']:.4f}"
            )
        res["custo"] = custo
        return res
    gravar_confirmacao(saida, df, res, custo, NOME_NOVO)
    res["custo"] = custo
    return res
