"""Comparação pareada N=200 (DESIGN §2.3): pipeline atual (produção, lida da origem e derivada com o MESMO harness) × pipeline novo (runs/confirmacao/novo); Δ por item com IC bootstrap percentil estratificado; tabela de custo. Só CPU, zero chamadas de modelo."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.artefatos import ler_resumo
from rag2.bootstrap import ic_percentil
from rag2.config import ConfigRag2
from rag2.confirmacao import DIR_CONFIRMACAO, NOME_NOVO
from rag2.constantes import (
    AGENTE_REFERENCIA,
    AGENTES,
    B_BOOTSTRAP,
    MAX_TOKENS_JUIZ,
    MODELO_JUIZ,
    RUN_PRODUCAO,
    TOLERANCIA_CANONICO,
)
from rag2.dados import itens_com_dispositivo
from rag2.escada import GoldenDivergente
from rag2.geracao import COLUNAS_CONTEXTO, julgar, por_item, preparar_cache_juiz
from rag2.metricas import Casador, avaliar_itens

NOME_ATUAL = "atual"
"""runs/confirmacao/atual/ — só o derivado rotulos.parquet (z da produção por prompt_hash); o resto da produção é lido da origem."""

METRICAS_PAREADAS = (
    "recall_passagem", "recall_arquivo", "z_media", "z_qwen35_9b", "z_granite41_8b", "z_gemma4_e4b", "z_b0",
    "recusa", "evasiva", "falha_parse", "suficiente", "insuficiente", "palavras", "n_chunks",
)
"""Ordem das linhas de comparacao.parquet; cada métrica tem colunas <m>_atual e <m>_novo em pareado.parquet."""
METRICAS_151 = ("recall_passagem", "recall_arquivo")
"""Só nos itens com dispositivo (com_dispositivo); NaN em pareado.parquet para os demais 49."""
COLUNAS_COMPARACAO = ("metrica", "n", "atual", "novo", "delta", "ic_lo", "ic_hi")
"""Schema de comparacao.parquet; delta = média por item de (novo − atual); IC percentil pareado do delta."""

CABECALHO_COMPARACAO = "| métrica | n | atual | novo | Δ novo − atual [IC95] |\n|---|---:|---:|---:|---:|"


class RotuloAusente(RuntimeError):
    """O juiz seria chamado: um par (item, resposta) da produção não está no cache — a produção deve estar 100 % cacheada."""


class ClienteSomenteCache:
    """Cliente do juiz que NUNCA chama a API: mesmos model/temperature/max_tokens da produção (= mesmo prompt_hash em ExecutorJuiz.julgar); invoke levanta RotuloAusente."""

    model = MODELO_JUIZ
    temperature = 0.0
    max_tokens = MAX_TOKENS_JUIZ

    def invoke(self, prompt: str, *, system_prompt: str | None = None):
        raise RotuloAusente("miss no cache do juiz: a produção deveria estar 100 % cacheada (chamada proibida em `confirmacao comparar`)")

    def validate_model(self) -> None:
        return None


def artefatos_producao(cfg: ConfigRag2) -> dict[str, Path]:
    """Caminhos da origem que a comparação lê (só leitura, arch §1); `faltam = [p for p in .values() if not p.exists()]` é a guarda."""
    P = cfg.dir_run(RUN_PRODUCAO)
    saida = {f"responses_{aid}": P / "agent_responses" / aid / "responses.parquet" for aid in AGENTES}
    saida.update(
        cache_juiz=P / "judge_labels" / MODELO_JUIZ.replace("/", "_") / "cache.parquet",
        recall=P / "metrics" / "recall_passagem.parquet",
        auditoria=P / "metrics" / "auditoria_recusas.parquet",
        suficiencia=P / "metrics" / "suficiencia_contexto.parquet",
        chunks=cfg.dir_indice_producao() / "denso" / "chunks.parquet",
    )
    return saida


def respostas_producao(cfg: ConfigRag2, itens: Sequence[ItemDataset]) -> dict[str, pd.DataFrame]:
    """responses.parquet de cada agente da produção, só os itens dados, na ordem de AGENTES (mesma ordem de colunas do por_item do novo)."""
    ids = {it.item_id for it in itens}
    art = artefatos_producao(cfg)
    return {aid: pd.read_parquet(art[f"responses_{aid}"]).pipe(lambda d: d[d["item_id"].isin(ids)].reset_index(drop=True)) for aid in AGENTES}


def contexto_producao(cfg: ConfigRag2, referencia: pd.DataFrame) -> pd.DataFrame:
    """Contexto servido na produção, com texto (schema COLUNAS_CONTEXTO): retrieved_chunk_ids do agente de referência (idêntico nos três — medido) + denso/chunks.parquet lido direto (sem carregar o BM25); combined_rank = posição 1-based.

    Um chunk_id servido que não esteja no chunks.parquet levanta KeyError (ruidoso, de propósito: o índice de produção deve conter tudo o que serviu — medido 0 ausentes nos 200), ao contrário do _contexto da origem, que pula em silêncio.
    """
    ch = pd.read_parquet(artefatos_producao(cfg)["chunks"], columns=["chunk_id", "source_doc", "text"])
    texto = dict(zip(ch["chunk_id"], ch["text"], strict=True))
    doc = dict(zip(ch["chunk_id"], ch["source_doc"], strict=True))
    linhas = [
        {"item_id": r.item_id, "combined_rank": k, "chunk_id": str(c), "source_doc": doc[str(c)], "text": texto[str(c)]}
        for r in referencia.itertuples(index=False)
        for k, c in enumerate(r.retrieved_chunk_ids, 1)
    ]
    return pd.DataFrame(linhas, columns=list(COLUNAS_CONTEXTO))


def por_item_producao(
    cfg: ConfigRag2, itens: Sequence[ItemDataset], caminho_rotulos: Path, *, log: Callable[[str], None] = print
) -> pd.DataFrame:
    """por_item do harness sobre a produção: mesmas colunas do por_item.parquet do novo. z via `julgar` com ClienteSomenteCache (prompt_hash → cache; miss = RotuloAusente); auditoria e suficiência lidas da origem; contexto de contexto_producao. Grava/retoma caminho_rotulos."""
    ids = {it.item_id for it in itens}
    art = artefatos_producao(cfg)
    respostas = respostas_producao(cfg, itens)
    contexto = contexto_producao(cfg, respostas[AGENTE_REFERENCIA])
    executor = ExecutorJuiz(client=ClienteSomenteCache(), cache=preparar_cache_juiz(cfg, log=log))   # type: ignore[arg-type]
    n0 = len(executor.cache)
    rotulos = julgar(itens, respostas, executor, caminho_rotulos, log=log)
    if len(executor.cache) != n0:
        raise RotuloAusente(f"o cache do juiz cresceu {len(executor.cache) - n0} ao reconstruir a produção — nunca deveria")
    auditoria = pd.read_parquet(art["auditoria"])
    auditoria = auditoria[auditoria["item_id"].isin(ids)].reset_index(drop=True)
    df = por_item(itens, contexto, respostas, rotulos, auditoria)
    suf = pd.read_parquet(art["suficiencia"], columns=["item_id", "veredicto_suficiencia"]).rename(columns={"veredicto_suficiencia": "suficiencia"})
    df = df.merge(suf, on="item_id", how="left")
    df["suficiencia"] = df["suficiencia"].fillna(-1).astype("int64")
    return df


def recall_pareado(cfg: ConfigRag2, itens: Sequence[ItemDataset], contexto_novo: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por item com dispositivo (ordem de `itens`): recall_{passagem,arquivo}_{atual,novo} e n_esperados. Atual = canônico metrics/recall_passagem.parquet; novo = avaliar_itens com um Casador construído só de contexto_novo (≡ R3 por_item nesses itens — oráculo)."""
    avaliados = [a for a in itens_com_dispositivo(list(itens))]
    ids = [a.item_id for a in avaliados]
    canon = pd.read_parquet(artefatos_producao(cfg)["recall"]).set_index("item_id").loc[ids]
    casador = Casador(dict(zip(contexto_novo["chunk_id"], contexto_novo["text"], strict=True)), dict(zip(contexto_novo["chunk_id"], contexto_novo["source_doc"], strict=True)))
    servidos = {i: g.sort_values("combined_rank")["chunk_id"].tolist() for i, g in contexto_novo.groupby("item_id")}
    novo = avaliar_itens(avaliados, servidos, casador).set_index("item_id").loc[ids]
    if not (canon["n_esperados"].to_numpy() == novo["n_esperados"].to_numpy()).all():
        raise ValueError("n_esperados difere entre o canônico e avaliar_itens — definição de dispositivo divergente")
    return pd.DataFrame(
        {
            "item_id": ids,
            "recall_passagem_atual": canon["recall_passagem"].to_numpy(dtype=float),
            "recall_passagem_novo": novo["recall_passagem"].to_numpy(dtype=float),
            "recall_arquivo_atual": canon["recall_arquivo"].to_numpy(dtype=float),
            "recall_arquivo_novo": novo["recall_arquivo"].to_numpy(dtype=float),
            "n_esperados": canon["n_esperados"].to_numpy(dtype=int),
        }
    )


def melhor_agente_fixo(por_item_atual: pd.DataFrame) -> str:
    """B0: o agente de maior adequação média no lado ATUAL (a produção, o baseline), fixado para os dois lados; empate → ordem de AGENTES."""
    medias = {aid: float(por_item_atual[f"z_{aid}"].mean()) for aid in AGENTES}
    return max(AGENTES, key=lambda aid: medias[aid])   # max é estável: o 1º em AGENTES vence o empate


def _lado(df: pd.DataFrame, tag: str, b0: str) -> pd.DataFrame:
    """As 12 métricas por item de um lado (colunas <m>_<tag>), a partir de um por_item (novo ou atual); os 2 recalls entram em parear."""
    return pd.DataFrame(
        {
            "item_id": df["item_id"],
            f"z_media_{tag}": df["z_media"].astype(float),
            **{f"z_{aid}_{tag}": df[f"z_{aid}"].astype(float) for aid in AGENTES},
            f"z_b0_{tag}": df[f"z_{b0}"].astype(float),
            f"recusa_{tag}": df["recusa_media"].astype(float),
            f"evasiva_{tag}": df["evasiva_media"].astype(float),
            f"falha_parse_{tag}": 1.0 - df["parse_media"].astype(float),
            f"suficiente_{tag}": (df["suficiencia"] == 2).astype(float),
            f"insuficiente_{tag}": (df["suficiencia"] == 0).astype(float),
            f"palavras_{tag}": df["palavras"].astype(float),
            f"n_chunks_{tag}": df["n_chunks"].astype(float),
        }
    )


def parear(atual: pd.DataFrame, novo: pd.DataFrame, recall: pd.DataFrame, b0: str) -> pd.DataFrame:
    """pareado: item_id, has_doc_ref, <m>_atual ×12, <m>_novo ×12, recall_* ×4 (NaN sem dispositivo), com_dispositivo — na ordem de `atual` (= a de `novo`, exigida)."""
    if atual["item_id"].tolist() != novo["item_id"].tolist():
        raise ValueError("atual e novo não têm os mesmos itens na mesma ordem")
    par = atual[["item_id", "has_doc_ref"]].merge(_lado(atual, "atual", b0), on="item_id").merge(_lado(novo, "novo", b0), on="item_id")
    par = par.merge(recall.drop(columns=["n_esperados"]), on="item_id", how="left")
    par["com_dispositivo"] = par["item_id"].isin(recall["item_id"])
    return par


def comparar(pareado: pd.DataFrame, *, b: int = B_BOOTSTRAP) -> pd.DataFrame:
    """Uma linha por métrica de METRICAS_PAREADAS: n, médias atual/novo, delta = média(novo − atual) e IC percentil pareado do delta (reamostra itens, estratos has_doc_ref, semente 42); recall só em com_dispositivo."""
    linhas = []
    for m in METRICAS_PAREADAS:
        d = pareado[pareado["com_dispositivo"]] if m in METRICAS_151 else pareado
        d = d.assign(delta=d[f"{m}_novo"] - d[f"{m}_atual"])
        ic = ic_percentil(d, lambda x: float(x["delta"].mean()), b=b, coluna_estrato="has_doc_ref")
        linhas.append(
            {
                "metrica": m, "n": len(d),
                "atual": float(d[f"{m}_atual"].mean()), "novo": float(d[f"{m}_novo"].mean()),
                "delta": float(d["delta"].mean()), "ic_lo": float(ic.ci_lo), "ic_hi": float(ic.ci_hi),
            }
        )
    return pd.DataFrame(linhas, columns=list(COLUNAS_COMPARACAO))


def custo_comparado(cfg: ConfigRag2, itens: Sequence[ItemDataset], custo_novo: Mapping) -> dict:
    """{'atual': {...}, 'novo': {...}} com chamadas e US$ por etapa (agentes, juiz, auditor, suficiencia) e custo_usd_total; atual do cache DA ORIGEM (595 rótulos nos 200) + metrics/*; novo de custo.json (segundos só aqui — a produção não os registrou)."""
    ids = {it.item_id for it in itens}
    art = artefatos_producao(cfg)
    oc = pd.read_parquet(art["cache_juiz"], columns=["item_id", "cost_usd"])
    oc = oc[oc["item_id"].isin(ids)]
    aud = pd.read_parquet(art["auditoria"], columns=["item_id", "cost_usd"])
    aud = aud[aud["item_id"].isin(ids)]
    suf = pd.read_parquet(art["suficiencia"], columns=["item_id", "cost_usd"])
    suf = suf[suf["item_id"].isin(ids)]
    e = custo_novo["etapas"]
    atual = {
        "agentes": {"chamadas": len(AGENTES) * len(itens), "custo_usd": 0.0},
        "juiz": {"chamadas": len(oc), "custo_usd": round(float(oc["cost_usd"].sum()), 4)},
        "auditor": {"chamadas": len(aud), "custo_usd": round(float(aud["cost_usd"].sum()), 4)},
        "suficiencia": {"chamadas": len(suf), "custo_usd": round(float(suf["cost_usd"].sum()), 4)},
        "segundos": None,
    }
    novo = {
        "agentes": {"chamadas": int(sum(e[f"agente_{aid}"]["geradas"] for aid in AGENTES)), "custo_usd": 0.0, "retries": int(sum(e[f"agente_{aid}"]["retries"] for aid in AGENTES))},
        "juiz": {"chamadas": int(e["juiz"]["chamadas"]), "custo_usd": float(e["juiz"]["custo_usd"])},
        "auditor": {"chamadas": int(e["auditor"]["auditadas"]), "custo_usd": float(e["auditor"]["custo_usd"])},
        "suficiencia": {"chamadas": int(e["suficiencia"]["julgados"]), "custo_usd": float(e["suficiencia"]["custo_usd"])},
        "segundos": {k: float(v["segundos"]) for k, v in e.items()},
    }
    for lado in (atual, novo):
        lado["custo_usd_total"] = round(sum(v["custo_usd"] for v in lado.values() if isinstance(v, dict) and "custo_usd" in v), 4)
    return {"atual": atual, "novo": novo}


def linha_comparacao(r: Mapping) -> str:
    """Linha da tabela de comparação (4 casas; Δ e IC com sinal)."""
    return f"| {r['metrica']} | {r['n']} | {r['atual']:.4f} | {r['novo']:.4f} | {r['delta']:+.4f} [{r['ic_lo']:+.4f}, {r['ic_hi']:+.4f}] |"


def tabela_custo_md(custo: Mapping, medias: Mapping) -> str:
    """custo.md: chamadas e US$ por etapa nos dois lados, palavras/chunks por item, segundos de parede por etapa (só novo)."""
    a, n = custo["atual"], custo["novo"]
    linhas = ["| etapa | atual: chamadas | atual: US$ | novo: chamadas | novo: US$ | novo: segundos |", "|---|---:|---:|---:|---:|---:|"]
    seg = n["segundos"]
    seg_agentes = sum(v for k, v in seg.items() if k.startswith("agente_"))
    for etapa, s in (("agentes", seg_agentes), ("juiz", seg.get("juiz")), ("auditor", seg.get("auditor")), ("suficiencia", seg.get("suficiencia"))):
        linhas.append(f"| {etapa} | {a[etapa]['chamadas']} | {a[etapa]['custo_usd']:.4f} | {n[etapa]['chamadas']} | {n[etapa]['custo_usd']:.4f} | {s:.0f} |")
    linhas.append(f"| contexto (recuperação) | — | — | — | — | {seg.get('contexto', 0.0):.0f} |")
    linhas.append(f"| **total US$** | | **{a['custo_usd_total']:.4f}** | | **{n['custo_usd_total']:.4f}** | {sum(seg.values()):.0f} |")
    return (
        "# Custo — atual × novo (N=200)\n\n"
        f"palavras/item: atual {medias['palavras_atual']:.1f} · novo {medias['palavras_novo']:.1f} | chunks/item: atual {medias['n_chunks_atual']:.2f} · novo {medias['n_chunks_novo']:.2f}\n\n"
        + "\n".join(linhas)
        + f"\n\nAgentes locais (US$ 0); novo com {n['agentes']['retries']} retries de parse. Segundos = parede da 1ª execução de `rag2 confirmacao executar` (custo.json); a produção não registrou tempo por etapa.\n"
    )


def gravar_comparacao(saida: Path, pareado: pd.DataFrame, comp: pd.DataFrame, res: Mapping) -> Path:
    """pareado.parquet, comparacao.parquet, comparacao.json (golden; sem sort_keys — ordem de METRICAS_PAREADAS), comparacao.md, custo.md."""
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    pareado.to_parquet(saida / "pareado.parquet", index=False)
    comp.to_parquet(saida / "comparacao.parquet", index=False)
    (saida / "comparacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    rb = res["revisao_b3"]
    lo, hi = rb["ic"]
    (saida / "comparacao.md").write_text(
        f"# Comparação pareada N={res['n_itens']}: atual ({res['atual']}) × novo ({res['novo']})\n\n"
        f"Itens: {res['n_itens']} (has_doc_ref {res['n_has_doc_ref']}; com dispositivo {res['n_com_dispositivo']} — recall só nestes). "
        f"B0 = {res['b0']} (melhor agente fixo, escolhido na produção). IC95 percentil pareado do Δ, estratificado por has_doc_ref, B = {res['b_bootstrap']}, semente 42.\n\n"
        f"{CABECALHO_COMPARACAO}\n" + "\n".join(linha_comparacao(r) for r in res["metricas"]) + "\n\n"
        f"Condição de revisão (B3 waived, decisions 2026-09-13): Δ adequação média = {rb['delta_adequacao_media']:+.4f} [{lo:+.4f}, {hi:+.4f}] — "
        + ("IC exclui 0 por cima: **significativamente melhor — revisar a decisão B3**" if rb["significativamente_melhor"] else "IC inclui 0: **não** significativamente melhor; a decisão B3 permanece")
        + ".\n",
        encoding="utf-8",
    )
    (saida / "custo.md").write_text(tabela_custo_md(res["custo"], res["medias"]), encoding="utf-8")
    return saida


def _ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def registrar_comparacao(
    atual: pd.DataFrame,
    novo: pd.DataFrame,
    recall: pd.DataFrame,
    custo: Mapping,
    saida: Path,
    *,
    nome_novo: str,
    nome_atual: str = RUN_PRODUCAO,
    b: int = B_BOOTSTRAP,
    sobrescrever: bool = False,
) -> dict:
    """Pareia, compara, avalia a condição de revisão e grava em `saida` — registrar uma vez.

    Se `saida/comparacao.json` existe e não há `sobrescrever`: tudo é recomputado (determinístico) e comparado ao
    golden — GoldenDivergente se |Δ delta| > TOLERANCIA_CANONICO em z_media ou recall_passagem, ou se o B0 mudou;
    nada é regravado. Puro (sem I/O da origem): testável com fakes.
    """
    saida = Path(saida)
    golden = _ler_json(saida / "comparacao.json")
    b0 = melhor_agente_fixo(atual)
    pareado = parear(atual, novo, recall, b0)
    comp = comparar(pareado, b=b)
    zm = comp.set_index("metrica").loc["z_media"]
    res: dict = {
        "n_itens": len(pareado),
        "n_has_doc_ref": int(pareado["has_doc_ref"].sum()),
        "n_com_dispositivo": int(pareado["com_dispositivo"].sum()),
        "atual": nome_atual,
        "novo": nome_novo,
        "b0": b0,
        "b_bootstrap": int(b),
        "metricas": comp.to_dict("records"),
        "medias": {c: float(pareado[c].mean()) for c in ("palavras_atual", "palavras_novo", "n_chunks_atual", "n_chunks_novo")},
        "custo": dict(custo),
        "revisao_b3": {"delta_adequacao_media": float(zm["delta"]), "ic": [float(zm["ic_lo"]), float(zm["ic_hi"])], "significativamente_melhor": bool(zm["ic_lo"] > 0)},
    }
    if golden is not None and not sobrescrever:
        g = {r["metrica"]: r for r in golden["metricas"]}
        c = comp.set_index("metrica")
        res["delta_vs_golden"] = {m: float(c.loc[m, "delta"] - g[m]["delta"]) for m in ("z_media", "recall_passagem")}
        if any(abs(v) > TOLERANCIA_CANONICO for v in res["delta_vs_golden"].values()) or golden.get("b0") != b0:
            raise GoldenDivergente(
                f"comparacao: delta z_media={c.loc['z_media', 'delta']:+.4f} vs golden {g['z_media']['delta']:+.4f}; "
                f"recall_passagem={c.loc['recall_passagem', 'delta']:+.4f} vs {g['recall_passagem']['delta']:+.4f}; b0={b0} vs {golden.get('b0')}"
            )
        return res
    gravar_comparacao(saida, pareado, comp, res)
    return res


def executar_comparacao(
    cfg: ConfigRag2,
    itens: Sequence[ItemDataset],
    *,
    dir_novo: Path | None = None,
    saida: Path | None = None,
    b: int = B_BOOTSTRAP,
    sobrescrever: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Lê o novo (por_item, contexto, custo.json, resumo.json), reconstrói o atual da origem (zero chamadas), pareia e registra em `saida` (padrão runs/confirmacao); atual/rotulos.parquet fica em `saida/atual/`."""
    dir_novo = Path(dir_novo) if dir_novo is not None else DIR_CONFIRMACAO / NOME_NOVO
    saida = Path(saida) if saida is not None else DIR_CONFIRMACAO
    resumo_novo = ler_resumo(dir_novo)
    if resumo_novo is None:
        raise FileNotFoundError(f"confirmação '{NOME_NOVO}' não registrada em {dir_novo}: rode `rag2 confirmacao executar`")
    novo = pd.read_parquet(dir_novo / "por_item.parquet")
    contexto_novo = pd.read_parquet(dir_novo / "contexto.parquet")
    custo_novo = json.loads((dir_novo / "custo.json").read_text(encoding="utf-8"))
    atual = por_item_producao(cfg, itens, saida / NOME_ATUAL / "rotulos.parquet", log=log)
    recall = recall_pareado(cfg, itens, contexto_novo)
    custo = custo_comparado(cfg, itens, custo_novo)
    return registrar_comparacao(
        atual, novo, recall, custo, saida, nome_novo=f"{resumo_novo['variante']}/{resumo_novo['degrau']}", b=b, sobrescrever=sobrescrever,
    )
