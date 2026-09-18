"""Analisa a verificação humana exploratória do juiz LLM.

Cruza `data/validacao_humana/anotacoes.parquet` com `amostra.parquet` e produz, por estrato:
tabela de confusão, acordo bruto com IC exato (Clopper–Pearson), kappa de Cohen com IC
bootstrap, concordância condicionada ao rótulo automático e a lista de desacordos com
os comentários do anotador. Escreve `data/validacao_humana/resultados.md`.

A amostra é estratificada pelo rótulo do juiz e foi formada depois da escolha de no máximo
um par por item e da resolução de sobreposições entre estratos. Portanto, o acordo em cada
estrato descreve P[humano concorda | juiz disse X] no quadro reduzido; ele não estima a
acurácia global nos 2.129 julgamentos.

Uso:
    uv run python scripts/analisar_validacao_humana.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

PASTA = Path("data/validacao_humana")
ROT_SUB = {1: "adequada", 0: "inadequada", -1: "não sei"}
ROT_REC = {2: "justificada", 1: "evasiva", 0: "não é recusa", -1: "não sei"}
EXCLUSOES = {
    ("q_0065", "gemma4_e4b"): (
        "O anotador aplicou justificabilidade contextual a um par do estrato substantivo, "
        "em desacordo com o protocolo de comparação com a referência. A exclusão foi "
        "decidida durante a revisão metodológica posterior à anotação."
    ),
}


def ic_clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def kappa_cohen(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    n = len(a)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def kappa_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ks.append(kappa_cohen(a[idx], b[idx]))
    ks = np.array([k for k in ks if np.isfinite(k)])
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))) if len(ks) else (float("nan"), float("nan"))


def fmt(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=PASTA / "resultados.md")
    args = ap.parse_args()

    an = pd.read_parquet(PASTA / "anotacoes.parquet")
    am = pd.read_parquet(PASTA / "amostra.parquet")
    pop = am["estrato"].value_counts().to_dict()
    m = an.merge(
        am[["item_id", "agent_id", "ordem_sorteio", "z", "veredicto_recusa", "pergunta", "confidence_raw"]],
        on=["item_id", "agent_id"], how="left",
    )
    m["excluida_analise"] = [
        (item_id, agent_id) in EXCLUSOES
        for item_id, agent_id in zip(m["item_id"], m["agent_id"], strict=True)
    ]
    m_analise = m[~m["excluida_analise"]].copy()
    out: list[str] = ["# Verificação humana exploratória do juiz LLM — resultados\n"]
    out.append(f"Anotador único (autor). {len(m)} pares anotados de {len(am)} registros no quadro amostral; "
               f"{len(m_analise)} pares incluídos na análise. "
               "Nos pares substantivos, o cegamento não foi assegurado pela interface. "
               f"Gerado por `scripts/analisar_validacao_humana.py`.\n")

    out.append("## Exclusão documentada\n")
    for (item_id, agent_id), motivo in EXCLUSOES.items():
        out.append(f"- `{item_id} · {agent_id}`: {motivo} O registro bruto foi preservado, mas não participa das métricas.\n")

    # ---- Estratos substantivos: juiz binário -----------------------------------------
    sub = m_analise[m_analise["estrato"] != "recusa"].copy()
    sub_j = sub[sub["rotulo_humano"] >= 0]
    out.append("## Juiz binário de adequação (estratos `adequada` e `inadequada`)\n")
    out.append(f"n incluídos na análise = {len(sub)}; 'não sei julgar' = {(sub['rotulo_humano'] < 0).sum()}.\n")
    ct = pd.crosstab(sub["z"].map(ROT_SUB), sub["rotulo_humano"].map(ROT_SUB), rownames=["juiz"], colnames=["humano"])
    out.append(ct.to_markdown() + "\n")

    linhas = []
    for estrato, z in (("adequada", 1), ("inadequada", 0)):
        g = sub_j[sub_j["estrato"] == estrato]
        k = int((g["rotulo_humano"] == z).sum()); n = len(g)
        lo, hi = ic_clopper_pearson(k, n)
        nome = "P[humano=adequada | juiz=adequada]" if z == 1 else "P[humano=inadequada | juiz=inadequada]"
        linhas.append({"estrato": estrato, "n": n, "concordam": k, "acordo": f"{k/n:.1%}", "IC 95% exato": f"[{lo:.1%}; {hi:.1%}]", "interpretação": nome})
    out.append(pd.DataFrame(linhas).to_markdown(index=False) + "\n")
    out.append(
        f"Os {pop['adequada']} registros `adequada` e {pop['inadequada']} `inadequada` resultam da seleção "
        "de no máximo um par por item e da resolução de sobreposições. Eles não constituem a população "
        "dos 2.129 julgamentos; por isso, não se reconstrói uma acurácia global.\n"
    )
    a = sub_j["z"].to_numpy(); b = sub_j["rotulo_humano"].to_numpy()
    kap = kappa_cohen(a, b); lo, hi = kappa_bootstrap(a, b)
    out.append(f"Acordo bruto na amostra = {np.mean(a == b):.1%}; kappa de Cohen = **{fmt(kap)}** "
               f"[IC bootstrap 95%: {fmt(lo)}; {fmt(hi)}] (n = {len(a)}; amostra estratificada, kappa é descritivo).\n")

    # Direção do viés
    fp = sub_j[(sub_j["z"] == 1) & (sub_j["rotulo_humano"] == 0)]
    fn = sub_j[(sub_j["z"] == 0) & (sub_j["rotulo_humano"] == 1)]
    out.append(f"Desacordos: juiz=adequada/humano=inadequada = {len(fp)}; juiz=inadequada/humano=adequada = {len(fn)}. "
               + ("O viés observado é unidirecional: o juiz é mais severo que o anotador." if len(fp) == 0 and len(fn) > 0
                  else "O viés observado é unidirecional: o juiz é mais leniente que o anotador." if len(fn) == 0 and len(fp) > 0
                  else "Sem direção dominante." if len(fp) == len(fn) else "") + "\n")

    cats = sub["categorias_erro"].explode().dropna().value_counts()
    if len(cats):
        out.append("Categorias de problema apontadas pelo anotador (estratos substantivos):\n")
        out.append(cats.rename("n").to_markdown() + "\n")

    # ---- Recusas: auditor ------------------------------------------------------------
    rec = m_analise[m_analise["estrato"] == "recusa"].copy()
    rec_j = rec[rec["rotulo_humano"] >= 0]
    out.append("## Auditor de recusas (estrato `recusa`)\n")
    out.append(f"n anotados = {len(rec)} de {pop.get('recusa', 0)}; 'não sei julgar' = {(rec['rotulo_humano'] < 0).sum()}.\n")
    ct = pd.crosstab(rec["veredicto_recusa"].astype(int).map(ROT_REC), rec["rotulo_humano"].map(ROT_REC), rownames=["auditor"], colnames=["humano"])
    out.append(ct.to_markdown() + "\n")
    a = rec_j["veredicto_recusa"].astype(int).to_numpy(); b = rec_j["rotulo_humano"].to_numpy()
    k = int((a == b).sum()); n = len(a); lo_a, hi_a = ic_clopper_pearson(k, n)
    kap = kappa_cohen(a, b); lo, hi = kappa_bootstrap(a, b)
    out.append(f"Acordo bruto = {k}/{n} = **{k/n:.1%}** [IC exato 95%: {lo_a:.1%}; {hi_a:.1%}]; "
               f"kappa de Cohen = {fmt(kap)} [{fmt(lo)}; {fmt(hi)}].\n")
    hum_just = int((b == 2).sum())
    lo_j, hi_j = ic_clopper_pearson(hum_just, n)
    out.append(f"Fração de recusas julgadas **justificadas pelo humano** = {hum_just}/{n} = {hum_just/n:.1%} "
               f"[{lo_j:.1%}; {hi_j:.1%}]; o auditor automático marcou justificadas {int((a == 2).sum())}/{n} nesta amostra "
               f"e {int((am['veredicto_recusa'] == 2).sum())}/{pop.get('recusa', 0)} no total.\n")
    aud_evas = rec_j[a != 2]
    if len(aud_evas):
        out.append(f"Nos {len(aud_evas)} casos em que o auditor NÃO marcou 'justificada', o humano marcou: "
                   + ", ".join(f"{ROT_REC[int(v)]} ×{c}" for v, c in aud_evas["rotulo_humano"].value_counts().items()) + ".\n")

    # ---- Desacordos e comentários -----------------------------------------------------
    out.append("## Desacordos e comentários\n")
    m_analise["ref"] = np.where(
        m_analise["estrato"] == "recusa",
        m_analise["veredicto_recusa"].fillna(-9).astype(int),
        m_analise["z"],
    )
    m_analise["concorda"] = np.where(
        m_analise["rotulo_humano"] < 0,
        None,
        m_analise["rotulo_humano"] == m_analise["ref"],
    )
    des = m_analise[
        (m_analise["concorda"] == False)  # noqa: E712
        | (m_analise["comentario"].fillna("") != "")
        | (m_analise["comentario_pos_revelacao"].fillna("") != "")
    ]
    if des.empty:
        out.append("Nenhum desacordo nem comentário registrado.\n")
    for _, r in des.sort_values(["estrato", "ordem_sorteio"]).iterrows():
        rot = ROT_REC if r["estrato"] == "recusa" else ROT_SUB
        tag = "DESACORDO" if r["concorda"] is False else ("não sei" if r["rotulo_humano"] < 0 else "concorda")
        out.append(f"### {r['item_id']} · {r['agent_id']} · {r['estrato']} #{int(r['ordem_sorteio'])} — {tag}\n")
        out.append(f"- pergunta: {r['pergunta'][:200]}\n- juiz/auditor: **{rot.get(int(r['ref']), r['ref'])}** · humano: **{rot.get(int(r['rotulo_humano']))}** · confiança do agente: {r['confidence_raw']:.2f}")
        if isinstance(r["categorias_erro"], (list, np.ndarray)) and len(r["categorias_erro"]):
            out.append(f"- categorias: {', '.join(r['categorias_erro'])}")
        if r["comentario"]:
            out.append(f"- comentário: {r['comentario']}")
        if r["comentario_pos_revelacao"]:
            out.append(f"- pós-revelação: {r['comentario_pos_revelacao']}")
        out.append("")

    texto = "\n".join(out)
    args.saida.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
