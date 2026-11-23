"""Recomputações da Fase 0 do retrabalho do artigo 2 (custo zero, leitura pura).

Quatro camadas, todas sobre artefatos já persistidos em runs/<experiment_id>/:

  A) ρ (co-erro) global e POR ESTRATO do eixo de suficiência de contexto.
     O ρ publicado usa o produto das taxas marginais como referência — isto é,
     independência NÃO condicionada. Parte da co-ocorrência de erros pode vir da
     heterogeneidade de dificuldade dos itens, e não do acoplamento pelo RAG
     compartilhado. Condicionar no eixo de suficiência separa as duas coisas.

  B) ΔAURC com protocolo estatístico unificado: bootstrap PERCENTIL PAREADO
     ESTRATIFICADO por has_doc_ref, B=10.000, usando o módulo padrão do projeto
     (`ensemble_llm.avaliacao.bootstrap_estratificado`), e AURC com tratamento
     de EMPATE MÉDIO (o argsort estável desempata por ordem de linha).

  C) Winner's curse medido: viés condicional à seleção,
     E[c_bruta − z | agente selecionado] vs. E[c_bruta − z | não selecionado].

  D) Distribuição de tamanhos dos chunks do índice RAG, contra o limite
     declarado em configs (chunk_max_tokens).

Nada é sobrescrito. Saída em texto no stdout e, opcionalmente, JSON.

Uso:
    uv run python scripts/recomputacao_fase0.py --run phase2_local_full_715 --c-min 0.7
    uv run python scripts/recomputacao_fase0.py --run phase2_local_full_715 --camadas A C D
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ablacao_arquitetura import Resultado, carregar, montar_variantes  # noqa: E402

from ensemble_llm.avaliacao.bootstrap_estratificado import (  # noqa: E402
    ic_bootstrap_estratificado,
)

ROTULO_SUFIC = {2: "suficiente", 1: "parcial", 0: "insuficiente", -1: "falha do juiz"}


# --------------------------------------------------------------------------- #
# A) ρ global e por estrato de suficiência
# --------------------------------------------------------------------------- #


def _rho(mat_z: np.ndarray) -> dict[str, float]:
    """Razão observado/esperado da fração de itens em que TODOS os agentes erram.

    Referência de esperado = produto das taxas marginais de erro (independência).
    """
    erros = mat_z == 0
    taxas = erros.mean(0)
    observado = float(erros.all(1).mean())
    esperado = float(np.prod(taxas))
    return {
        "n": int(len(mat_z)),
        "taxas_erro": [float(t) for t in taxas],
        "fracao_todos_erram": observado,
        "esperado_sob_independencia": esperado,
        "rho": float(observado / esperado) if esperado > 0 else float("nan"),
    }


def _rho_ic(mat_z: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    """IC95% percentil do ρ por reamostragem de itens."""
    rng = np.random.default_rng(seed)
    n = len(mat_z)
    amostras = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        r = _rho(mat_z[idx])["rho"]
        if np.isfinite(r):
            amostras.append(r)
    if len(amostras) < b * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return float(lo), float(hi)


def camada_a(
    d: pd.DataFrame, agentes: list[str], run_dir: Path, b: int, seed: int
) -> dict:
    suf = pd.read_parquet(run_dir / "metrics" / "suficiencia_contexto.parquet")
    dd = d.merge(suf[["item_id", "veredicto_suficiencia"]], on="item_id", how="left")
    cols_z = [f"z_agent_{a}" for a in agentes]

    print("=" * 96)
    print("A) ρ — co-erro observado / esperado sob independência (referência marginal)")
    print("=" * 96)
    print(f"{'estrato':>16s} {'n':>5s} {'todos erram':>12s} {'esperado':>10s} "
          f"{'ρ':>7s} {'IC95%':>18s}")

    mat_z = dd[cols_z].to_numpy()
    glob = _rho(mat_z)
    lo, hi = _rho_ic(mat_z, b, seed)
    glob["ic95"] = [lo, hi]
    print(f"{'GLOBAL':>16s} {glob['n']:5d} {glob['fracao_todos_erram']:12.3f} "
          f"{glob['esperado_sob_independencia']:10.3f} {glob['rho']:7.3f} "
          f"[{lo:+.2f}, {hi:+.2f}]".rjust(0))

    por_estrato: dict[str, dict] = {}
    esperado_condicional = 0.0  # Σ_s (n_s/N) · Π_i p_i,s
    for v in sorted(dd["veredicto_suficiencia"].dropna().unique(), reverse=True):
        sub = dd[dd["veredicto_suficiencia"] == v]
        m = sub[cols_z].to_numpy()
        r = _rho(m)
        lo_s, hi_s = _rho_ic(m, b, seed)
        r["ic95"] = [lo_s, hi_s]
        nome = ROTULO_SUFIC.get(int(v), str(v))
        por_estrato[nome] = r
        esperado_condicional += (len(sub) / len(dd)) * r["esperado_sob_independencia"]
        print(f"{nome:>16s} {r['n']:5d} {r['fracao_todos_erram']:12.3f} "
              f"{r['esperado_sob_independencia']:10.3f} {r['rho']:7.3f} "
              f"[{lo_s:+.2f}, {hi_s:+.2f}]")

    rho_cond = (
        glob["fracao_todos_erram"] / esperado_condicional
        if esperado_condicional > 0
        else float("nan")
    )
    print("-" * 96)
    print(f"ρ marginal (publicado)            = {glob['rho']:.3f}")
    print(f"ρ condicional à suficiência       = {rho_cond:.3f}   "
          f"(esperado = Σ_s (n_s/N)·Π_i p_i,s = {esperado_condicional:.4f})")
    print("Leitura: a diferença entre os dois é a parcela do co-erro explicada por")
    print("heterogeneidade de dificuldade entre estratos; o que sobra em ρ condicional")
    print("é co-erro dentro de itens de mesma suficiência de contexto.\n")

    return {
        "global": glob,
        "por_estrato": por_estrato,
        "rho_condicional_suficiencia": rho_cond,
        "esperado_condicional": esperado_condicional,
    }


# --------------------------------------------------------------------------- #
# B) ΔAURC com empate médio + bootstrap percentil pareado estratificado
# --------------------------------------------------------------------------- #


def aurc_empate_medio(score: np.ndarray, z: np.ndarray) -> float:
    """AURC com empates resolvidos pela média do grupo (esperado sobre ordenações).

    Para um grupo de itens com score idêntico, a soma acumulada esperada sobre
    permutações aleatórias do grupo equivale a substituir cada rótulo pela média
    do grupo — o que torna a métrica invariante à ordem das linhas.
    """
    ordem = np.argsort(-score, kind="stable")
    s = score[ordem]
    y = z[ordem].astype(float)
    _, inv = np.unique(-s, return_inverse=True)
    somas = np.bincount(inv, weights=y)
    contagens = np.bincount(inv)
    y_medio = (somas / contagens)[inv]
    return float(np.mean(1.0 - np.cumsum(y_medio) / np.arange(1, len(y) + 1)))


def aurc_ordem_linha(score: np.ndarray, z: np.ndarray) -> float:
    """AURC como implementada hoje (desempate pela ordem das linhas)."""
    y = z[np.argsort(-score, kind="stable")]
    return float(np.mean(1.0 - np.cumsum(y) / np.arange(1, len(y) + 1)))


def _n_empatados(score: np.ndarray) -> int:
    _, contagens = np.unique(score, return_counts=True)
    return int(contagens[contagens > 1].sum())


def _delta_ic(
    a: Resultado,
    base: Resultado,
    estrato: np.ndarray,
    fn_aurc,
    b: int,
    seed: int,
):
    """IC95% percentil pareado, estratificado por has_doc_ref (padrão do projeto)."""
    dados = pd.DataFrame(
        {
            "has_doc_ref": estrato,
            "score_a": a.score,
            "z_a": a.z_vencedor,
            "score_base": base.score,
            "z_base": base.z_vencedor,
        }
    )

    def estatistica(df: pd.DataFrame) -> float:
        return fn_aurc(df["score_a"].to_numpy(), df["z_a"].to_numpy()) - fn_aurc(
            df["score_base"].to_numpy(), df["z_base"].to_numpy()
        )

    return ic_bootstrap_estratificado(
        dados=dados,
        fn_estatistica=estatistica,
        coluna_estratificacao="has_doc_ref",
        n_reamostras=b,
        metodo="percentile",
        seed=seed,
    )


def camada_b(
    d: pd.DataFrame, agentes: list[str], b: int, seed: int
) -> dict:
    variantes = montar_variantes(d, agentes)
    estrato = d["has_doc_ref"].to_numpy()
    por_nome = {v.nome.split(":")[0].split(" ")[0]: v for v in variantes}
    a0 = variantes[0]
    b0 = next(v for v in variantes if v.nome.startswith("B0"))

    print("=" * 118)
    print("B) AURC: ordem-de-linha vs. empate médio, e ΔAURC com bootstrap percentil")
    print(f"   pareado estratificado por has_doc_ref, B={b:,}".replace(",", "."))
    print("=" * 118)
    print(f"{'variante':46s} {'AURC linha':>11s} {'AURC empate':>12s} "
          f"{'itens empat.':>12s} {'ΔAURC vs A0 (IC95%)':>26s}")
    print("-" * 118)

    saida: dict[str, dict] = {}
    for v in variantes:
        chave = v.nome.split(" ")[0]
        linha = aurc_ordem_linha(v.score, v.z_vencedor)
        medio = aurc_empate_medio(v.score, v.z_vencedor)
        emp = _n_empatados(v.score)
        reg = {
            "aurc_ordem_linha": linha,
            "aurc_empate_medio": medio,
            "itens_empatados": emp,
        }
        if v is a0:
            txt = "—  (referência)"
        else:
            ic = _delta_ic(v, a0, estrato, aurc_empate_medio, b, seed)
            reg["delta_vs_A0"] = {
                "point": ic.point,
                "ci_lo": ic.ci_lo,
                "ci_hi": ic.ci_hi,
            }
            marca = " *" if ic.ci_hi < 0 else (" !" if ic.ci_lo > 0 else "  ")
            txt = f"{ic.point:+.3f} [{ic.ci_lo:+.3f}, {ic.ci_hi:+.3f}]{marca}"
        saida[chave] = reg
        print(f"{v.nome:46s} {linha:11.3f} {medio:12.3f} {emp:12d} {txt:>26s}")

    print("-" * 118)
    a5 = por_nome.get("A5")
    if a5 is not None:
        ic = _delta_ic(a5, b0, estrato, aurc_empate_medio, b, seed)
        saida["A5_vs_B0"] = {
            "point": ic.point,
            "ci_lo": ic.ci_lo,
            "ci_hi": ic.ci_hi,
        }
        print(f"contraste A5 − B0 (empate médio): {ic.point:+.3f} "
              f"[{ic.ci_lo:+.3f}, {ic.ci_hi:+.3f}]")
        ic_l = _delta_ic(a5, b0, estrato, aurc_ordem_linha, b, seed)
        saida["A5_vs_B0_ordem_linha"] = {
            "point": ic_l.point,
            "ci_lo": ic_l.ci_lo,
            "ci_hi": ic_l.ci_hi,
        }
        print(f"contraste A5 − B0 (ordem de linha): {ic_l.point:+.3f} "
              f"[{ic_l.ci_lo:+.3f}, {ic_l.ci_hi:+.3f}]")
        ic_a0 = _delta_ic(a5, a0, estrato, aurc_ordem_linha, b, seed)
        saida["A5_vs_A0_ordem_linha"] = {
            "point": ic_a0.point,
            "ci_lo": ic_a0.ci_lo,
            "ci_hi": ic_a0.ci_hi,
        }
        print(f"contraste A5 − A0 (ordem de linha): {ic_a0.point:+.3f} "
              f"[{ic_a0.ci_lo:+.3f}, {ic_a0.ci_hi:+.3f}]")
    print()
    return saida


# --------------------------------------------------------------------------- #
# C) Winner's curse medido
# --------------------------------------------------------------------------- #


def camada_c(d: pd.DataFrame, agentes: list[str], b: int, seed: int) -> dict:
    mat_b = d[[f"confidence_raw_{a}" for a in agentes]].to_numpy()
    mat_z = d[[f"z_agent_{a}" for a in agentes]].to_numpy().astype(float)
    escolhido = mat_b.argmax(1)
    n, k = mat_b.shape
    linhas = np.arange(n)

    vies = mat_b - mat_z  # c_bruta − z, por item × agente
    sel = np.zeros((n, k), dtype=bool)
    sel[linhas, escolhido] = True

    vies_sel = vies[sel]
    vies_nsel = vies[~sel]
    dif = float(vies_sel.mean() - vies_nsel.mean())

    rng = np.random.default_rng(seed)
    amostras = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        vs = vies[idx][sel[idx]].mean()
        vn = vies[idx][~sel[idx]].mean()
        amostras[i] = vs - vn
    lo, hi = np.percentile(amostras, [2.5, 97.5])

    c_max = mat_b.max(1)
    c_media = mat_b.mean(1)
    z_esc = mat_z[linhas, escolhido]

    print("=" * 96)
    print("C) Winner's curse — viés condicional à seleção (c_bruta − z)")
    print("=" * 96)
    print(f"agente selecionado (argmax c_bruta) : {vies_sel.mean():+.3f}")
    print(f"agentes não selecionados            : {vies_nsel.mean():+.3f}")
    print(f"diferença                           : {dif:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"{'  *' if lo > 0 else ''}")
    print("-" * 96)
    print(f"E[max(c_bruta)]                     : {c_max.mean():.3f}")
    print(f"E[media(c_bruta)]                   : {c_media.mean():.3f}")
    print(f"E[z do agente selecionado]          : {z_esc.mean():.3f}")
    print(f"superestimação do máximo            : {c_max.mean() - z_esc.mean():+.3f}")
    print(f"superestimação da média             : {c_media.mean() - z_esc.mean():+.3f}")
    print("Leitura: se a diferença é positiva e o IC não cruza zero, a confiança do")
    print("agente vencedor é sistematicamente mais otimista que a dos demais — o")
    print("winner's curse deixa de ser explicação compatível e passa a ser medido.\n")

    return {
        "vies_selecionado": float(vies_sel.mean()),
        "vies_nao_selecionado": float(vies_nsel.mean()),
        "diferenca": dif,
        "ic95": [float(lo), float(hi)],
        "e_max_c": float(c_max.mean()),
        "e_media_c": float(c_media.mean()),
        "e_z_selecionado": float(z_esc.mean()),
    }


# --------------------------------------------------------------------------- #
# D) Distribuição de tamanhos dos chunks
# --------------------------------------------------------------------------- #


def camada_d(limite_declarado: int) -> dict:
    base = Path("runs/_shared/rag")
    caminhos = sorted(base.glob("*/denso/chunks.parquet"))
    if not caminhos:
        print("D) chunks.parquet não encontrado em runs/_shared/rag/*/denso/\n")
        return {}
    c = pd.read_parquet(caminhos[0])
    col = next(
        (x for x in ("chunk_text", "text", "texto", "conteudo", "content")
         if x in c.columns),
        None,
    )
    if col is None:
        print(f"D) coluna de texto não identificada em {caminhos[0]}: {list(c.columns)}\n")
        return {"colunas": list(c.columns)}

    palavras = c[col].fillna("").str.split().str.len()
    caracteres = c[col].fillna("").str.len()
    acima = int((palavras > limite_declarado).sum())

    print("=" * 96)
    print(f"D) Chunks do índice — {caminhos[0]}")
    print("=" * 96)
    print(f"n chunks                    : {len(c):,}".replace(",", "."))
    print(f"limite declarado (config)   : {limite_declarado} tokens")
    print(f"palavras  — média/mediana   : {palavras.mean():.1f} / {palavras.median():.0f}")
    print(f"palavras  — p95/p99/máx     : {palavras.quantile(.95):.0f} / "
          f"{palavras.quantile(.99):.0f} / {palavras.max()}")
    print(f"caracteres — máx            : {caracteres.max():,}".replace(",", "."))
    print(f"chunks acima do limite      : {acima} ({100 * acima / len(c):.1f}%)")
    print()
    return {
        "arquivo": str(caminhos[0]),
        "n_chunks": int(len(c)),
        "limite_declarado": limite_declarado,
        "palavras_media": float(palavras.mean()),
        "palavras_mediana": float(palavras.median()),
        "palavras_p95": float(palavras.quantile(0.95)),
        "palavras_p99": float(palavras.quantile(0.99)),
        "palavras_max": int(palavras.max()),
        "caracteres_max": int(caracteres.max()),
        "chunks_acima_do_limite": acima,
        "pct_acima_do_limite": float(100 * acima / len(c)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--c-min", type=float, default=0.7)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--bootstrap-rho", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-limite", type=int, default=512)
    p.add_argument("--camadas", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--json-saida", type=str, default=None)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    d, agentes = carregar(run_dir, args.c_min)
    print(f"run={args.run}  c_min={args.c_min}  n={len(d)}  agentes={agentes}\n")

    res: dict[str, dict] = {"run": args.run, "c_min": args.c_min, "n": int(len(d))}
    if "A" in args.camadas:
        res["A_rho"] = camada_a(d, agentes, run_dir, args.bootstrap_rho, args.seed)
    if "B" in args.camadas:
        res["B_aurc"] = camada_b(d, agentes, args.bootstrap, args.seed)
    if "C" in args.camadas:
        res["C_winners_curse"] = camada_c(d, agentes, args.bootstrap_rho, args.seed)
    if "D" in args.camadas:
        res["D_chunks"] = camada_d(args.chunk_limite)

    if args.json_saida:
        Path(args.json_saida).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"JSON escrito em {args.json_saida}")


if __name__ == "__main__":
    main()
