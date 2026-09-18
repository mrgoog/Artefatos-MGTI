"""Ablação de arquitetura sobre artefatos de Nível 3 (custo zero).

Recomputa, out-of-fold, variantes de (peso, calibração, agregação) sobre as
respostas e rótulos já persistidos em runs/<experiment_id>/, e compara:

  - qualidade do ROTEAMENTO  : acurácia do vencedor a cobertura total
  - qualidade da ABSTENÇÃO   : AUROC e AURC do score de incerteza

Nenhuma chamada de agente ou de juiz é feita.

Uso:
    uv run python scripts/ablacao_arquitetura.py --run phase2_local_full_715
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado

# Regex de auto-recusa.
PADRAO_RECUSA = re.compile(
    r"não é possível responder|nao e possivel responder|não posso responder"
    r"|contexto (?:fornecido )?(?:é|e) insuficiente|não há informa|nao ha informa"
    r"|material fornecido não|não contém informa|não permite responder",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Resultado:
    """Saída de uma variante de arquitetura, item a item."""

    nome: str
    z_vencedor: np.ndarray  # rótulo do agente escolhido (roteamento)
    score: np.ndarray  # certeza; maior = responder (abstenção)
    escolhido: np.ndarray  # índice do agente escolhido, na ordem de `agentes`
    estrato: np.ndarray | None = None  # estrato do bootstrap (preenchido em montar_variantes)


def carregar(run_dir: Path, c_min: float = 0.8) -> tuple[pd.DataFrame, list[str]]:
    """Carrega o consolidado e anexa a flag de auto-recusa por agente."""
    d = pd.read_parquet(run_dir / f"consolidated_test_full_cmin={c_min}.parquet")
    agentes = [c.removeprefix("z_agent_") for c in d.columns if c.startswith("z_agent_")]
    for a in agentes:
        r = pd.read_parquet(run_dir / "agent_responses" / a / "responses.parquet")
        r[f"recusa_{a}"] = r["response_text"].fillna("").str.contains(PADRAO_RECUSA)
        d = d.merge(r[["item_id", f"recusa_{a}"]], on="item_id")
    manifesto = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    d.attrs["fold_permutation"] = manifesto["fold_permutation"]
    d.attrs["fit_sizes"] = {
        int(k): json.loads(
            (
                run_dir
                / "folds"
                / f"fold_k={int(k):02d}"
                / f"cmin={c_min:.1f}"
                / "weights.json"
            ).read_text(encoding="utf-8")
        )["fit_size"]
        for k in d["fold_k"].unique()
    }
    return d, sorted(agentes)


def _mascara_fit(d: pd.DataFrame, k: int) -> pd.Series:
    """Recupera D_fit do fold externo a partir da permutação persistida no manifesto."""
    permutacao = d.attrs.get("fold_permutation")
    fit_sizes = d.attrs.get("fit_sizes")
    if permutacao is None or fit_sizes is None:
        raise RuntimeError("Metadados de partição ausentes no DataFrame da ablação")
    restantes = [f for f in permutacao if f != k]
    candidatos = []
    for n_val_folds in range(1, len(restantes)):
        folds_fit = restantes[n_val_folds:]
        mascara = d["fold_k"].isin(folds_fit)
        if int(mascara.sum()) == int(fit_sizes[k]):
            candidatos.append(mascara)
    if len(candidatos) != 1:
        raise RuntimeError(
            f"Não foi possível identificar D_fit unicamente no fold {k}: "
            f"{len(candidatos)} candidatos para fit_size={fit_sizes[k]}"
        )
    return candidatos[0]


def _pesos_oof(
    d: pd.DataFrame, agentes: list[str], criterio: str
) -> dict[int, dict[str, float]]:
    """w_i por fold, estimado no D_fit original de cada partição tripartite.

    criterio:
      'alpha'          — w_i ∝ α_i (arquitetura atual)
      'alpha_sem_rec'  — w_i ∝ α_i com auto-recusas fora do denominador
      'informatividade'— w_i ∝ max(AUROC_i − 0.5, 0): o quanto a confiança de i
                         discrimina, e não o quanto i acerta
    """
    pesos: dict[int, dict[str, float]] = {}
    for k in d["fold_k"].unique():
        t = d[_mascara_fit(d, int(k))]
        bruto: dict[str, float] = {}
        for a in agentes:
            z, c, rec = t[f"z_agent_{a}"], t[f"confidence_raw_{a}"], t[f"recusa_{a}"]
            if criterio == "alpha":
                bruto[a] = float(z.mean())
            elif criterio == "alpha_sem_rec":
                bruto[a] = float(z[~rec].mean())
            elif criterio == "informatividade":
                bruto[a] = max(float(roc_auc_score(z, c)) - 0.5, 0.0)
            else:
                raise ValueError(f"critério desconhecido: {criterio!r}")
        soma = sum(bruto.values())
        pesos[int(k)] = {a: v / soma for a, v in bruto.items()}
    return pesos


def _colher(d: pd.DataFrame, agentes: list[str], escolhido: np.ndarray) -> np.ndarray:
    """z do agente escolhido, item a item."""
    mat_z = d[[f"z_agent_{a}" for a in agentes]].to_numpy()
    return mat_z[np.arange(len(d)), escolhido]


def montar_variantes(d: pd.DataFrame, agentes: list[str]) -> list[Resultado]:
    """Constrói as variantes de arquitetura, todas com scores out-of-fold."""
    # Reutiliza as calibrações OOF produzidas sobre D_fit no pipeline original. Reajustar
    # sobre todo o complemento do teste mudaria o protocolo de 70/20/10 para 90/0/10.
    mat_c = d[[f"confidence_calib_{a}" for a in agentes]].to_numpy()
    mat_b = d[[f"confidence_raw_{a}" for a in agentes]].to_numpy()  # bruta
    fold = d["fold_k"].to_numpy()
    out: list[Resultado] = []

    # --- variantes com w_i · ĉ_i e argmax (família da arquitetura atual) ---
    for nome, criterio in [
        ("A0 atual: w=alpha, argmax(w*c_calib)", "alpha"),
        ("A2 w=alpha sem recusas", "alpha_sem_rec"),
        ("A3 w=informatividade (AUROC-0.5)", "informatividade"),
    ]:
        pesos = _pesos_oof(d, agentes, criterio)
        mat_w = np.array([[pesos[int(k)][a] for a in agentes] for k in fold])
        mat_s = mat_w * mat_c
        esc = mat_s.argmax(1)
        out.append(Resultado(nome, _colher(d, agentes, esc), mat_s.max(1), esc))

    # A0 deve reproduzir exatamente o run materializado, salvo arredondamento numérico.
    a0 = out[0]
    agentes_a0 = np.asarray(agentes, dtype=object)[a0.escolhido]
    if not np.array_equal(agentes_a0, d["agent_chosen"].to_numpy()):
        raise RuntimeError("A0 da ablação não reproduz os vencedores do run materializado")
    if not np.allclose(a0.score, 1.0 - d["eta"].to_numpy(dtype=float)):
        raise RuntimeError("A0 da ablação não reproduz os scores do run materializado")

    # --- sem peso: a confiança calibrada é a própria estimativa de P(z=1) ---
    esc = mat_c.argmax(1)
    out.append(
        Resultado("A1 sem w: argmax(c_calib)", _colher(d, agentes, esc), mat_c.max(1), esc)
    )

    # --- agregação por média (sem winner's curse), roteamento por argmax ---
    out.append(
        Resultado(
            "A4 argmax(c_calib) + score=media(c_calib)",
            _colher(d, agentes, mat_c.argmax(1)),
            mat_c.mean(1),
            mat_c.argmax(1),
        )
    )
    out.append(
        Resultado(
            "A5 argmax(c_bruta) + score=media(c_bruta)",
            _colher(d, agentes, mat_b.argmax(1)),
            mat_b.mean(1),
            mat_b.argmax(1),
        )
    )
    out.append(
        Resultado(
            "A7 sem w: argmax(c_bruta) + score=max(c_bruta)",
            _colher(d, agentes, mat_b.argmax(1)),
            mat_b.max(1),
            mat_b.argmax(1),
        )
    )

    # --- score aprendido out-of-fold sobre features gratuitas ---
    mat_rec = d[[f"recusa_{a}" for a in agentes]].to_numpy().astype(float)
    feats = np.column_stack(
        [mat_c.mean(1), mat_c.min(1), mat_c.max(1), mat_c.std(1), mat_rec.sum(1)]
    )
    esc = mat_c.argmax(1)
    zv = _colher(d, agentes, esc)
    score = np.empty(len(d))
    for k in np.unique(fold):
        tr, te = _mascara_fit(d, int(k)).to_numpy(), fold == k
        lr = LogisticRegression(max_iter=1000).fit(feats[tr], zv[tr])
        score[te] = lr.predict_proba(feats[te])[:, 1]
    out.append(Resultado("A6 logistica(media,min,max,std,n_recusas)", zv, score, esc))

    # --- baseline degenerado: melhor agente fixo, sem roteamento ---
    # ATENÇÃO: a escolha do "melhor" usa a média sobre TODOS os rótulos, inclusive
    # os de teste — é seleção retrospectiva, e portanto uma linha de base OTIMISTA.
    # O contraste A5 − B0 é, por isso, conservador: compara o ensemble contra um
    # agente único escolhido com informação que ele não teria em produção.
    alphas = [d[f"z_agent_{a}"].mean() for a in agentes]
    melhor = int(np.argmax(alphas))
    out.append(
        Resultado(
            f"B0 sempre {agentes[melhor]} + score=c_bruta dele",
            d[f"z_agent_{agentes[melhor]}"].to_numpy(),
            mat_b[:, melhor],
            np.full(len(d), melhor, dtype=int),
        )
    )
    estrato = d["has_doc_ref"].to_numpy()
    return [replace(r, estrato=estrato) for r in out]


def _rotulos_empate_medio(score: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Rótulos ordenados por score decrescente, com empates resolvidos pela média.

    A confiança autodeclarada tem resolução baixa (∼15 níveis por agente), o que
    produz grupos de empate grandes — em `max(c_bruta)` um único grupo cobre 344
    dos 715 itens. Ordenar com `argsort` estável desempata pela ordem das linhas,
    tornando AURC e risco@cobertura dependentes de um artefato do arquivo. A soma
    acumulada ESPERADA sobre permutações aleatórias de um grupo empatado equivale
    a atribuir a cada item a média do grupo, o que restaura a invariância à ordem.
    """
    ordem = np.argsort(-score, kind="stable")
    s, y = score[ordem], z[ordem].astype(float)
    _, inv = np.unique(-s, return_inverse=True)
    somas = np.bincount(inv, weights=y)
    contagens = np.bincount(inv)
    return (somas / contagens)[inv]


def aurc(score: np.ndarray, z: np.ndarray) -> float:
    """Área sob a curva risco-cobertura (menor = melhor), invariante a empates."""
    y = _rotulos_empate_medio(score, z)
    return float(np.mean(1.0 - np.cumsum(y) / np.arange(1, len(y) + 1)))


def risco_em(score: np.ndarray, z: np.ndarray, cobertura: float) -> float:
    y = _rotulos_empate_medio(score, z)
    n = max(int(cobertura * len(y)), 1)
    return float(1.0 - y[:n].mean())


def bootstrap_delta(
    a: Resultado, base: Resultado, b: int, seed: int
) -> tuple[float, float]:
    """IC95% percentil pareado do ΔAURC (a − base), estratificado por referência.

    Negativo = a é melhor. Usa o módulo padrão de bootstrap do projeto
    (`ensemble_llm.avaliacao.bootstrap_estratificado`): a reamostragem preserva o
    tamanho dos estratos com/sem referência documental, e o par (a, base) é reamostrado
    com os MESMOS índices, de modo que o IC é do contraste e não da diferença de dois
    ICs independentes.
    """
    estrato = (
        a.estrato
        if a.estrato is not None
        else np.zeros(len(a.z_vencedor), dtype=int)
    )
    dados = pd.DataFrame(
        {
            "has_doc_ref": estrato,
            "score_a": a.score,
            "z_a": a.z_vencedor,
            "score_base": base.score,
            "z_base": base.z_vencedor,
        }
    )

    def delta(df: pd.DataFrame) -> float:
        return aurc(df["score_a"].to_numpy(), df["z_a"].to_numpy()) - aurc(
            df["score_base"].to_numpy(), df["z_base"].to_numpy()
        )

    ic = ic_bootstrap_estratificado(
        dados=dados,
        fn_estatistica=delta,
        coluna_estratificacao="has_doc_ref",
        n_reamostras=b,
        metodo="percentile",
        seed=seed,
    )
    return float(ic.ci_lo), float(ic.ci_hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--c-min", type=float, default=0.8)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    d, agentes = carregar(run_dir, args.c_min)
    print(f"run={args.run}  n={len(d)}  agentes={agentes}\n")

    vars_ = montar_variantes(d, agentes)
    base = vars_[0]

    print("=" * 108)
    print(f"{'arquitetura':44s} {'acur.@cob=1':>11s} {'AUROC':>7s} {'AURC':>7s} "
          f"{'risco@.3':>9s} {'risco@.5':>9s} {'ΔAURC vs A0 (IC95%)':>24s}")
    print("=" * 108)
    for v in vars_:
        z, s = v.z_vencedor, v.score
        d_aurc = aurc(s, z) - aurc(base.score, base.z_vencedor)
        if v is base:
            ic = "—  (referência)"
        else:
            lo, hi = bootstrap_delta(v, base, args.bootstrap, args.seed)
            marca = " *" if hi < 0 else (" !" if lo > 0 else "  ")
            ic = f"{d_aurc:+.3f} [{lo:+.3f},{hi:+.3f}]{marca}"
        print(f"{v.nome:44s} {z.mean():11.3f} {roc_auc_score(z, s):7.3f} {aurc(s, z):7.3f} "
              f"{risco_em(s, z, 0.3):9.3f} {risco_em(s, z, 0.5):9.3f} {ic:>24s}")
    print("=" * 108)
    mat_z = d[[f"z_agent_{a}" for a in agentes]].to_numpy()
    print(f"referência: oráculo any-correct@3 = {mat_z.max(1).mean():.3f} | "
          f"melhor agente isolado = {mat_z.mean(0).max():.3f}")
    print("*  = melhor que A0 (IC95% do ΔAURC não cruza zero)")


if __name__ == "__main__":
    main()
