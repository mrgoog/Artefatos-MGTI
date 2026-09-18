"""Métricas sobre tabelas consolidadas do experimento."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_llm.persistencia.caminhos import caminho_limiar_fold, caminho_pesos_fold


def risco_seletivo(dados: pd.DataFrame) -> float:
    """Taxa de erro entre os itens respondidos (não abstidos) no DataFrame consolidado."""
    respondidos = dados[~dados["abstained"]]
    if len(respondidos) == 0:
        return float("nan")
    return float((1 - respondidos["z_system"]).mean())


def cobertura_empirica(dados: pd.DataFrame) -> float:
    """Fração de itens respondidos (não abstidos) no DataFrame consolidado."""
    if len(dados) == 0:
        return float("nan")
    return float((~dados["abstained"]).mean())


def risco_seletivo_por_estrato(dados: pd.DataFrame, valor_estrato: bool) -> float:
    """Risco seletivo filtrado por valor de has_doc_ref (True=com referência, False=sem)."""
    sub = dados[dados["has_doc_ref"] == valor_estrato]
    return risco_seletivo(sub)


def cobertura_por_estrato(dados: pd.DataFrame, valor_estrato: bool) -> float:
    """Cobertura empírica filtrada por valor de has_doc_ref."""
    sub = dados[dados["has_doc_ref"] == valor_estrato]
    return cobertura_empirica(sub)


def erro_calibracao_esperado(
    confiancas: np.ndarray | pd.Series,
    rotulos: np.ndarray | pd.Series,
    n_bins: int = 15,
    estrategia: str = "equal_width",
) -> float:
    """ECE: diferença ponderada entre confiança média e acurácia em bins de histograma."""
    conf = np.asarray(confiancas, dtype=float)
    rot = np.asarray(rotulos, dtype=float)
    if conf.shape != rot.shape:
        raise ValueError(f"Shape mismatch: confidences {conf.shape} vs labels {rot.shape}")
    n = len(conf)
    if n == 0:
        return float("nan")

    if estrategia == "equal_width":
        bordas = np.linspace(0.0, 1.0, n_bins + 1)
    elif estrategia == "quantile":
        bordas = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        bordas[0] = 0.0
        bordas[-1] = 1.0 + 1e-9
    else:
        raise ValueError(f"estrategia desconhecida: {estrategia!r}")

    ece = 0.0
    for m in range(n_bins):
        lo, hi = bordas[m], bordas[m + 1]
        if m == n_bins - 1:
            no_bin = (conf >= lo) & (conf <= hi)
        else:
            no_bin = (conf >= lo) & (conf < hi)
        contagem = no_bin.sum()
        if contagem == 0:
            continue
        acuracia = rot[no_bin].mean()
        conf_media = conf[no_bin].mean()
        ece += (contagem / n) * abs(acuracia - conf_media)
    return float(ece)


def ece_por_agente(
    dados_agente: pd.DataFrame,
    coluna_confianca: str,
    coluna_rotulo: str,
    n_bins: int = 15,
) -> float:
    """ECE para colunas específicas de confiança e rótulo dentro de um DataFrame."""
    return erro_calibracao_esperado(
        dados_agente[coluna_confianca],
        dados_agente[coluna_rotulo],
        n_bins=n_bins,
    )


def dados_diagrama_confiabilidade(
    confiancas: np.ndarray | pd.Series,
    rotulos: np.ndarray | pd.Series,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Gera DataFrame de bins (média de confiança vs. acurácia) para o reliability diagram."""
    conf = np.asarray(confiancas, dtype=float)
    rot = np.asarray(rotulos, dtype=float)
    bordas = np.linspace(0.0, 1.0, n_bins + 1)

    linhas = []
    for m in range(n_bins):
        lo, hi = bordas[m], bordas[m + 1]
        no_bin = (conf >= lo) & ((conf <= hi) if m == n_bins - 1 else (conf < hi))
        contagem = int(no_bin.sum())
        linhas.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": contagem,
                "avg_confidence": float(conf[no_bin].mean()) if contagem else float("nan"),
                "accuracy": float(rot[no_bin].mean()) if contagem else float("nan"),
            }
        )
    return pd.DataFrame(linhas)


def _coeficiente_variacao(media: float, dp: float) -> float:
    """CV = dp/|media|; retorna NaN quando média ≈ 0 (CV indefinido)."""
    if abs(media) < 1e-12:
        return float("nan")
    return float(dp / abs(media))


def estatisticas_estabilidade_pesos(
    base_dir: Path,
    experiment_id: str,
    fold_ks: list[int],
    c_min: float,
) -> dict[str, dict[str, float]]:
    """Agrega `weights.json` dos K folds em estatísticas por agente."""
    pesos_por_agente: dict[str, list[float]] = {}
    for k in fold_ks:
        path = caminho_pesos_fold(base_dir, experiment_id, k, c_min)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for aid, w in data["weights"].items():
            pesos_por_agente.setdefault(aid, []).append(float(w))

    out: dict[str, dict[str, float]] = {}
    for aid, vals in pesos_por_agente.items():
        arr = np.asarray(vals, dtype=float)
        media = float(arr.mean())
        dp = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[aid] = {
            "mean": media,
            "std": dp,
            "cv": _coeficiente_variacao(media, dp),
            "n_folds": int(len(arr)),
        }
    return out


def estatisticas_estabilidade_limiar(
    base_dir: Path,
    experiment_id: str,
    fold_ks: list[int],
    c_min: float,
) -> dict[str, float]:
    """Agrega `threshold.json` dos K folds em estatísticas de τ*."""
    taus: list[float] = []
    constraint_satisfied: list[bool] = []
    for k in fold_ks:
        path = caminho_limiar_fold(base_dir, experiment_id, k, c_min)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        taus.append(float(data["tau_star"]))
        constraint_satisfied.append(bool(data["constraint_satisfied"]))

    arr = np.asarray(taus, dtype=float)
    media = float(arr.mean()) if len(arr) else float("nan")
    dp = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": media,
        "std": dp,
        "cv": _coeficiente_variacao(media, dp),
        "n_folds": int(len(arr)),
        "prop_constraint_satisfied": (
            float(np.mean(constraint_satisfied)) if constraint_satisfied else float("nan")
        ),
    }
