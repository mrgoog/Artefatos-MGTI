"""Bootstrap BCa estratificado."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ensemble_llm.esquemas import ICBootstrap


def _quantis_bca(
    theta_hat: float,
    theta_boot: np.ndarray,
    theta_jack: np.ndarray,
    nivel_confianca: float,
) -> tuple[float, float]:
    """Calcula quantis BCa corrigidos por viés (z₀) e aceleração (a) via jackknife."""
    frac_abaixo = float((theta_boot < theta_hat).mean())
    eps = 0.5 / len(theta_boot)
    frac_abaixo = float(np.clip(frac_abaixo, eps, 1 - eps))
    z0 = norm.ppf(frac_abaixo)

    media_jack = theta_jack.mean()
    diff = media_jack - theta_jack
    num = (diff**3).sum()
    den = 6.0 * ((diff**2).sum() ** 1.5)
    a = float(num / den) if den != 0.0 else 0.0

    alpha = 1.0 - nivel_confianca
    z_lo = norm.ppf(alpha / 2.0)
    z_hi = norm.ppf(1.0 - alpha / 2.0)

    def ajustar(z_alpha: float) -> float:
        return float(norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha))))

    return ajustar(z_lo), ajustar(z_hi)


def ic_bootstrap_estratificado(
    dados: pd.DataFrame,
    fn_estatistica: Callable[[pd.DataFrame], float],
    coluna_estratificacao: str | None = None,
    n_reamostras: int = 10000,
    nivel_confianca: float = 0.95,
    metodo: str = "BCa",
    seed: int = 42,
) -> ICBootstrap:
    """IC bootstrap BCa ou percentil com reamostragem estratificada (Hesterberg 2015)."""
    if metodo not in ("BCa", "percentile"):
        raise ValueError(f"metodo deve ser 'BCa' ou 'percentile', recebeu {metodo!r}")
    if not (0.0 < nivel_confianca < 1.0):
        raise ValueError(
            f"nivel_confianca deve estar em (0,1), recebeu {nivel_confianca}"
        )
    if n_reamostras < 100:
        raise ValueError(f"n_reamostras deve ser >= 100, recebeu {n_reamostras}")

    rng = np.random.default_rng(seed)

    theta_hat = float(fn_estatistica(dados))
    if not np.isfinite(theta_hat):
        raise ValueError("fn_estatistica retornou NaN/inf no dado original")

    if coluna_estratificacao is None:
        indices_estratos = [np.arange(len(dados))]
    else:
        if coluna_estratificacao not in dados.columns:
            raise ValueError(
                f"Coluna de estratificação '{coluna_estratificacao}' não está em dados"
            )
        indices_estratos = [
            np.asarray(dados.index.get_indexer(dados[dados[coluna_estratificacao] == valor].index))
            for valor in dados[coluna_estratificacao].unique()
        ]

    theta_boot = np.empty(n_reamostras)
    n_descartadas = 0
    valores_dados = dados.reset_index(drop=True)

    for b in range(n_reamostras):
        indices_reamostra: list[int] = []
        for idx_estrato in indices_estratos:
            n = len(idx_estrato)
            escolhidos = rng.integers(0, n, size=n)
            indices_reamostra.extend(idx_estrato[escolhidos].tolist())
        reamostra = valores_dados.iloc[indices_reamostra].reset_index(drop=True)
        try:
            valor = float(fn_estatistica(reamostra))
        except (ZeroDivisionError, ValueError):
            valor = float("nan")
        if not np.isfinite(valor):
            n_descartadas += 1
        theta_boot[b] = valor

    validos = np.isfinite(theta_boot)
    theta_boot_validos = theta_boot[validos]
    if len(theta_boot_validos) < n_reamostras * 0.5:
        raise RuntimeError(
            f"Mais de 50% das reamostras descartadas (n_descartadas={n_descartadas} "
            f"de {n_reamostras}). IC não é confiável."
        )

    if metodo == "percentile":
        alpha = 1.0 - nivel_confianca
        ci_lo = float(np.quantile(theta_boot_validos, alpha / 2.0))
        ci_hi = float(np.quantile(theta_boot_validos, 1.0 - alpha / 2.0))
    else:
        n = len(dados)
        theta_jack = np.empty(n)
        for i in range(n):
            d_loo = valores_dados.drop(index=i).reset_index(drop=True)
            try:
                theta_jack[i] = float(fn_estatistica(d_loo))
            except (ZeroDivisionError, ValueError):
                theta_jack[i] = float("nan")
        theta_jack_validos = theta_jack[np.isfinite(theta_jack)]
        if len(theta_jack_validos) < n * 0.5:
            raise RuntimeError("Mais de 50% das jackknife-replicates descartadas; BCa inviável")

        p_lo, p_hi = _quantis_bca(
            theta_hat,
            theta_boot_validos,
            theta_jack_validos,
            nivel_confianca,
        )
        ci_lo = float(np.quantile(theta_boot_validos, p_lo))
        ci_hi = float(np.quantile(theta_boot_validos, p_hi))

    return ICBootstrap(
        point=theta_hat,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        confidence_level=nivel_confianca,
        method=metodo,  # type: ignore[arg-type]
        n_resamples=n_reamostras,
        n_dropped=n_descartadas,
    )
