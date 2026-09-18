"""IC bootstrap percentil (B=10.000, semente 42) — wrapper de ic_bootstrap_estratificado da origem."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado
from ensemble_llm.esquemas import ICBootstrap

from rag2.constantes import B_BOOTSTRAP, NIVEL_IC, SEMENTE


def ic_percentil(
    dados: pd.DataFrame,
    fn_estatistica: Callable[[pd.DataFrame], float],
    *,
    b: int = B_BOOTSTRAP,
    semente: int = SEMENTE,
    coluna_estrato: str | None = None,
    nivel: float = NIVEL_IC,
) -> ICBootstrap:
    """IC percentil por reamostragem (estratificada se `coluna_estrato`); b >= 100 exigido pela origem."""
    return ic_bootstrap_estratificado(
        dados,
        fn_estatistica,
        coluna_estratificacao=coluna_estrato,
        n_reamostras=b,
        nivel_confianca=nivel,
        metodo="percentile",
        seed=semente,
    )
