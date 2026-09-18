"""Esquemas deste repositório: extensões de BlocoRecuperado com os escores dos degraus (viram colunas extras em servidos.parquet)."""

from __future__ import annotations

from ensemble_llm.esquemas import BlocoRecuperado


class BlocoFundido(BlocoRecuperado):
    """Chunk servido por R1+: dense_score/bm25_score brutos dos ramos (None se o chunk não veio daquele ramo) + score RRF."""

    rrf_score: float


class BlocoReranqueado(BlocoFundido):
    """Chunk servido por R2+: escores brutos + RRF + score sigmoide do cross-encoder em [0, 1] (a escala em que R4 soma β e γ)."""

    rerank_score: float
