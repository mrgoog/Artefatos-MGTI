"""Contrato dos recuperadores da escada: recuperar(consulta) -> list[BlocoRecuperado] com validação de saída."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ensemble_llm.esquemas import BlocoRecuperado

from rag2.constantes import N_FINAL


@runtime_checkable
class Recuperador(Protocol):
    """Um degrau da escada: nome e função de recuperação por consulta."""

    degrau: str

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]: ...


def validar_saida(blocos: list[BlocoRecuperado], n_max: int = N_FINAL) -> list[BlocoRecuperado]:
    """Exige ≤ n_max blocos, chunk_id únicos e combined_rank = 1..n consecutivo; ValueError caso contrário."""
    if len(blocos) > n_max:
        raise ValueError(f"{len(blocos)} blocos > n_max={n_max}")
    ids = [b.chunk_id for b in blocos]
    if len(set(ids)) != len(ids):
        raise ValueError(f"chunk_id repetido na saída: {ids}")
    ranks = [b.combined_rank for b in blocos]
    if ranks != list(range(1, len(blocos) + 1)):
        raise ValueError(f"combined_rank não é 1..n: {ranks}")
    return blocos
