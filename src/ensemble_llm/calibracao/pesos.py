"""Pesos por adequação validada."""

from __future__ import annotations

import warnings

import numpy as np

from ensemble_llm.esquemas import PesosAgentes


def calcular_pesos_agentes(
    z_por_agente: dict[str, list[int]],
) -> PesosAgentes:
    """Calcula pesos por agente a partir dos rótulos em D_fit."""
    if not z_por_agente:
        raise ValueError("z_por_agente não pode ser vazio")

    tamanhos = {len(zs) for zs in z_por_agente.values()}
    if len(tamanhos) != 1:
        raise ValueError(f"Inconsistência de tamanhos entre agentes: {tamanhos}")

    tamanho_fit = tamanhos.pop()
    if tamanho_fit == 0:
        raise ValueError("D_fit vazio")

    alfas: dict[str, float] = {}
    for agente_id, zs in z_por_agente.items():
        arr = np.asarray(zs, dtype=float)
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"Agente {agente_id}: z deve ser binário em {{0,1}}")
        alfas[agente_id] = float(arr.mean())

    total = sum(alfas.values())
    if total == 0.0:
        warnings.warn(
            "Todos os agentes têm α=0 em D_fit (sempre erram). "
            "Atribuindo pesos uniformes como fallback.",
            UserWarning,
            stacklevel=2,
        )
        n = len(alfas)
        pesos = dict.fromkeys(alfas, 1.0 / n)
    else:
        pesos = {a: alfa / total for a, alfa in alfas.items()}

    return PesosAgentes(weights=pesos, alphas_raw=alfas, fit_size=tamanho_fit)
