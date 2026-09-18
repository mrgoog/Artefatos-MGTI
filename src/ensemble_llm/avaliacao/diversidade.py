"""Matriz de concordância de erros entre pares de agentes."""

from __future__ import annotations

import pandas as pd


def matriz_concordancia_erros(
    z_por_agente: dict[str, list[int]],
) -> pd.DataFrame:
    """Matriz P(agente j erra | agente i errou) para análise de diversidade do ensemble."""
    if not z_por_agente:
        raise ValueError("z_por_agente não pode ser vazio")
    tamanhos = {len(v) for v in z_por_agente.values()}
    if len(tamanhos) != 1:
        raise ValueError(f"Inconsistência de tamanhos: {tamanhos}")

    agentes = list(z_por_agente.keys())
    df = pd.DataFrame(z_por_agente)
    matriz = pd.DataFrame(index=agentes, columns=agentes, dtype=float)

    for i in agentes:
        erros_i = df[i] == 0
        n_erros_i = int(erros_i.sum())
        for j in agentes:
            if n_erros_i == 0:
                matriz.loc[i, j] = float("nan")
                continue
            erros_j_dado_i = (df[j] == 0) & erros_i
            matriz.loc[i, j] = float(erros_j_dado_i.sum() / n_erros_i)

    return matriz


def fracao_erro_todos_agentes(
    z_por_agente: dict[str, list[int]],
) -> dict[str, float]:
    """Fração de itens onde todos os agentes erram vs. esperado sob independência."""
    df = pd.DataFrame(z_por_agente)
    taxas_erro = (df == 0).mean()
    fracao_todos_erram = float((df == 0).all(axis=1).mean())
    esperado_independencia = float(taxas_erro.prod())
    return {
        "fraction_all_wrong": fracao_todos_erram,
        "expected_under_independence": esperado_independencia,
        "ratio_observed_to_expected": (
            fracao_todos_erram / esperado_independencia
            if esperado_independencia > 0
            else float("nan")
        ),
    }
