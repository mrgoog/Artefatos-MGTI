"""Política seletiva calibrada por risco."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ensemble_llm.esquemas import LimiarSeletivo


@dataclass(frozen=True)
class PontoRiscoCobertura:
    """Ponto (τ, cobertura, risco_seletivo) na curva risco × cobertura."""

    tau: float
    cobertura: float
    risco_seletivo: float
    n_respondidos: int


def cobertura_em(etas: np.ndarray, tau: float) -> float:
    """Fração de itens com η ≤ τ (cobertura empírica para um dado limiar)."""
    return float((etas <= tau).mean())


def risco_seletivo_em(
    etas: np.ndarray,
    z: np.ndarray,
    tau: float,
) -> tuple[float, int]:
    """Taxa de erro e contagem de respondidos para itens com η ≤ τ."""
    respondidos = etas <= tau
    n_respondidos = int(respondidos.sum())
    if n_respondidos == 0:
        return float("nan"), 0
    risco = float((1 - z[respondidos]).mean())
    return risco, n_respondidos


def encontrar_tau_estrela(
    etas: np.ndarray,
    z: np.ndarray,
    c_min: float,
) -> LimiarSeletivo:
    """Minimiza risco seletivo sujeito a cobertura ≥ C_min sobre D_val."""
    if etas.shape != z.shape:
        raise ValueError(f"Shape mismatch: etas {etas.shape} vs z {z.shape}")
    if not np.all((z == 0) | (z == 1)):
        raise ValueError("z deve ser binário em {0,1}")
    if not (0.0 < c_min <= 1.0):
        raise ValueError(f"c_min deve estar em (0,1], recebeu {c_min}")

    candidatos = np.unique(np.concatenate([etas, np.array([1.0])]))

    melhor_tau: float | None = None
    melhor_risco: float = float("inf")
    melhor_cobertura: float = 0.0

    for tau in candidatos:
        cob = cobertura_em(etas, float(tau))
        if cob < c_min:
            continue
        risco, n_resp = risco_seletivo_em(etas, z, float(tau))
        if n_resp == 0:
            continue
        if (risco < melhor_risco) or (risco == melhor_risco and cob > melhor_cobertura):
            melhor_tau = float(tau)
            melhor_risco = risco
            melhor_cobertura = cob

    if melhor_tau is None:
        warnings.warn(
            f"Nenhum τ candidato satisfez C_min={c_min}; "
            "usando τ=1.0 (cobertura plena) como fallback.",
            UserWarning,
            stacklevel=2,
        )
        melhor_tau = 1.0
        melhor_cobertura = cobertura_em(etas, 1.0)
        melhor_risco, _ = risco_seletivo_em(etas, z, 1.0)
        restricao_satisfeita = False
    else:
        restricao_satisfeita = True

    return LimiarSeletivo(
        tau_star=melhor_tau,
        c_min=c_min,
        coverage_emp_val=melhor_cobertura,
        risk_sel_emp_val=melhor_risco,
        val_size=len(etas),
        constraint_satisfied=restricao_satisfeita,
    )


def curva_risco_cobertura(
    etas: np.ndarray,
    z: np.ndarray,
    n_pontos: int = 200,
) -> list[PontoRiscoCobertura]:
    """Gera a curva risco × cobertura varrendo τ em grade uniforme + valores únicos de η."""
    grade = np.linspace(0.0, 1.0, n_pontos)
    etas_unicos = np.unique(etas)
    taus = np.unique(np.concatenate([grade, etas_unicos, [1.0]]))

    pontos: list[PontoRiscoCobertura] = []
    for tau in taus:
        cob = cobertura_em(etas, float(tau))
        risco, n_resp = risco_seletivo_em(etas, z, float(tau))
        pontos.append(
            PontoRiscoCobertura(
                tau=float(tau),
                cobertura=cob,
                risco_seletivo=risco,
                n_respondidos=n_resp,
            )
        )
    return pontos
