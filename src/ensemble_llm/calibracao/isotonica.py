"""Calibração isotônica da confiança autodeclarada por agente."""

from __future__ import annotations

from sklearn.isotonic import IsotonicRegression


def ajustar_calibrador_isotonico(
    confiancas: list[float],
    rotulos: list[int],
) -> IsotonicRegression:
    """Ajusta f_i sobre pares (c_i, z_i) em D_fit."""
    if len(confiancas) != len(rotulos):
        raise ValueError("confiancas e rotulos devem ter o mesmo comprimento")
    if len(confiancas) == 0:
        raise ValueError("D_fit vazio")

    iso = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
        increasing=True,
    )
    iso.fit(confiancas, rotulos)
    return iso


def aplicar_calibracao(
    calibrador: IsotonicRegression,
    confianca: float,
) -> float:
    """Aplica f_i a uma confiança individual; retorna ĉ_i ∈ [0,1]."""
    return float(calibrador.predict([confianca])[0])
