from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mensuracoes_decisao_artigo2",
    RAIZ / "scripts" / "mensuracoes_decisao_artigo2.py",
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_pesos_resposta_fracionam_apenas_empate_de_corte() -> None:
    score = np.array([0.9, 0.8, 0.8, 0.1])
    pesos = mod.pesos_resposta_para_cobertura(score, 0.5)
    np.testing.assert_allclose(pesos, [1.0, 0.5, 0.5, 0.0])
    assert pesos.mean() == 0.5


def test_metricas_separam_abstencao_sistemica_e_recusa_textual() -> None:
    dados = pd.DataFrame(
        {
            "z_winner": [1, 0, 0, 1],
            "textual_refusal": [False, False, True, False],
            "refusal_verdict": [np.nan, np.nan, 2.0, np.nan],
            "context_sufficiency": [2, 0, 0, 2],
        }
    )
    # O sistema responde aos itens 0, 2 e 3; o item 2 é recusa textual justificada.
    metricas = mod.calcular_metricas(dados, np.array([1.0, 0.0, 1.0, 1.0]))

    assert metricas["cobertura_nominal"] == 0.75
    assert metricas["cobertura_efetiva"] == 0.5
    assert metricas["utilidade_convencional"] == 0.5
    assert metricas["utilidade_contexto_estrita"] == 1.0
    assert metricas["penalizacao_contextual_estrita"] == 0.5
