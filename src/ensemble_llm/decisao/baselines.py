"""Baselines B1, B2, B3.

Reusam por design toda a infraestrutura do pipeline principal:
- mesmas respostas dos agentes (`agent_responses/`),
- mesmos rótulos do juiz (`judge_labels/`),
- mesmas partições K-fold,
- mesmos pesos `w_i` e calibradores `f_i` por fold.

O que varia entre as versões:
- B1: `agent_chosen = argmax_i α_i` (frequentista bruta em D_fit); sempre o mesmo
  agente em todo D_test daquele fold; sem abstenção.
- B2: `s_i = w_i · c_i` (confiança bruta); seleção direta; sem abstenção.
- B3: `s_i = w_i · ĉ_i` (confiança calibrada); seleção direta; sem abstenção.
- Full (não está aqui): B3 + abstenção via `τ*`, implementada em `execucao.py`.

Seleção direta, não fusão: `z_system` é sempre derivado por lookup de `z_agent` do
vencedor, nunca por chamada adicional ao juiz.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ensemble_llm.calibracao.isotonica import aplicar_calibracao
from ensemble_llm.decisao.pontuacao import (
    calcular_eta,
    calcular_pontuacoes_agentes,
    selecionar_vencedor,
)
from ensemble_llm.decisao.risco_seletivo import encontrar_tau_estrela
from ensemble_llm.esquemas import PesosAgentes

logger = logging.getLogger(__name__)


def b1_agente_unico(
    *,
    pesos_art: PesosAgentes,
    by_agent: dict[str, pd.DataFrame],
    test_ids: list[str],
    z_lookup: dict[tuple[str, str], int],
    has_doc_ref_lookup: dict[str, bool],
    retrieval_recall_lookup: dict[str, float],
    fold_k: int,
) -> list[dict[str, Any]]:
    """B1: responde sempre com o agente de maior `α_i` em `D_fit`; sem abstenção."""
    del by_agent
    if not pesos_art.alphas_raw:
        raise ValueError("pesos_art.alphas_raw vazio — B1 não tem agente para escolher")
    agente_unico = max(pesos_art.alphas_raw, key=lambda a: pesos_art.alphas_raw[a])

    rows: list[dict[str, Any]] = []
    for iid in test_ids:
        rows.append(
            {
                "item_id": iid,
                "fold_k": fold_k,
                "has_doc_ref": has_doc_ref_lookup[iid],
                "retrieval_recall": retrieval_recall_lookup.get(iid, float("nan")),
                "agent_chosen": agente_unico,
                "eta": float("nan"),
                "abstained": False,
                "z_winner": z_lookup[(iid, agente_unico)],
                "z_system": z_lookup[(iid, agente_unico)],
                "baseline": "B1",
            }
        )
    return rows


def b2_ponderado_sem_calibracao(
    *,
    pesos_art: PesosAgentes,
    by_agent: dict[str, pd.DataFrame],
    test_ids: list[str],
    z_lookup: dict[tuple[str, str], int],
    has_doc_ref_lookup: dict[str, bool],
    retrieval_recall_lookup: dict[str, float],
    fold_k: int,
    tiebreak: str,
) -> list[dict[str, Any]]:
    """B2: `s_i = w_i · c_i` (confiança bruta); seleção direta; sem abstenção."""
    rows: list[dict[str, Any]] = []
    for iid in test_ids:
        confs: dict[str, float] = {}
        for aid, table in by_agent.items():
            raw = table.loc[iid, "confidence_raw"]
            confs[aid] = float(raw) if pd.notna(raw) else 0.0
        scores = calcular_pontuacoes_agentes(pesos_art.weights, confs)
        winner = selecionar_vencedor(scores, pesos_art.weights, tiebreak)
        rows.append(
            {
                "item_id": iid,
                "fold_k": fold_k,
                "has_doc_ref": has_doc_ref_lookup[iid],
                "retrieval_recall": retrieval_recall_lookup.get(iid, float("nan")),
                "agent_chosen": winner,
                "eta": calcular_eta(scores),
                "abstained": False,
                "z_winner": z_lookup[(iid, winner)],
                "z_system": z_lookup[(iid, winner)],
                "baseline": "B2",
            }
        )
    return rows


def b3_ponderado_calibrado(
    *,
    pesos_art: PesosAgentes,
    calibradores: dict[str, Any],
    by_agent: dict[str, pd.DataFrame],
    test_ids: list[str],
    z_lookup: dict[tuple[str, str], int],
    has_doc_ref_lookup: dict[str, bool],
    retrieval_recall_lookup: dict[str, float],
    fold_k: int,
    tiebreak: str,
) -> list[dict[str, Any]]:
    """B3: `s_i = w_i · ĉ_i` (confiança calibrada); seleção direta; sem abstenção."""
    rows: list[dict[str, Any]] = []
    for iid in test_ids:
        calibs: dict[str, float] = {}
        for aid, table in by_agent.items():
            raw = table.loc[iid, "confidence_raw"]
            c = float(raw) if pd.notna(raw) else 0.0
            calibs[aid] = aplicar_calibracao(calibradores[aid], c)
        scores = calcular_pontuacoes_agentes(pesos_art.weights, calibs)
        winner = selecionar_vencedor(scores, pesos_art.weights, tiebreak)
        rows.append(
            {
                "item_id": iid,
                "fold_k": fold_k,
                "has_doc_ref": has_doc_ref_lookup[iid],
                "retrieval_recall": retrieval_recall_lookup.get(iid, float("nan")),
                "agent_chosen": winner,
                "eta": calcular_eta(scores),
                "abstained": False,
                "z_winner": z_lookup[(iid, winner)],
                "z_system": z_lookup[(iid, winner)],
                "baseline": "B3",
            }
        )
    return rows


def bsolo_agente_calibrado_com_abstencao(
    *,
    agent_id: str,
    calibrador: Any,
    by_agent: dict[str, pd.DataFrame],
    val_ids: list[str],
    test_ids: list[str],
    z_lookup: dict[tuple[str, str], int],
    has_doc_ref_lookup: dict[str, bool],
    retrieval_recall_lookup: dict[str, float],
    fold_k: int,
    c_min_grid: list[float],
) -> dict[float, list[dict[str, Any]]]:
    """B_solo: agente único calibrado com abstenção seletiva por `C_min`.

    O limiar `tau*` é estimado exclusivamente em `D_val` usando o rótulo do
    próprio agente (`z_agent_i`), não o rótulo do vencedor do ensemble.
    """
    table = by_agent[agent_id]

    etas_val: list[float] = []
    z_val: list[int] = []
    for iid in val_ids:
        raw = table.loc[iid, "confidence_raw"]
        conf = float(raw) if pd.notna(raw) else 0.0
        conf_calib = aplicar_calibracao(calibrador, conf)
        etas_val.append(1.0 - conf_calib)
        z_val.append(z_lookup[(iid, agent_id)])

    test_base: list[dict[str, Any]] = []
    for iid in test_ids:
        raw = table.loc[iid, "confidence_raw"]
        conf = float(raw) if pd.notna(raw) else 0.0
        conf_calib = aplicar_calibracao(calibrador, conf)
        eta_solo = 1.0 - conf_calib
        z_agent = z_lookup[(iid, agent_id)]
        test_base.append(
            {
                "item_id": iid,
                "fold_k": fold_k,
                "c_min": None,
                "agent": agent_id,
                "agent_chosen": agent_id,
                "has_doc_ref": has_doc_ref_lookup[iid],
                "retrieval_recall": retrieval_recall_lookup.get(iid, float("nan")),
                "eta": eta_solo,
                "eta_solo": eta_solo,
                "z_agent": z_agent,
                "baseline": "Bsolo",
            }
        )

    out: dict[float, list[dict[str, Any]]] = {}
    val_df = pd.DataFrame({"eta": etas_val, "z_agent": z_val})
    for c_min in c_min_grid:
        threshold = encontrar_tau_estrela(
            etas=val_df["eta"].to_numpy(),
            z=val_df["z_agent"].to_numpy(),
            c_min=c_min,
        )
        rows_cmin: list[dict[str, Any]] = []
        for base in test_base:
            abstained = bool(base["eta_solo"] > threshold.tau_star)
            z_agent = int(base["z_agent"])
            rows_cmin.append(
                {
                    **base,
                    "c_min": c_min,
                    "tau_star_solo": threshold.tau_star,
                    "abstained": abstained,
                    "abstained_solo": abstained,
                    "z_system": None if abstained else z_agent,
                }
            )
        out[c_min] = rows_cmin
    return out
