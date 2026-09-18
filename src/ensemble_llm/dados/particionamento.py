"""Particionamento Stratified K-fold com lógica tri-partite."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ensemble_llm.esquemas import ParticaoTri


def _mapear_item_para_fold(
    item_ids: Sequence[str],
    rotulos_estratificacao: Sequence[int],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Atribui cada item_id a um fold via StratifiedKFold, preservando proporção de estratos."""
    if len(item_ids) != len(rotulos_estratificacao):
        raise ValueError("item_ids e rotulos_estratificacao devem ter o mesmo comprimento")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros((len(item_ids), 1))
    y = np.asarray(rotulos_estratificacao)

    item_para_fold: dict[str, int] = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        for i in test_idx:
            item_para_fold[item_ids[i]] = fold_idx
    return item_para_fold


def _construir_permutacao_folds(n_splits: int, seed: int) -> list[int]:
    """Permutação determinística dos índices de fold para rotação val/fit entre execuções."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_splits).tolist()


def criar_particoes(
    item_ids: Sequence[str],
    rotulos_estratificacao: Sequence[int],
    n_splits: int = 10,
    n_folds_val: int = 2,
    seed: int = 42,
) -> tuple[list[ParticaoTri], list[int]]:
    """Gera as K partições fit/val/test garantindo cobertura completa e sem vazamento."""
    if n_folds_val >= n_splits:
        raise ValueError(f"n_folds_val ({n_folds_val}) deve ser < n_splits ({n_splits})")

    item_para_fold = _mapear_item_para_fold(item_ids, rotulos_estratificacao, n_splits, seed)
    permutacao_folds = _construir_permutacao_folds(n_splits, seed)

    fold_para_itens: dict[int, list[str]] = {f: [] for f in range(n_splits)}
    for item_id, f in item_para_fold.items():
        fold_para_itens[f].append(item_id)

    particoes: list[ParticaoTri] = []
    for k in range(n_splits):
        test_ids = tuple(fold_para_itens[k])
        folds_restantes = [f for f in permutacao_folds if f != k]
        folds_val = folds_restantes[:n_folds_val]
        folds_fit = folds_restantes[n_folds_val:]

        val_ids = tuple(item_id for f in folds_val for item_id in fold_para_itens[f])
        fit_ids = tuple(item_id for f in folds_fit for item_id in fold_para_itens[f])

        particoes.append(ParticaoTri(k=k, fit_ids=fit_ids, val_ids=val_ids, test_ids=test_ids))

    return particoes, permutacao_folds


def iterar_particoes(
    item_ids: Sequence[str],
    rotulos_estratificacao: Sequence[int],
    n_splits: int = 10,
    n_folds_val: int = 2,
    seed: int = 42,
) -> Iterator[ParticaoTri]:
    """Iterador lazy sobre as K partições fit/val/test."""
    particoes, _ = criar_particoes(item_ids, rotulos_estratificacao, n_splits, n_folds_val, seed)
    yield from particoes


def validar_cobertura_completa(particoes: Sequence[ParticaoTri]) -> None:
    """Garante que cada item aparece no conjunto de test exatamente uma vez nas K partições."""
    contagens_teste: dict[str, int] = {}
    for p in particoes:
        for item_id in p.test_ids:
            contagens_teste[item_id] = contagens_teste.get(item_id, 0) + 1

    violacoes = {iid: c for iid, c in contagens_teste.items() if c != 1}
    if violacoes:
        raise AssertionError(
            f"Violação de cobertura completa: {len(violacoes)} itens aparecem em test "
            f"um número de vezes != 1. Exemplos: {list(violacoes.items())[:5]}"
        )
