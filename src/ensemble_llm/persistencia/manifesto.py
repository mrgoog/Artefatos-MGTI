"""Manifesto de execução para reprodutibilidade."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ManifestoExecucao(BaseModel):
    """Snapshot de reprodutibilidade do experimento (seed, permutação de folds, modelos usados)."""

    experiment_id: str
    created_at: str
    git_sha: str
    root_seed: int
    fold_permutation: list[int]
    agents: dict[str, str] = Field(default_factory=dict)
    judge_model: str


def agora_iso() -> str:
    """Retorna timestamp UTC atual em formato ISO 8601."""
    return datetime.now(UTC).isoformat()


def escrever_manifesto(path: Path, manifesto: ManifestoExecucao) -> None:
    """Persiste manifesto em YAML, criando diretórios pai se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifesto.model_dump(), f, sort_keys=False, allow_unicode=True)


def ler_manifesto(path: Path) -> ManifestoExecucao:
    """Carrega e valida ManifestoExecucao a partir de um YAML."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ManifestoExecucao.model_validate(data)
