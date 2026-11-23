"""Leitura/escrita das tabelas parquet principais."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def escrever_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persiste DataFrame em parquet, criando diretórios pai se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def ler_parquet(path: Path) -> pd.DataFrame:
    """Lê parquet do disco e retorna DataFrame pandas."""
    return pd.read_parquet(path)
