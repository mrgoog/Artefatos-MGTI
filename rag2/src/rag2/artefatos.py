"""Gravação padronizada dos artefatos de um degrau/variante em runs/… (por_item.parquet, resumo.json, resumo.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rag2.metricas import CABECALHO_MARKDOWN, linha_markdown


def gravar_artefatos(diretorio: Path, por_item: pd.DataFrame, resumo: dict, nome: str) -> Path:
    """Cria o diretório e grava por_item.parquet, resumo.json (indentado, UTF-8) e resumo.md; devolve o diretório."""
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    por_item.to_parquet(diretorio / "por_item.parquet", index=False)
    (diretorio / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (diretorio / "resumo.md").write_text(
        f"# {nome}\n\n{CABECALHO_MARKDOWN}\n{linha_markdown(nome, resumo)}\n", encoding="utf-8"
    )
    return diretorio


def ler_resumo(diretorio: Path) -> dict | None:
    """resumo.json do diretório, ou None se ainda não foi registrado."""
    caminho = Path(diretorio) / "resumo.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))
