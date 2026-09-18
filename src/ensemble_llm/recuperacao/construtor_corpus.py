"""Construtor do corpus RAG."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ensemble_llm.esquemas import BlocoDocumento
from ensemble_llm.recuperacao.hibrido import RecuperadorHibrido
from ensemble_llm.recuperacao.indice_bm25 import IndiceBM25
from ensemble_llm.recuperacao.indice_denso import IndiceDenso
from ensemble_llm.recuperacao.segmentacao import (
    ConfigSegmentacao,
    Tokenizador,
    segmentar_documento,
    tokenizador_espacos,
)
from ensemble_llm.recuperacao.vetorizacao import Vetorizador

logger = logging.getLogger(__name__)
NOME_ARQUIVO_METADADOS_RAG = "index_metadata.json"


def construir_chunks_corpus(
    documentos: dict[str, str],
    *,
    tokenizador: Tokenizador = tokenizador_espacos,
    config_segmentacao: ConfigSegmentacao | None = None,
) -> list[BlocoDocumento]:
    """Segmenta todos os documentos do corpus em BlocoDocumento prontos para indexação."""
    todos_chunks: list[BlocoDocumento] = []
    for doc_id, texto in documentos.items():
        if not texto or not texto.strip():
            logger.debug("Documento %s vazio — pulando", doc_id)
            continue
        chunks = segmentar_documento(
            texto,
            source_doc=doc_id,
            tokenizador=tokenizador,
            config=config_segmentacao,
        )
        todos_chunks.extend(chunks)
    return todos_chunks


def construir_recuperador_hibrido(
    documentos: dict[str, str],
    *,
    vetorizador: Vetorizador,
    tokenizador: Tokenizador = tokenizador_espacos,
    config_segmentacao: ConfigSegmentacao | None = None,
    top_k_denso: int = 5,
    top_k_bm25: int = 5,
    diretorio_persistencia: Path | None = None,
) -> RecuperadorHibrido:
    """Constrói índices denso e BM25, e opcionalmente persiste ambos em disco."""
    chunks = construir_chunks_corpus(
        documentos,
        tokenizador=tokenizador,
        config_segmentacao=config_segmentacao,
    )
    if not chunks:
        raise ValueError("Nenhum chunk gerado a partir dos documentos")

    indice_denso = IndiceDenso(vetorizador=vetorizador)
    indice_denso.construir(chunks)

    indice_bm25 = IndiceBM25()
    indice_bm25.construir(chunks)

    if diretorio_persistencia is not None:
        diretorio_persistencia = Path(diretorio_persistencia)
        indice_denso.salvar(diretorio_persistencia / "denso")
        indice_bm25.salvar(diretorio_persistencia / "bm25")

    return RecuperadorHibrido(
        indice_denso=indice_denso,
        indice_bm25=indice_bm25,
        top_k_denso=top_k_denso,
        top_k_bm25=top_k_bm25,
    )


def carregar_recuperador_hibrido(
    diretorio_persistencia: Path,
    *,
    vetorizador: Vetorizador,
    top_k_denso: int = 5,
    top_k_bm25: int = 5,
) -> RecuperadorHibrido:
    """Restaura RecuperadorHibrido a partir de índices persistidos em disco."""
    diretorio_persistencia = Path(diretorio_persistencia)
    indice_denso = IndiceDenso(vetorizador=vetorizador)
    indice_denso.carregar(diretorio_persistencia / "denso")

    indice_bm25 = IndiceBM25()
    indice_bm25.carregar(diretorio_persistencia / "bm25")

    return RecuperadorHibrido(
        indice_denso=indice_denso,
        indice_bm25=indice_bm25,
        top_k_denso=top_k_denso,
        top_k_bm25=top_k_bm25,
    )


def salvar_metadados_indice_rag(
    diretorio_persistencia: Path,
    metadados: dict[str, Any],
) -> Path:
    """Persiste metadados descritivos do índice RAG no diretório raiz do índice."""
    diretorio_persistencia = Path(diretorio_persistencia)
    diretorio_persistencia.mkdir(parents=True, exist_ok=True)
    caminho = diretorio_persistencia / NOME_ARQUIVO_METADADOS_RAG
    caminho.write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return caminho


def carregar_metadados_indice_rag(
    diretorio_persistencia: Path,
) -> dict[str, Any] | None:
    """Carrega metadados do índice RAG quando o arquivo existe."""
    caminho = Path(diretorio_persistencia) / NOME_ARQUIVO_METADADOS_RAG
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))
