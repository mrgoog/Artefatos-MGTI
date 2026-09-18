"""Leitura somente-leitura de um índice híbrido persistido (denso + BM25) em arrays alinhados."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag2.dados import tipo_fonte


@dataclass
class IndiceHibrido:
    """Índice carregado em memória: chunks, vetores densos e BM25 com seus ids."""

    diretorio: Path
    chunks: pd.DataFrame            # chunk_id, text, source_doc, section_path, n_tokens — ordem do denso
    vetores: np.ndarray             # (n_chunks, dim) float32; linha i ↔ chunks.iloc[i]
    bm25: BM25Okapi
    ids_bm25: np.ndarray            # chunk_id na ordem interna do BM25 (pode diferir da do denso)
    remover_stopwords: bool
    texto: dict[str, str] = field(init=False)
    doc: dict[str, str] = field(init=False)
    ids_denso: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Constrói os mapas chunk_id → texto / source_doc e o vetor de ids na ordem densa."""
        self.texto = dict(zip(self.chunks["chunk_id"], self.chunks["text"], strict=True))
        self.doc = dict(zip(self.chunks["chunk_id"], self.chunks["source_doc"], strict=True))
        self.ids_denso = self.chunks["chunk_id"].to_numpy()

    def __len__(self) -> int:
        return len(self.chunks)

    def composicao(self) -> dict[str, dict[str, int]]:
        """Contagem de documentos-fonte e de chunks por tipo_fonte."""
        por_chunk = self.chunks["source_doc"].map(tipo_fonte)
        por_doc = self.chunks.drop_duplicates("source_doc")["source_doc"].map(tipo_fonte)
        return {
            "documentos": {k: int(v) for k, v in por_doc.value_counts().items()},
            "chunks": {k: int(v) for k, v in por_chunk.value_counts().items()},
        }


def carregar_indice(diretorio: Path) -> IndiceHibrido:
    """Lê denso/chunks.parquet, denso/vetores.npz e bm25/bm25.pkl e valida o alinhamento."""
    diretorio = Path(diretorio)
    chunks = pd.read_parquet(diretorio / "denso" / "chunks.parquet")
    vetores = np.load(diretorio / "denso" / "vetores.npz")["vetores"].astype(np.float32)
    with (diretorio / "bm25" / "bm25.pkl").open("rb") as f:
        dados = pickle.load(f)
    ids_bm25 = np.array([c["chunk_id"] for c in dados["chunks"]])
    if not (len(chunks) == len(vetores) == len(ids_bm25)):
        raise ValueError(
            f"índice desalinhado em {diretorio}: chunks={len(chunks)} vetores={len(vetores)} bm25={len(ids_bm25)}"
        )
    return IndiceHibrido(
        diretorio=diretorio,
        chunks=chunks,
        vetores=vetores,
        bm25=dados["bm25"],
        ids_bm25=ids_bm25,
        remover_stopwords=bool(dados["remover_stopwords"]),
    )
