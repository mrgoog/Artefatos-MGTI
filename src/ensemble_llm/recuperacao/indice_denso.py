"""Índice denso em memória com busca exata por similaridade cosseno."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ensemble_llm.recuperacao.vetorizacao import Vetorizador
from ensemble_llm.esquemas import BlocoDocumento

logger = logging.getLogger(__name__)


class IndiceDenso:
    """Índice de busca semântica exata em memória via produto interno (similaridade cosseno)."""

    def __init__(self, vetorizador: Vetorizador) -> None:
        self.vetorizador = vetorizador
        self._chunks: list[BlocoDocumento] = []
        self._vetores: np.ndarray | None = None

    def construir(self, chunks: list[BlocoDocumento]) -> None:
        """Codifica os chunks e armazena a matriz de embeddings em memória."""
        if not chunks:
            raise ValueError("chunks não pode ser vazio")
        textos = [c.text for c in chunks]
        logger.info(
            "IndiceDenso: codificando %d chunks com %s",
            len(textos),
            self.vetorizador.nome_modelo,
        )
        self._vetores = self.vetorizador.codificar(textos)
        self._chunks = list(chunks)

    def buscar(self, consulta: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retorna os top-k chunk_ids com maior similaridade cosseno à consulta."""
        if self._vetores is None:
            raise RuntimeError("Índice não construído. Chame .construir() primeiro.")
        if top_k < 1:
            raise ValueError(f"top_k deve ser >= 1, recebeu {top_k}")

        vetor_consulta = self.vetorizador.codificar([consulta])
        escores = (self._vetores @ vetor_consulta[0]).astype(np.float32)

        k = min(top_k, len(self._chunks))
        idx_parcial = np.argpartition(-escores, k - 1)[:k]
        idx_ordenado = idx_parcial[np.argsort(-escores[idx_parcial])]
        return [(self._chunks[i].chunk_id, float(escores[i])) for i in idx_ordenado]

    def salvar(self, caminho: Path) -> None:
        """Persiste matriz de embeddings (npz) e metadados dos chunks (parquet) em disco."""
        if self._vetores is None:
            raise RuntimeError("Índice não construído")
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(caminho / "vetores.npz", vetores=self._vetores)

        import pandas as pd

        df_chunks = pd.DataFrame([c.model_dump() for c in self._chunks])
        df_chunks.to_parquet(caminho / "chunks.parquet", index=False)

    def carregar(self, caminho: Path) -> None:
        """Restaura embeddings e metadados dos chunks a partir do diretório de persistência."""
        caminho = Path(caminho)
        npz = np.load(caminho / "vetores.npz")
        self._vetores = npz["vetores"]

        import pandas as pd

        df = pd.read_parquet(caminho / "chunks.parquet")
        self._chunks = [BlocoDocumento.model_validate(linha) for linha in df.to_dict(orient="records")]

    def __len__(self) -> int:
        return len(self._chunks)
