"""Vetorização para retrieval denso."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Vetorizador(Protocol):
    """Protocolo para vetorizadores de texto usados no índice denso."""

    dim: int
    nome_modelo: str

    def codificar(self, textos: list[str]) -> np.ndarray: ...


class VetorizadorHash:
    """Vetorizador determinístico para testes."""

    def __init__(self, dim: int = 64, semente: int = 42) -> None:
        self.dim = dim
        self.nome_modelo = f"vetorizador-hash-d{dim}-seed{semente}"
        self.semente = semente

    def codificar(self, textos: list[str]) -> np.ndarray:
        vetores = np.zeros((len(textos), self.dim), dtype=np.float32)
        for i, texto in enumerate(textos):
            h = hashlib.sha256(texto.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big") ^ self.semente)
            v = rng.standard_normal(self.dim).astype(np.float32)
            norma = float(np.linalg.norm(v))
            if norma > 0:
                v /= norma
            vetores[i] = v
        return vetores


class VetorizadorSentenceTransformer:
    """Wrapper lazy sobre sentence-transformers."""

    def __init__(
        self,
        nome_modelo: str = "BAAI/bge-m3",
        tamanho_lote: int = 32,
        dispositivo: str | None = None,
    ) -> None:
        self.nome_modelo = nome_modelo
        self.tamanho_lote = tamanho_lote
        self.dispositivo = dispositivo
        self._modelo = None
        self.dim: int = -1

    def _garantir_carregado(self) -> None:
        """Carrega o modelo SentenceTransformer sob demanda (lazy loading)."""
        if self._modelo is not None:
            return
        self._modelo = self._carregar_modelo(self.dispositivo)
        amostra = self._modelo.encode(["x"], normalize_embeddings=True)
        self.dim = int(amostra.shape[1])

    def _carregar_modelo(self, dispositivo: str | None) -> Any:
        """Cria a instância SentenceTransformer no dispositivo solicitado."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers não instalado. Use `uv add sentence-transformers`."
            ) from e
        logger.info(
            "Carregando modelo de embedding: %s (device=%s, batch_size=%d)",
            self.nome_modelo,
            dispositivo or "auto",
            self.tamanho_lote,
        )
        return SentenceTransformer(self.nome_modelo, device=dispositivo)

    @staticmethod
    def _eh_oom_cuda(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "outofmemoryerror" in msg

    @staticmethod
    def _limpar_cache_cuda() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def codificar(self, textos: list[str]) -> np.ndarray:
        """Codifica lista de textos em embeddings float32 normalizados via bge-m3."""
        self._garantir_carregado()
        assert self._modelo is not None
        tamanhos_lote: list[int] = []
        lote = self.tamanho_lote
        while lote >= 1:
            if lote not in tamanhos_lote:
                tamanhos_lote.append(lote)
            lote //= 2

        ultimo_erro: BaseException | None = None
        for tamanho_lote in tamanhos_lote:
            try:
                logger.info(
                    "Vetorizando %d texto(s) com %s (device=%s, batch_size=%d)",
                    len(textos),
                    self.nome_modelo,
                    self.dispositivo or "auto",
                    tamanho_lote,
                )
                saida = self._modelo.encode(
                    textos,
                    batch_size=tamanho_lote,
                    normalize_embeddings=True,
                    show_progress_bar=len(textos) >= 1000,
                    convert_to_numpy=True,
                )
                return saida.astype(np.float32)
            except Exception as e:
                if not self._eh_oom_cuda(e):
                    raise
                ultimo_erro = e
                self._limpar_cache_cuda()
                logger.warning(
                    "OOM no embedding com device=%s e batch_size=%d; tentando batch menor.",
                    self.dispositivo or "auto",
                    tamanho_lote,
                )

        if self.dispositivo != "cpu":
            logger.warning(
                "OOM persistente em GPU para %s; recarregando modelo em CPU e repetindo vetorização.",
                self.nome_modelo,
            )
            self.dispositivo = "cpu"
            self._modelo = self._carregar_modelo(self.dispositivo)
            saida = self._modelo.encode(
                textos,
                batch_size=min(self.tamanho_lote, 8),
                normalize_embeddings=True,
                show_progress_bar=len(textos) >= 1000,
                convert_to_numpy=True,
            )
            return saida.astype(np.float32)

        assert ultimo_erro is not None
        raise RuntimeError(
            f"Falha ao vetorizar textos com {self.nome_modelo} mesmo após reduzir batch_size."
        ) from ultimo_erro
