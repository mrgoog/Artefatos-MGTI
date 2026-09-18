"""Cross-encoder de reordenação (R2+): Pontuador lazy sobre CrossEncoder fp16 e cache parquet dos scores, por consulta."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from rag2.config import RAIZ
from rag2.constantes import MODELO_RERANKER, RERANK_BATCH
from rag2.pool import hash_consulta, indice_id

DIR_RERANKER = RAIZ / "runs" / "_shared" / "reranker"
"""Caches de scores do reranker (gitignored); um arquivo por (índice, k_pool, pool_rerank, modelo)."""
COLUNAS = ("consulta_hash", "rank", "chunk_id", "score")
"""Esquema do cache: rank = posição 1-based na lista RRF (a ordem em que os pares foram pontuados); score = sigmoide em [0, 1]."""
DTYPES = {"rank": "int32", "score": "float32"}
MAX_LENGTH = 8192
"""Tokens por par (pergunta + chunk): o máximo do tokenizador do bge-reranker-v2-m3; nenhum dos 48.458 pares o excede (máx. 6.630)."""


@runtime_checkable
class Pontuador(Protocol):
    """Quem pontua pares (consulta, texto): o Reranker real ou um falso determinístico nos testes."""

    nome: str

    def pontuar(self, consulta: str, textos: Sequence[str]) -> np.ndarray: ...


def caminho_cache_reranker(dir_indice: Path, k_pool: int, pool_rerank: int, modelo: str = MODELO_RERANKER) -> Path:
    """runs/_shared/reranker/<indice_id>_k<k_pool>_r<pool_rerank>_<modelo sem organização>.parquet — o nome carrega tudo de que o score depende."""
    return DIR_RERANKER / f"{indice_id(dir_indice)}_k{k_pool}_r{pool_rerank}_{modelo.split('/')[-1]}.parquet"


class Reranker:
    """CrossEncoder(MODELO_RERANKER) em fp16, carregado na primeira chamada; sigmoide padrão (num_labels=1) → score em [0, 1]."""

    def __init__(
        self, dispositivo: str, *, modelo: str = MODELO_RERANKER, tamanho_lote: int = RERANK_BATCH, max_length: int = MAX_LENGTH
    ) -> None:
        self.nome = modelo
        self.dispositivo = dispositivo
        self.tamanho_lote = tamanho_lote
        self.max_length = max_length
        self._modelo = None

    def _garantir_carregado(self):
        """Carrega o CrossEncoder sob demanda: com todas as consultas em cache, o modelo nunca entra na GPU."""
        if self._modelo is None:
            from sentence_transformers import CrossEncoder

            self._modelo = CrossEncoder(
                self.nome, device=self.dispositivo, max_length=self.max_length, model_kwargs={"dtype": "float16"}
            )
        return self._modelo

    @property
    def max_seq_length(self) -> int:
        """Comprimento máximo efetivo do tokenizador (carrega o modelo)."""
        return int(self._garantir_carregado().max_seq_length)

    def pontuar(self, consulta: str, textos: Sequence[str]) -> np.ndarray:
        """Scores float32 dos pares (consulta, texto) na ordem dada, em uma única chamada de predict (o lote é parte do resultado)."""
        if not textos:
            return np.zeros(0, dtype=np.float32)
        saida = self._garantir_carregado().predict(
            [(consulta, t) for t in textos], batch_size=self.tamanho_lote, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(saida, dtype=np.float32).reshape(-1)


class CacheReranker:
    """Scores persistidos em um parquet (COLUNAS), carregado inteiro; a unidade é a consulta: todos os pares da sua lista RRF de uma vez."""

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self._scores: dict[str, list[tuple[str, float]]] = {}
        if self.caminho.exists():
            self._indexar(pd.read_parquet(self.caminho))

    def _indexar(self, df: pd.DataFrame) -> None:
        cols = ["consulta_hash", "chunk_id", "score"]
        for h, cid, score in df.sort_values(["consulta_hash", "rank"])[cols].itertuples(index=False):
            self._scores.setdefault(h, []).append((cid, float(score)))

    def __len__(self) -> int:
        return len(self._scores)

    def __contains__(self, consulta: str) -> bool:
        return hash_consulta(consulta) in self._scores

    def faltantes(self, consultas: Sequence[str]) -> list[str]:
        """Consultas (únicas por hash, na ordem dada) ainda não pontuadas."""
        vistos: set[str] = set()
        saida: list[str] = []
        for c in consultas:
            h = hash_consulta(c)
            if h not in self._scores and h not in vistos:
                vistos.add(h)
                saida.append(c)
        return saida

    def scores(self, consulta: str) -> list[tuple[str, float]]:
        """[(chunk_id, score), …] na ordem RRF em que foram pontuados; KeyError se a consulta não foi pontuada."""
        return self._scores[hash_consulta(consulta)]

    def adicionar(self, df: pd.DataFrame) -> int:
        """Acrescenta as consultas de `df` ainda ausentes e regrava o parquet (tmp + os.replace); devolve quantas entraram."""
        novos = df[~df["consulta_hash"].isin(list(self._scores))]
        n = novos["consulta_hash"].nunique()
        if n == 0:
            return 0
        self._indexar(novos)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        atual = pd.read_parquet(self.caminho) if self.caminho.exists() else None
        tudo = novos if atual is None else pd.concat([atual, novos], ignore_index=True)
        tmp = self.caminho.with_name(self.caminho.name + ".tmp")
        tudo.to_parquet(tmp, index=False)
        os.replace(tmp, self.caminho)
        return int(n)
