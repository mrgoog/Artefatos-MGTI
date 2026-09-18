"""Pools por ramo (denso e BM25) para consultas em lote, com cache parquet chaveado por hash do texto da consulta."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_llm.recuperacao.construtor_corpus import carregar_metadados_indice_rag
from ensemble_llm.recuperacao.indice_bm25 import tokenizar_pt
from ensemble_llm.recuperacao.vetorizacao import Vetorizador

from rag2.config import RAIZ
from rag2.indice import IndiceHibrido

DIR_POOLS = RAIZ / "runs" / "_shared" / "pools"
"""Caches de pools (gitignored); um arquivo por (índice, k)."""
COLUNAS = ("consulta_hash", "ramo", "rank", "chunk_id", "score")
"""Esquema do cache: ramo ∈ {denso, bm25}; rank 1-based dentro do ramo; score = cosseno (denso) ou BM25 bruto."""
DTYPES = {"rank": "int32", "score": "float32"}
RAMOS = ("denso", "bm25")


def hash_consulta(consulta: str) -> str:
    """sha256 hex do texto exato da consulta (UTF-8, sem normalização): a chave que R5 reutiliza para reformulações."""
    return hashlib.sha256(consulta.encode("utf-8")).hexdigest()


def indice_id(dir_indice: Path) -> str:
    """`indice_id` do index_metadata.json, ou o nome do diretório quando não há metadados (índice de produção)."""
    meta = carregar_metadados_indice_rag(Path(dir_indice))
    return meta["indice_id"] if meta else Path(dir_indice).name


def caminho_cache(dir_indice: Path, k: int) -> Path:
    """runs/_shared/pools/<indice_id>_k<k>.parquet — o nome carrega o índice, logo um cache nunca é lido por outro índice."""
    return DIR_POOLS / f"{indice_id(dir_indice)}_k{k}.parquet"


def _topk_estavel(escores: np.ndarray, k: int) -> np.ndarray:
    """Índices dos k maiores escores, decrescente, empates pelo menor índice (a ordem de carga) — como `sorted(range(n), key=-escore)` do IndiceBM25."""
    ordem = np.lexsort((np.arange(len(escores)), -escores))
    return ordem[: min(k, len(escores))]


def buscar_pools(
    indice: IndiceHibrido, consultas: Sequence[str], k: int, vetorizador: Vetorizador, *, bloco: int = 64
) -> pd.DataFrame:
    """Top-k por ramo para cada consulta, como DataFrame COLUNAS.

    Denso: vetoriza as consultas em lote (o mesmo modelo dos chunks; embeddings normalizados) e faz o
    produto interno contra `indice.vetores` em blocos de `bloco` consultas. BM25: `tokenizar_pt` com o
    `remover_stopwords` do índice e `get_scores`; consulta sem tokens não gera linhas de BM25 (como o
    `buscar` da origem devolve []). Os dois ramos usam _topk_estavel — 45/556 consultas empatam na
    fronteira k=50 e 1.839 chunks de texto idêntico têm vetores idênticos.
    """
    consultas = list(consultas)
    linhas: list[tuple[str, str, int, str, float]] = []
    if consultas:
        q = vetorizador.codificar(consultas)
        for i0 in range(0, len(consultas), bloco):
            escores = (indice.vetores @ q[i0 : i0 + bloco].T).astype(np.float32)   # (n_chunks, b)
            for j in range(escores.shape[1]):
                h = hash_consulta(consultas[i0 + j])
                for r, i in enumerate(_topk_estavel(escores[:, j], k), 1):
                    linhas.append((h, "denso", r, str(indice.ids_denso[i]), float(escores[i, j])))
        for consulta in consultas:
            tokens = tokenizar_pt(consulta, remover_stopwords=indice.remover_stopwords)
            if not tokens:
                continue
            e = np.asarray(indice.bm25.get_scores(tokens), dtype=np.float32)
            h = hash_consulta(consulta)
            for r, i in enumerate(_topk_estavel(e, k), 1):
                linhas.append((h, "bm25", r, str(indice.ids_bm25[i]), float(e[i])))
    return pd.DataFrame(linhas, columns=list(COLUNAS)).astype(DTYPES)


class CachePools:
    """Pools persistidos em um parquet (COLUNAS), carregado inteiro; uma consulta está no cache se tem linhas do ramo denso."""

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self._pools: dict[str, dict[str, list[tuple[str, float]]]] = {}
        if self.caminho.exists():
            self._indexar(pd.read_parquet(self.caminho))

    def _indexar(self, df: pd.DataFrame) -> None:
        cols = ["consulta_hash", "ramo", "chunk_id", "score"]
        for h, ramo, cid, score in df.sort_values(["consulta_hash", "ramo", "rank"])[cols].itertuples(index=False):
            self._pools.setdefault(h, {r: [] for r in RAMOS})[ramo].append((cid, float(score)))

    def __len__(self) -> int:
        return len(self._pools)

    def __contains__(self, consulta: str) -> bool:
        return hash_consulta(consulta) in self._pools

    def faltantes(self, consultas: Sequence[str]) -> list[str]:
        """Consultas (únicas por hash, na ordem dada) ainda não buscadas."""
        vistos: set[str] = set()
        saida: list[str] = []
        for c in consultas:
            h = hash_consulta(c)
            if h not in self._pools and h not in vistos:
                vistos.add(h)
                saida.append(c)
        return saida

    def pools(self, consulta: str) -> dict[str, list[tuple[str, float]]]:
        """{'denso': [(chunk_id, score), …], 'bm25': […]} na ordem de rank; KeyError se a consulta não foi buscada."""
        return self._pools[hash_consulta(consulta)]

    def adicionar(self, df: pd.DataFrame) -> int:
        """Acrescenta as consultas de `df` ainda ausentes e regrava o parquet (tmp + os.replace); devolve quantas entraram."""
        novos = df[~df["consulta_hash"].isin(list(self._pools))]
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
