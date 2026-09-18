"""Retriever híbrido com união deduplicada de denso + BM25."""

from __future__ import annotations

import logging

from ensemble_llm.recuperacao.indice_bm25 import IndiceBM25
from ensemble_llm.recuperacao.indice_denso import IndiceDenso
from ensemble_llm.esquemas import BlocoDocumento, BlocoRecuperado

logger = logging.getLogger(__name__)


class RecuperadorHibrido:
    """Combina busca densa e lexical com união deduplicada e reranking por consenso.

    Após o merge dense + BM25, deduplica por ``source_doc``: mantém apenas o
    chunk de melhor rank por documento de origem.  Isso maximiza a cobertura
    documental no top-K — cada slot traz um documento distinto.
    """

    def __init__(
        self,
        indice_denso: IndiceDenso,
        indice_bm25: IndiceBM25,
        top_k_denso: int = 5,
        top_k_bm25: int = 5,
    ) -> None:
        if top_k_denso < 1 or top_k_bm25 < 1:
            raise ValueError("top_k deve ser >= 1")
        self.indice_denso = indice_denso
        self.indice_bm25 = indice_bm25
        self.top_k_denso = top_k_denso
        self.top_k_bm25 = top_k_bm25
        self._lookup_chunks: dict[str, BlocoDocumento] | None = None

    def _construir_lookup(self) -> dict[str, BlocoDocumento]:
        """Constrói mapa chunk_id → BlocoDocumento unindo os dois índices (inicialização lazy)."""
        lookup: dict[str, BlocoDocumento] = {}
        for c in self.indice_denso._chunks:
            lookup[c.chunk_id] = c
        for c in self.indice_bm25._chunks:
            if c.chunk_id not in lookup:
                lookup[c.chunk_id] = c
        return lookup

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]:
        """Recupera e reordena chunks: primeiro os presentes em ambos os índices, depois por escore denso.

        Após o ranking, deduplica por ``source_doc``: para cada documento de
        origem, mantém apenas o chunk de melhor ``combined_rank``.  Isso libera
        slots para documentos distintos, aumentando a cobertura documental
        efetiva sem alterar o índice nem os top-K dos sub-retrievers.
        """
        if self._lookup_chunks is None:
            self._lookup_chunks = self._construir_lookup()

        resultados_densos = dict(self.indice_denso.buscar(consulta, top_k=self.top_k_denso))
        resultados_bm25 = dict(self.indice_bm25.buscar(consulta, top_k=self.top_k_bm25))
        todos_ids = set(resultados_densos) | set(resultados_bm25)

        def chave_ordenacao(chunk_id: str) -> tuple[int, float, str]:
            em_ambos = chunk_id in resultados_densos and chunk_id in resultados_bm25
            escore_denso = resultados_densos.get(chunk_id, -float("inf"))
            return (0 if em_ambos else 1, -escore_denso, chunk_id)

        ids_ordenados = sorted(todos_ids, key=chave_ordenacao)

        # Deduplicar por source_doc: manter o chunk de melhor combined_rank
        # (primeiro na lista ordenada) por documento de origem.
        source_docs_vistos: set[str] = set()
        recuperados: list[BlocoRecuperado] = []
        rank_dedup = 0

        for chunk_id in ids_ordenados:
            chunk = self._lookup_chunks.get(chunk_id)
            if chunk is None:
                logger.warning("chunk_id %s não encontrado no lookup", chunk_id)
                continue

            source_doc = chunk.source_doc
            if source_doc in source_docs_vistos:
                continue
            source_docs_vistos.add(source_doc)

            rank_dedup += 1
            recuperados.append(
                BlocoRecuperado(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_doc=source_doc,
                    dense_score=resultados_densos.get(chunk_id),
                    bm25_score=resultados_bm25.get(chunk_id),
                    combined_rank=rank_dedup,
                )
            )
        return recuperados

