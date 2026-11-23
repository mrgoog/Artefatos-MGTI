"""R2: reordena os até POOL_RERANK primeiros da lista RRF de R1 pelo cross-encoder e corta em N_FINAL; sem dedup por documento."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
from ensemble_llm.esquemas import BlocoRecuperado

from rag2.constantes import N_FINAL, POOL_RERANK
from rag2.esquemas import BlocoReranqueado
from rag2.pool import hash_consulta
from rag2.recuperadores.base import validar_saida
from rag2.recuperadores.fusao import RecuperadorRRF
from rag2.reranker import COLUNAS, DTYPES, CacheReranker, Pontuador


class CacheIncompativel(RuntimeError):
    """O cache do reranker tem, para esta consulta, uma lista de chunks diferente da lista RRF atual (pool ou índice mudou)."""


class RecuperadorReranker:
    """base (R1) dá pools e lista RRF; pontuador dá o score; cache evita pontuar duas vezes; ordem = score desc, posição RRF asc."""

    def __init__(
        self,
        base: RecuperadorRRF,
        pontuador: Pontuador,
        cache: CacheReranker,
        *,
        pool_rerank: int = POOL_RERANK,
        n_final: int = N_FINAL,
        bloco: int = 50,
        degrau: str = "R2",
        log: Callable[[str], None] = print,
    ) -> None:
        self.base = base
        self.pontuador = pontuador
        self.cache = cache
        self.pool_rerank = pool_rerank
        self.n_final = n_final
        self.bloco = bloco
        self.degrau = degrau
        self.log = log

    def _lista(self, consulta: str) -> list[tuple[str, float]]:
        """Os até pool_rerank primeiros da lista RRF (busca os pools se faltarem)."""
        return self.base.fundir(consulta)[: self.pool_rerank]

    def preaquecer(self, consultas: Sequence[str]) -> int:
        """Pools (base) e depois scores das consultas ausentes, uma chamada de predict por consulta, persistindo a cada `bloco`; devolve quantas pontuou."""
        self.base.preaquecer(consultas)
        faltantes = self.cache.faltantes(consultas)
        linhas: list[tuple[str, int, str, float]] = []
        n_pares = 0
        for i, consulta in enumerate(faltantes, 1):
            ids = [c for c, _ in self._lista(consulta)]
            scores = self.pontuador.pontuar(consulta, [self.base.indice.texto[c] for c in ids])
            if len(scores) != len(ids):
                raise RuntimeError(f"pontuador devolveu {len(scores)} scores para {len(ids)} pares")
            h = hash_consulta(consulta)
            linhas.extend((h, r, c, float(s)) for r, (c, s) in enumerate(zip(ids, scores, strict=True), 1))
            n_pares += len(ids)
            if i % self.bloco == 0 or i == len(faltantes):
                self.cache.adicionar(pd.DataFrame(linhas, columns=list(COLUNAS)).astype(DTYPES))
                linhas = []
        self.log(
            f"reranker: {self.cache.caminho} | pontuadas={len(faltantes)} em_cache={len(consultas) - len(faltantes)} "
            f"pares={n_pares} modelo={self.pontuador.nome}"
        )
        return len(faltantes)

    def pontuados(self, consulta: str) -> list[tuple[str, float, float]]:
        """(chunk_id, rrf_score, rerank_score) na ordem RRF — o gancho de R4 (soma β e γ ao terceiro campo); CacheIncompativel se o cache não bate."""
        if consulta not in self.cache:
            self.preaquecer([consulta])
        lista = self._lista(consulta)
        cacheado = self.cache.scores(consulta)
        if [c for c, _ in lista] != [c for c, _ in cacheado]:
            raise CacheIncompativel(f"{self.cache.caminho}: lista RRF atual ≠ lista pontuada para {consulta[:60]!r}; apague o cache")
        return [(c, rrf, score) for (c, rrf), (_, score) in zip(lista, cacheado, strict=True)]

    @staticmethod
    def ordenar(pontuados: Sequence[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        """rerank_score desc; empate (textos idênticos dão scores idênticos) pela posição RRF — sorted é estável, a entrada está em ordem RRF."""
        return sorted(pontuados, key=lambda t: -t[2])

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]:
        """pontuados → ordenar → os n_final primeiros → BlocoReranqueado com os escores dos ramos (None se ausente), RRF e reranker."""
        escolhidos = self.ordenar(self.pontuados(consulta))[: self.n_final]
        p = self.base.cache.pools(consulta)
        denso, bm25 = dict(p["denso"]), dict(p["bm25"])
        blocos: list[BlocoRecuperado] = [
            BlocoReranqueado(
                chunk_id=cid,
                text=self.base.indice.texto[cid],
                source_doc=self.base.indice.doc[cid],
                dense_score=denso.get(cid),
                bm25_score=bm25.get(cid),
                combined_rank=i,
                rrf_score=rrf,
                rerank_score=score,
            )
            for i, (cid, rrf, score) in enumerate(escolhidos, 1)
        ]
        return validar_saida(blocos, n_max=self.n_final)
