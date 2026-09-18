"""R5: RecuperadorRRF cuja fusão une os pools das 4 variantes (original + 3 reformulações) antes do reranker (DESIGN §4.5)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ensemble_llm.recuperacao.vetorizacao import Vetorizador

from rag2.constantes import N_FINAL, POOL_POR_RAMO, RRF_K
from rag2.indice import IndiceHibrido
from rag2.pool import CachePools
from rag2.recuperadores.fusao import RecuperadorRRF, rrf
from rag2.reformulacao import Reformulador


class RecuperadorMultiQuery(RecuperadorRRF):
    """Como RecuperadorRRF, mas fundir(consulta) faz RRF sobre as 8 listas (4 variantes × ramos denso/BM25) do mesmo subíndice.

    A única diferença de R3 para R5: a fonte do pool. Downstream (RecuperadorReranker, RecuperadorCota,
    executor) compõe este objeto sem alteração — o reranker ainda pontua base.fundir(consulta)[:pool_rerank]
    contra a pergunta ORIGINAL (relevância ao que foi perguntado; as reformulações só ampliam o pool).
    """

    def __init__(
        self,
        indice: IndiceHibrido,
        cache: CachePools,
        vetorizador: Vetorizador,
        reformulador: Reformulador,
        *,
        k_pool: int = POOL_POR_RAMO,
        k_rrf: int = RRF_K,
        n_final: int = N_FINAL,
        degrau: str = "R5",
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(indice, cache, vetorizador, k_pool=k_pool, k_rrf=k_rrf, n_final=n_final, degrau=degrau, log=log)
        self.reformulador = reformulador

    def _variantes(self, consulta: str) -> list[str]:
        """[original, normativa, decomposta, literal] — reformula (e cacheia) se faltar."""
        return [consulta, *self.reformulador.reformular(consulta)]

    def preaquecer(self, consultas: Sequence[str]) -> int:
        """Reformula as consultas faltantes, depois busca em lote os pools de TODAS as variantes (original + 3); devolve pools buscados.

        Chama RecuperadorRRF.preaquecer (super) sobre a lista achatada de variantes, para não recursar em si mesmo.
        """
        self.reformulador.preaquecer(consultas)
        variantes: list[str] = []
        for c in consultas:
            variantes.extend(self._variantes(c))
        return super().preaquecer(variantes)

    def fundir(self, consulta: str) -> list[tuple[str, float]]:
        """RRF sobre as 8 listas das 4 variantes (busca — e loga — só as variantes que faltam no cache)."""
        variantes = self._variantes(consulta)
        faltam = self.cache.faltantes(variantes)
        if faltam:
            super().preaquecer(faltam)   # nunca chama self.preaquecer (sem recursão); silencioso se tudo em cache
        listas: list[list[str]] = []
        for v in variantes:
            p = self.cache.pools(v)
            listas.append([c for c, _ in p["denso"]])
            listas.append([c for c, _ in p["bm25"]])
        return rrf(listas, k=self.k_rrf)
