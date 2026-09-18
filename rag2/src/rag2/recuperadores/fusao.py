"""R1: fusão Reciprocal Rank Fusion dos pools denso e BM25, sem dedup por documento, corte em N_FINAL."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from ensemble_llm.esquemas import BlocoRecuperado
from ensemble_llm.recuperacao.vetorizacao import Vetorizador

from rag2.constantes import N_FINAL, POOL_POR_RAMO, RRF_K
from rag2.esquemas import BlocoFundido
from rag2.indice import IndiceHibrido
from rag2.pool import CachePools, buscar_pools
from rag2.recuperadores.base import validar_saida


def rrf(listas: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """score(chunk) = Σ 1/(k + rank) sobre as listas em que aparece (rank 1-based); ordenado por score desc, chunk_id asc."""
    acumulado: dict[str, float] = {}
    for lista in listas:
        for rank, cid in enumerate(lista, 1):
            acumulado[cid] = acumulado.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(acumulado.items(), key=lambda par: (-par[1], par[0]))


def cortar(
    fundidos: Sequence[tuple[str, float]], doc_de: Mapping[str, str], n_final: int, max_por_doc: int | None
) -> list[tuple[str, float]]:
    """Os primeiros n_final da lista fundida; com max_por_doc, pula chunks de documentos que já têm esse número."""
    por_doc: Counter[str] = Counter()
    saida: list[tuple[str, float]] = []
    for cid, score in fundidos:
        doc = doc_de[cid]
        if max_por_doc is not None and por_doc[doc] >= max_por_doc:
            continue
        por_doc[doc] += 1
        saida.append((cid, score))
        if len(saida) == n_final:
            break
    return saida


class RecuperadorRRF:
    """Pools de k_pool por ramo (do cache, buscando o que falta) → rrf → corte; blocos carregam os escores brutos e o RRF."""

    def __init__(
        self,
        indice: IndiceHibrido,
        cache: CachePools,
        vetorizador: Vetorizador,
        *,
        k_pool: int = POOL_POR_RAMO,
        k_rrf: int = RRF_K,
        n_final: int = N_FINAL,
        max_por_doc: int | None = None,
        degrau: str = "R1",
        log: Callable[[str], None] = print,
    ) -> None:
        if max_por_doc is not None and max_por_doc < 1:
            raise ValueError(f"max_por_doc deve ser ≥ 1 ou None, recebeu {max_por_doc}")
        self.indice = indice
        self.cache = cache
        self.vetorizador = vetorizador
        self.k_pool = k_pool
        self.k_rrf = k_rrf
        self.n_final = n_final
        self.max_por_doc = max_por_doc
        self.degrau = degrau
        self.log = log

    def preaquecer(self, consultas: Sequence[str]) -> int:
        """Busca em lote as consultas ausentes do cache e persiste; devolve quantas foram buscadas (0 = tudo em cache)."""
        faltantes = self.cache.faltantes(consultas)
        if faltantes:
            self.cache.adicionar(buscar_pools(self.indice, faltantes, self.k_pool, self.vetorizador))
        self.log(f"pools: {self.cache.caminho} | buscadas={len(faltantes)} em_cache={len(consultas) - len(faltantes)}")
        return len(faltantes)

    def _pools(self, consulta: str) -> dict[str, list[tuple[str, float]]]:
        if consulta not in self.cache:
            self.preaquecer([consulta])
        return self.cache.pools(consulta)

    def fundir(self, consulta: str) -> list[tuple[str, float]]:
        """Lista fundida completa (união dos dois pools) por RRF — R2 pontua os primeiros POOL_RERANK dela."""
        p = self._pools(consulta)
        return rrf(([c for c, _ in p["denso"]], [c for c, _ in p["bm25"]]), k=self.k_rrf)

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]:
        """Fusão → corte (n_final, max_por_doc) → BlocoFundido com dense_score/bm25_score do ramo (None se ausente) e rrf_score."""
        p = self._pools(consulta)
        denso, bm25 = dict(p["denso"]), dict(p["bm25"])
        escolhidos = cortar(self.fundir(consulta), self.indice.doc, self.n_final, self.max_por_doc)
        blocos: list[BlocoRecuperado] = [
            BlocoFundido(
                chunk_id=cid,
                text=self.indice.texto[cid],
                source_doc=self.indice.doc[cid],
                dense_score=denso.get(cid),
                bm25_score=bm25.get(cid),
                combined_rank=i,
                rrf_score=score,
            )
            for i, (cid, score) in enumerate(escolhidos, 1)
        ]
        return validar_saida(blocos, n_max=self.n_final)
