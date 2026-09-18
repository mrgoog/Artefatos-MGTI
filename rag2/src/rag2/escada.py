"""Executor de um degrau: recupera para os 556 itens, avalia com as métricas canônicas e grava runs/escada/<degrau>/."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from rag2.artefatos import gravar_artefatos, ler_resumo
from rag2.config import RAIZ, ConfigRag2
from rag2.constantes import B_BOOTSTRAP, CANONICO_R0, TOLERANCIA_CANONICO
from rag2.dados import ItemAvaliado, tipo_fonte
from rag2.metricas import (
    CABECALHO_MARKDOWN,
    Casador,
    avaliar_itens,
    linha_markdown,
    resumo,
    teto_oracular,
)
from rag2.recuperadores.base import Recuperador

DIR_ESCADA = RAIZ / "runs" / "escada"
ORDEM_DEGRAUS = ("R0", "R0'", "R1", "R2", "R3", "R4", "R5")

CAMPOS_BLOCO = frozenset({"chunk_id", "text", "source_doc", "dense_score", "bm25_score", "combined_rank"})
"""Campos de BlocoRecuperado já cobertos pelas seis colunas fixas de servidos.parquet; o resto vira coluna extra."""


class GoldenDivergente(RuntimeError):
    """Rerodar um degrau já registrado produziu recall de passagem fora da tolerância do golden."""


def composicao_servido(servidos: dict[str, list[str]], casador: Casador) -> dict[str, float]:
    """Fração dos chunks servidos por tipo_fonte (norma, acordao_carf, pseudo_doc)."""
    cont = Counter(tipo_fonte(casador.doc[c]) for ids in servidos.values() for c in ids if c in casador.doc)
    total = sum(cont.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(cont.items())}


def executar_degrau(
    nome: str,
    recuperador: Recuperador,
    itens: Sequence[ItemAvaliado],
    casador: Casador,
    *,
    saida: Path | None = None,
    b: int = B_BOOTSTRAP,
    limite: int | None = None,
    sobrescrever: bool = False,
) -> dict:
    """Recupera item a item (ordem fixa), avalia, grava servidos/por_item/resumo e devolve o resumo.

    Regra registrar-uma-vez: se `saida/resumo.json` já existe e não há `sobrescrever`,
    nada é gravado; o novo resultado é comparado ao golden e GoldenDivergente é
    levantado se |Δ recall_passagem| > TOLERANCIA_CANONICO.
    """
    if limite:
        itens = list(itens)[:limite]
    if hasattr(recuperador, "preaquecer"):
        recuperador.preaquecer([it.pergunta for it in itens])   # R1+: pools em lote; R0/R0' não têm o método
    saida = Path(saida) if saida is not None else DIR_ESCADA / nome
    servidos: dict[str, list[str]] = {}
    linhas: list[dict] = []
    for it in itens:
        blocos = recuperador.recuperar(it.pergunta)
        servidos[it.item_id] = [bl.chunk_id for bl in blocos]
        for bl in blocos:
            extras = {k: v for k, v in bl.model_dump().items() if k not in CAMPOS_BLOCO}
            linhas.append(
                {
                    "item_id": it.item_id,
                    "combined_rank": bl.combined_rank,
                    "chunk_id": bl.chunk_id,
                    "source_doc": bl.source_doc,
                    "dense_score": bl.dense_score,
                    "bm25_score": bl.bm25_score,
                    **extras,
                }
            )
    df = avaliar_itens(itens, servidos, casador)
    res = resumo(df, b=b)
    res["degrau"] = nome
    res["teto_oracular"] = teto_oracular(itens, casador)
    res["composicao_servido"] = composicao_servido(servidos, casador)
    golden = ler_resumo(saida)
    if golden is not None and not sobrescrever:
        delta = res["recall_passagem"] - golden["recall_passagem"]
        res["delta_vs_golden"] = delta
        if abs(delta) > TOLERANCIA_CANONICO:
            raise GoldenDivergente(f"{nome}: recall_passagem={res['recall_passagem']:.4f} vs golden {golden['recall_passagem']:.4f}")
        return res
    gravar_artefatos(saida, df, res, nome)
    pd.DataFrame(linhas).to_parquet(saida / "servidos.parquet", index=False)
    return res


def checar_canonico(res: dict, alvo: float = CANONICO_R0, tol: float = TOLERANCIA_CANONICO) -> tuple[bool, float]:
    """(dentro da tolerância?, Δ) entre o recall de passagem do resumo e o valor canônico."""
    delta = res["recall_passagem"] - alvo
    return abs(delta) <= tol, delta


def tabela(dir_escada: Path = DIR_ESCADA) -> str:
    """Tabela markdown com uma linha por resumo.json em dir_escada, na ordem R0, R0', R1…; demais ao fim."""
    resumos = {p.name: ler_resumo(p) for p in sorted(dir_escada.iterdir()) if (p / "resumo.json").exists()}
    ordem = [d for d in ORDEM_DEGRAUS if d in resumos] + [d for d in resumos if d not in ORDEM_DEGRAUS]
    return "\n".join([CABECALHO_MARKDOWN] + [linha_markdown(d, resumos[d]) for d in ordem])


def fabrica_r0(cfg: ConfigRag2) -> Recuperador:
    """R0 sobre o índice de produção."""
    from rag2.recuperadores.producao import RecuperadorProducao

    return RecuperadorProducao(cfg.dir_indice_producao(), cfg.device)


def fabrica_r0_linha(cfg: ConfigRag2) -> Recuperador:
    """R0' — o recuperador de produção, inalterado (k = 5 + 5, união, dedup 1 chunk/doc), sobre o índice reparado."""
    from rag2.recuperadores.producao import RecuperadorProducao

    return RecuperadorProducao(cfg.dir_indice_reparado(), cfg.device, degrau="R0'")


def fabrica_r1(cfg: ConfigRag2, *, max_por_doc: int | None = None) -> Recuperador:
    """R1 — pool 50+50 do cache, RRF k=60, sem dedup, N=7, sobre o índice reparado; max_por_doc é a variante declarada."""
    from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

    from rag2.constantes import MODELO_EMBEDDER, POOL_POR_RAMO
    from rag2.indice import carregar_indice
    from rag2.pool import CachePools, caminho_cache
    from rag2.recuperadores.fusao import RecuperadorRRF

    dir_indice = cfg.dir_indice_reparado()
    vetorizador = VetorizadorSentenceTransformer(nome_modelo=MODELO_EMBEDDER, tamanho_lote=32, dispositivo=cfg.device)
    degrau = "R1" if max_por_doc is None else f"R1_maxdoc{max_por_doc}"
    return RecuperadorRRF(
        carregar_indice(dir_indice), CachePools(caminho_cache(dir_indice, POOL_POR_RAMO)), vetorizador,
        max_por_doc=max_por_doc, degrau=degrau,
    )


def fabrica_r2(cfg: ConfigRag2) -> Recuperador:
    """R2 — o R1 registrado (pools do cache, RRF, sem dedup) + cross-encoder fp16 sobre os até 100 primeiros, corte em 7; scores em cache por consulta."""
    from rag2.constantes import POOL_POR_RAMO, POOL_RERANK
    from rag2.recuperadores.fusao import RecuperadorRRF
    from rag2.recuperadores.reranqueado import RecuperadorReranker
    from rag2.reranker import CacheReranker, Reranker, caminho_cache_reranker

    base = fabrica_r1(cfg)
    assert isinstance(base, RecuperadorRRF)
    cache = CacheReranker(caminho_cache_reranker(cfg.dir_indice_reparado(), POOL_POR_RAMO, POOL_RERANK))
    return RecuperadorReranker(base, Reranker(cfg.device), cache)


def fabrica_r3(cfg: ConfigRag2) -> Recuperador:
    """R3 — R2 por subíndice (normas × CARF, IDF separado) com cota 5 + 2; embedder e reranker compartilhados e lazy; caches por subíndice."""
    from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

    from rag2.constantes import COTA_CARF, COTA_NORMAS, MODELO_EMBEDDER, POOL_POR_RAMO, POOL_RERANK
    from rag2.indice import carregar_indice
    from rag2.indices import dir_subindice
    from rag2.pool import CachePools, caminho_cache
    from rag2.recuperadores.cota import RecuperadorCota
    from rag2.recuperadores.fusao import RecuperadorRRF
    from rag2.recuperadores.reranqueado import RecuperadorReranker
    from rag2.reranker import CacheReranker, Reranker, caminho_cache_reranker

    vetorizador = VetorizadorSentenceTransformer(nome_modelo=MODELO_EMBEDDER, tamanho_lote=32, dispositivo=cfg.device)
    reranker = Reranker(cfg.device)   # um modelo, os dois subíndices; lazy
    cotas = {"norma": COTA_NORMAS, "acordao_carf": COTA_CARF}
    por_tipo: dict[str, RecuperadorReranker] = {}
    for tipo, cota in cotas.items():
        d = dir_subindice(cfg.dir_indice_reparado(), tipo)
        base = RecuperadorRRF(
            carregar_indice(d), CachePools(caminho_cache(d, POOL_POR_RAMO)), vetorizador, degrau=f"R3_{tipo}"
        )
        por_tipo[tipo] = RecuperadorReranker(
            base, reranker, CacheReranker(caminho_cache_reranker(d, POOL_POR_RAMO, POOL_RERANK)),
            n_final=cota, degrau=f"R3_{tipo}",
        )
    return RecuperadorCota(por_tipo=por_tipo, cotas=cotas)


def fabrica_r5(cfg: ConfigRag2) -> Recuperador:
    """R5 — R3 (cota 5+2, corte 7, dois subíndices) com pool multi-query: o reformulador é compartilhado; caches de pool reusados de R3, cache de reranker próprio (sufixo _mq)."""
    from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap
    from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

    from rag2.constantes import (
        COTA_CARF,
        COTA_NORMAS,
        MODELO_EMBEDDER,
        MODELO_REFORMULADOR,
        POOL_POR_RAMO,
        POOL_RERANK,
        SEMENTE,
    )
    from rag2.indice import carregar_indice
    from rag2.indices import dir_subindice
    from rag2.pool import CachePools, caminho_cache
    from rag2.recuperadores.cota import RecuperadorCota
    from rag2.recuperadores.multiquery import RecuperadorMultiQuery
    from rag2.recuperadores.reranqueado import RecuperadorReranker
    from rag2.reformulacao import (
        MAX_TOKENS_REFORMULACAO,
        CacheReformulacoes,
        Reformulador,
        caminho_reformulacoes,
    )
    from rag2.reranker import CacheReranker, Reranker, caminho_cache_reranker

    vetorizador = VetorizadorSentenceTransformer(nome_modelo=MODELO_EMBEDDER, tamanho_lote=32, dispositivo=cfg.device)
    reranker = Reranker(cfg.device)
    cliente = ClienteLlamaSwap(MODELO_REFORMULADOR, temperature=0.0, max_tokens=MAX_TOKENS_REFORMULACAO, seed=SEMENTE)
    reformulador = Reformulador(cliente, CacheReformulacoes(caminho_reformulacoes(DIR_ESCADA / "R5")))
    cotas = {"norma": COTA_NORMAS, "acordao_carf": COTA_CARF}
    por_tipo: dict[str, RecuperadorReranker] = {}
    for tipo, cota in cotas.items():
        d = dir_subindice(cfg.dir_indice_reparado(), tipo)
        base = RecuperadorMultiQuery(
            carregar_indice(d), CachePools(caminho_cache(d, POOL_POR_RAMO)), vetorizador, reformulador, degrau=f"R5_{tipo}"
        )
        cache_rr = caminho_cache_reranker(d, POOL_POR_RAMO, POOL_RERANK)
        cache_rr = cache_rr.with_stem(cache_rr.stem + "_mq")   # lista pontuada difere de R3 (união de 4 variantes)
        por_tipo[tipo] = RecuperadorReranker(base, reranker, CacheReranker(cache_rr), n_final=cota, degrau=f"R5_{tipo}")
    return RecuperadorCota(por_tipo=por_tipo, cotas=cotas, degrau="R5")


FABRICAS: dict[str, Callable[[ConfigRag2], Recuperador]] = {
    "R0": fabrica_r0, "R0'": fabrica_r0_linha, "R1": fabrica_r1, "R2": fabrica_r2, "R3": fabrica_r3, "R5": fabrica_r5,
}
"""Degrau → construtor; milestones seguintes registram os seus aqui."""
