"""Reranker falso determinístico sobre o índice sintético: cache por consulta, ordem por score com desempate pela posição RRF, corte, executor com rerank_score."""

import re

import numpy as np
import pandas as pd
import pytest
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.dados import ItemAvaliado
from rag2.escada import executar_degrau
from rag2.esquemas import BlocoReranqueado
from rag2.metricas import Casador
from rag2.pool import CachePools
from rag2.recuperadores.base import Recuperador
from rag2.recuperadores.fusao import RecuperadorRRF
from rag2.recuperadores.reranqueado import CacheIncompativel, RecuperadorReranker
from rag2.reranker import COLUNAS, DIR_RERANKER, CacheReranker, Pontuador, caminho_cache_reranker

Q = "aluguel entre pai e filho"
LEI2, LEI3, LEI1 = "Lei nº 2.txt#0000", "Lei nº 3.txt#0000", "Lei nº 1.txt#0000"
CARF1, CARF2 = "10000_1.txt#0000", "10000_2.txt#0000"


def _tokens(s):
    return set(re.findall(r"\w+", s.lower()))


class PontuadorFalso:
    """score = fração dos tokens da consulta presentes no texto; conta chamadas e pares para provar que o cache evita o modelo."""

    nome = "falso"

    def __init__(self):
        self.chamadas = 0
        self.pares = 0

    def pontuar(self, consulta, textos):
        self.chamadas += 1
        self.pares += len(textos)
        q = _tokens(consulta)
        return np.array([len(q & _tokens(t)) / len(q) for t in textos], dtype=np.float32)


def _r2(indice, tmp_path, **kw):
    log = kw.pop("log", lambda s: None)
    base = RecuperadorRRF(indice, CachePools(tmp_path / "pools.parquet"), VetorizadorHash(dim=16), k_pool=5, log=log)
    pontuador = kw.pop("pontuador", PontuadorFalso())
    return RecuperadorReranker(base, pontuador, CacheReranker(tmp_path / "rr.parquet"), log=log, **kw)


def test_caminho_cache_reranker(tmp_path, indice_sintetico):
    assert caminho_cache_reranker(indice_sintetico.diretorio, 50, 100) == DIR_RERANKER / "rag2_reparado_k50_r100_bge-reranker-v2-m3.parquet"
    assert caminho_cache_reranker(tmp_path / "x", 5, 10, modelo="org/m") == DIR_RERANKER / "x_k5_r10_m.parquet"


def test_pontuador_falso(indice_sintetico):
    p = PontuadorFalso()
    assert isinstance(p, Pontuador)
    s = p.pontuar(Q, [indice_sintetico.texto[c] for c in (LEI2, CARF1, CARF2, LEI1)])
    assert s.dtype == np.float32 and s.tolist() == [1.0, pytest.approx(0.2), pytest.approx(0.2), 0.0]
    assert p.chamadas == 1 and p.pares == 4


def test_recuperar_ordena_por_score_e_desempata_por_rrf(indice_sintetico, tmp_path):
    rec = _r2(indice_sintetico, tmp_path, n_final=5)
    assert isinstance(rec, Recuperador) and rec.degrau == "R2"
    blocos = rec.recuperar(Q)
    fund = rec.base.fundir(Q)
    pos = {c: i for i, (c, _) in enumerate(fund)}
    assert rec.pontuados(Q) == [(c, r, s) for (c, r), (_, s) in zip(fund, rec.cache.scores(Q))]
    assert [b.chunk_id for b in blocos] == sorted((c for c, _ in fund), key=lambda c: (-dict(rec.cache.scores(Q))[c], pos[c]))
    assert [b.rerank_score for b in blocos] == [1.0, 1.0, pytest.approx(0.2), pytest.approx(0.2), 0.0]
    assert [b.chunk_id for b in blocos[:2]] == [LEI2, LEI3] and blocos[4].chunk_id == LEI1   # empate 1,0: Lei 2 antes da 3 (posição RRF)
    assert [b.combined_rank for b in blocos] == [1, 2, 3, 4, 5] and all(isinstance(b, BlocoReranqueado) for b in blocos)
    p = rec.base.cache.pools(Q)
    denso, bm25, rrf = dict(p["denso"]), dict(p["bm25"]), dict(fund)
    for b in blocos:   # k_pool=5 = todos os chunks: cada bloco veio dos dois ramos
        assert b.rrf_score == rrf[b.chunk_id] and b.dense_score == denso[b.chunk_id] and b.bm25_score == bm25[b.chunk_id]
        assert b.text == indice_sintetico.texto[b.chunk_id] and b.source_doc == indice_sintetico.doc[b.chunk_id]


def test_pool_rerank_e_corte(indice_sintetico, tmp_path):
    rec = _r2(indice_sintetico, tmp_path, pool_rerank=2, n_final=7)
    fund = rec.base.fundir(Q)
    assert [c for c, _, _ in rec.pontuados(Q)] == [c for c, _ in fund[:2]]   # teto: pontua só os 2 primeiros por RRF
    assert len(rec.recuperar(Q)) == 2                                        # e serve ≤ n_final
    rec3 = _r2(indice_sintetico, tmp_path / "b", n_final=3)
    assert [b.rerank_score for b in rec3.recuperar(Q)] == [1.0, 1.0, pytest.approx(0.2)]


def test_cache_evita_o_modelo(indice_sintetico, tmp_path):
    p = PontuadorFalso()
    rec = _r2(indice_sintetico, tmp_path, pontuador=p)
    assert rec.preaquecer([Q, "lente intraocular"]) == 2 and (p.chamadas, p.pares) == (2, 10)
    assert rec.preaquecer([Q, "lente intraocular"]) == 0 and p.chamadas == 2
    rec.recuperar(Q)
    assert p.chamadas == 2
    df = pd.read_parquet(tmp_path / "rr.parquet")
    assert list(df.columns) == list(COLUNAS) and len(df) == 10 and str(df["rank"].dtype) == "int32" and str(df.score.dtype) == "float32"
    outro = PontuadorFalso()
    rec2 = _r2(indice_sintetico, tmp_path, pontuador=outro)   # mesmo cache em disco, novo processo
    assert len(rec2.cache) == 2 and rec2.cache.scores(Q) == rec.cache.scores(Q)
    assert [b.chunk_id for b in rec2.recuperar(Q)] == [b.chunk_id for b in rec.recuperar(Q)] and outro.chamadas == 0


def test_sob_demanda_e_faltantes(indice_sintetico, tmp_path):
    rec = _r2(indice_sintetico, tmp_path)
    assert "usufruto" not in rec.cache and rec.cache.faltantes([Q, Q, "usufruto"]) == [Q, "usufruto"]
    rec.recuperar("usufruto")                                  # fora do cache: busca pools e pontua só ela
    assert "usufruto" in rec.cache and "usufruto" in rec.base.cache and Q not in rec.cache
    assert rec.cache.faltantes([Q, Q, "usufruto"]) == [Q]
    with pytest.raises(KeyError):
        rec.cache.scores("inexistente")
    assert len(CacheReranker(tmp_path / "rr.parquet")) == 1


def test_cache_incompativel(indice_sintetico, tmp_path):
    _r2(indice_sintetico, tmp_path, pool_rerank=5).preaquecer([Q])
    with pytest.raises(CacheIncompativel):
        _r2(indice_sintetico, tmp_path, pool_rerank=2).pontuados(Q)   # mesmo arquivo, outra lista RRF


def test_logs_e_blocos(indice_sintetico, tmp_path):
    logs = []
    rec = _r2(indice_sintetico, tmp_path, bloco=2, log=logs.append)
    assert rec.preaquecer([Q, "lente intraocular", "usufruto"]) == 3
    assert logs[-2].startswith("pools: ") and "buscadas=3" in logs[-2]
    assert logs[-1].startswith("reranker: ") and logs[-1].endswith("| pontuadas=3 em_cache=0 pares=15 modelo=falso")
    assert len(CacheReranker(tmp_path / "rr.parquet")) == 3                # dois blocos (2 + 1) gravados


def _item(iid, pergunta, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=pergunta, ground_truth="g", has_doc_ref=True), esperados=frozenset(esperados)
    )


def test_executar_degrau_grava_rerank_score(indice_sintetico, tmp_path):
    logs = []
    rec = _r2(indice_sintetico, tmp_path, n_final=3, log=logs.append)
    itens = [_item("q_0001", Q, {("Lei nº 2.txt", "7")}), _item("q_0002", "lente intraocular", {("Lei nº 1.txt", "1")})]
    res = executar_degrau("R2", rec, itens, Casador(indice_sintetico.texto, indice_sintetico.doc), saida=tmp_path / "R2", b=200)
    assert [x.split(":")[0] for x in logs[:2]] == ["pools", "reranker"] and "pontuadas=2" in logs[1]   # pré-aquecimento antes do laço
    assert res["n_chunks_medio"] == 3.0 and res["degrau"] == "R2" and res["recall_passagem"] == 1.0
    servidos = pd.read_parquet(tmp_path / "R2" / "servidos.parquet")
    assert list(servidos.columns) == [
        "item_id", "combined_rank", "chunk_id", "source_doc", "dense_score", "bm25_score", "rrf_score", "rerank_score",
    ]
    assert len(servidos) == 6 and servidos["rerank_score"].between(0, 1).all()
    assert servidos[servidos.item_id == "q_0001"].chunk_id.iloc[0] == LEI2
