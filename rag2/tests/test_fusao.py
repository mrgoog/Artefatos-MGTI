"""RRF com exemplos à mão, corte com e sem max_por_doc, RecuperadorRRF sobre o índice sintético e o executor com rrf_score."""

import pandas as pd
import pytest
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.dados import ItemAvaliado
from rag2.escada import executar_degrau
from rag2.esquemas import BlocoFundido
from rag2.metricas import Casador
from rag2.pool import CachePools
from rag2.recuperadores.base import Recuperador
from rag2.recuperadores.fusao import RecuperadorRRF, cortar, rrf

Q = "aluguel entre pai e filho"


def test_rrf_exemplo_a_mao():
    assert rrf((["a", "b", "c"], ["c", "d"])) == [
        ("c", pytest.approx(1 / 63 + 1 / 61)), ("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 62)), ("d", pytest.approx(1 / 62)),
    ]
    assert [c for c, _ in rrf((["a", "b", "c"], ["c", "d"]))] == ["c", "a", "b", "d"]   # b e d empatam: chunk_id asc


def test_rrf_bordas():
    assert rrf((["x"], [])) == [("x", pytest.approx(1 / 61))]
    assert rrf(([], [])) == []
    assert rrf((["b"], ["a"]), k=1) == [("a", 0.5), ("b", 0.5)]


def test_cortar():
    fund = [("A#0", 0.9), ("A#1", 0.8), ("B#0", 0.7), ("A#2", 0.6), ("C#0", 0.5)]
    doc = {c: c.split("#")[0] for c, _ in fund}
    assert cortar(fund, doc, 3, None) == fund[:3]
    assert cortar(fund, doc, 3, 1) == [("A#0", 0.9), ("B#0", 0.7), ("C#0", 0.5)]
    assert cortar(fund, doc, 10, 2) == [("A#0", 0.9), ("A#1", 0.8), ("B#0", 0.7), ("C#0", 0.5)]
    assert cortar([], doc, 7, None) == []


def _rec(indice, tmp_path, **kw):
    return RecuperadorRRF(
        indice, CachePools(tmp_path / "pools.parquet"), VetorizadorHash(dim=16),
        k_pool=kw.pop("k_pool", 5), log=kw.pop("log", lambda s: None), **kw,
    )


def test_recuperar_blocos_e_escores(indice_sintetico, tmp_path):
    rec = _rec(indice_sintetico, tmp_path, n_final=3)
    assert isinstance(rec, Recuperador) and rec.degrau == "R1"
    blocos = rec.recuperar(Q)
    p = rec.cache.pools(Q)
    esperado = rrf(([c for c, _ in p["denso"]], [c for c, _ in p["bm25"]]))
    assert len(esperado) == 5 and rec.fundir(Q) == esperado
    assert [(b.chunk_id, b.rrf_score) for b in blocos] == esperado[:3]
    assert [b.combined_rank for b in blocos] == [1, 2, 3] and all(isinstance(b, BlocoFundido) for b in blocos)
    denso, bm25 = dict(p["denso"]), dict(p["bm25"])
    for b in blocos:   # k_pool=5 = todos os chunks: cada bloco veio dos dois ramos
        assert b.dense_score == denso[b.chunk_id] and b.bm25_score == bm25[b.chunk_id]
        assert b.text == indice_sintetico.texto[b.chunk_id] and b.source_doc == indice_sintetico.doc[b.chunk_id]


def test_escores_none_fora_do_ramo(indice_sintetico, tmp_path):
    rec = _rec(indice_sintetico, tmp_path, k_pool=2, n_final=4)
    blocos = rec.recuperar(Q)
    p = rec.cache.pools(Q)
    assert [c for c, _ in p["bm25"]] == ["Lei nº 2.txt#0000", "Lei nº 3.txt#0000"]
    ids_d, ids_b = {c for c, _ in p["denso"]}, {c for c, _ in p["bm25"]}
    assert 2 <= len(blocos) <= 4
    for b in blocos:
        assert (b.dense_score is None) == (b.chunk_id not in ids_d) and (b.bm25_score is None) == (b.chunk_id not in ids_b)


def test_preaquecer_e_busca_sob_demanda(indice_sintetico, tmp_path):
    logs = []
    rec = _rec(indice_sintetico, tmp_path, log=logs.append)
    assert rec.preaquecer(["lente intraocular", "usufruto"]) == 2 and "buscadas=2 em_cache=0" in logs[-1]
    assert rec.preaquecer(["lente intraocular", "usufruto"]) == 0 and "buscadas=0 em_cache=2" in logs[-1]
    assert Q not in rec.cache
    rec.recuperar(Q)                                   # consulta fora do cache: busca só ela e persiste
    assert Q in rec.cache and len(CachePools(tmp_path / "pools.parquet")) == 3


def test_max_por_doc(indice_sintetico, tmp_path):
    assert _rec(indice_sintetico, tmp_path, max_por_doc=3, degrau="R1_maxdoc3").degrau == "R1_maxdoc3"
    with pytest.raises(ValueError):
        _rec(indice_sintetico, tmp_path, max_por_doc=0)


def _item(iid, pergunta, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=pergunta, ground_truth="g", has_doc_ref=True), esperados=frozenset(esperados)
    )


def test_executar_degrau_grava_rrf_score(indice_sintetico, tmp_path):
    logs = []
    rec = _rec(indice_sintetico, tmp_path, n_final=3, log=logs.append)
    itens = [_item("q_0001", Q, {("Lei nº 2.txt", "7")}), _item("q_0002", "lente intraocular", {("Lei nº 1.txt", "1")})]
    res = executar_degrau("R1", rec, itens, Casador(indice_sintetico.texto, indice_sintetico.doc), saida=tmp_path / "R1", b=200)
    assert logs and logs[0].startswith("pools: ") and "buscadas=2" in logs[0]   # pré-aquecimento antes do laço
    assert res["n_chunks_medio"] == 3.0 and res["degrau"] == "R1"
    servidos = pd.read_parquet(tmp_path / "R1" / "servidos.parquet")
    assert list(servidos.columns) == ["item_id", "combined_rank", "chunk_id", "source_doc", "dense_score", "bm25_score", "rrf_score"]
    assert len(servidos) == 6 and servidos["rrf_score"].notna().all()
