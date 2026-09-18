"""Pools em lote sobre o índice sintético: hash, id do cache, top-k estável, escores, consulta sem tokens e persistência."""

import hashlib

import numpy as np
import pytest
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.pool import DIR_POOLS, COLUNAS, CachePools, _topk_estavel, buscar_pools, caminho_cache, hash_consulta, indice_id

Q = "aluguel entre pai e filho"
ORDEM_BM25_Q = ["Lei nº 2.txt#0000", "Lei nº 3.txt#0000", "10000_2.txt#0000", "10000_1.txt#0000", "Lei nº 1.txt#0000"]


def test_hash_e_indice_id(tmp_path, cfg):
    assert hash_consulta(Q) == hashlib.sha256(Q.encode("utf-8")).hexdigest() and hash_consulta(Q) != hash_consulta(Q + " ")
    assert indice_id(tmp_path / "sem_metadados") == "sem_metadados"
    assert caminho_cache(tmp_path / "sem_metadados", 50) == DIR_POOLS / "sem_metadados_k50.parquet"
    assert indice_id(cfg.dir_indice_producao()) == cfg.dir_indice_producao().name   # produção não tem index_metadata.json


def test_indice_id_dos_metadados(indice_sintetico):
    assert indice_id(indice_sintetico.diretorio) == "rag2_reparado"


def test_topk_estavel():
    e = np.array([0.5, 0.9, 0.5, 0.9, 0.1], dtype=np.float32)
    assert list(_topk_estavel(e, 3)) == [1, 3, 0]
    assert list(_topk_estavel(e, 10)) == [1, 3, 0, 2, 4]


def test_buscar_pools_forma_escores_e_empates(indice_sintetico):
    idx, vet = indice_sintetico, VetorizadorHash(dim=16)
    df = buscar_pools(idx, [Q, "lente intraocular"], 5, vet)
    assert list(df.columns) == list(COLUNAS) and len(df) == 20
    assert str(df["rank"].dtype) == "int32" and str(df["score"].dtype) == "float32"
    b = df[(df.consulta_hash == hash_consulta(Q)) & (df.ramo == "bm25")].sort_values("rank")
    assert list(b.chunk_id) == ORDEM_BM25_Q and list(b["rank"]) == [1, 2, 3, 4, 5]
    assert b.score.iloc[0] == b.score.iloc[1] > b.score.iloc[2] > b.score.iloc[3] == b.score.iloc[4] == 0
    d = df[(df.consulta_hash == hash_consulta(Q)) & (df.ramo == "denso")].sort_values("rank")
    esperado = idx.vetores @ vet.codificar([Q])[0]
    assert list(d.score) == pytest.approx(sorted(esperado, reverse=True))
    ids = list(d.chunk_id)
    assert ids.index("Lei nº 2.txt#0000") < ids.index("Lei nº 3.txt#0000")   # vetores idênticos → empate exato → ordem de carga
    assert d.set_index("chunk_id").score["Lei nº 2.txt#0000"] == d.set_index("chunk_id").score["Lei nº 3.txt#0000"]
    lente = df[(df.consulta_hash == hash_consulta("lente intraocular")) & (df.ramo == "bm25")].sort_values("rank")
    assert list(lente.chunk_id[:2]) == ["10000_1.txt#0000", "Lei nº 1.txt#0000"]


def test_consulta_sem_tokens_bm25(indice_sintetico):
    df = buscar_pools(indice_sintetico, ["de a o"], 3, VetorizadorHash(dim=16))
    assert df.ramo.tolist() == ["denso"] * 3
    assert buscar_pools(indice_sintetico, [], 3, VetorizadorHash(dim=16)).empty


def test_cache_persistencia(indice_sintetico, tmp_path):
    caminho = tmp_path / "pools" / "x_k5.parquet"
    cache = CachePools(caminho)
    assert len(cache) == 0 and cache.faltantes([Q, Q, "lente intraocular"]) == [Q, "lente intraocular"]
    df = buscar_pools(indice_sintetico, [Q, "lente intraocular"], 5, VetorizadorHash(dim=16))
    assert cache.adicionar(df) == 2 and cache.adicionar(df) == 0 and caminho.exists()
    cache2 = CachePools(caminho)
    assert len(cache2) == 2 and cache2.pools(Q) == cache.pools(Q) and cache2.faltantes([Q]) == [] and Q in cache2
    assert [c for c, _ in cache2.pools(Q)["bm25"]] == ORDEM_BM25_Q
    with pytest.raises(KeyError):
        cache2.pools("inexistente")
    assert cache2.adicionar(buscar_pools(indice_sintetico, ["de a o"], 5, VetorizadorHash(dim=16))) == 1
    assert cache2.pools("de a o")["bm25"] == [] and len(cache2.pools("de a o")["denso"]) == 5
    assert len(CachePools(caminho)) == 3
