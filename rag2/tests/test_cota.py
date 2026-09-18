"""RecuperadorCota sobre os subíndices do índice sintético com um pontuador fixo: cotas 5+2, reordenação por rerank_score entre tipos, slot vazio não compensado, cache por subíndice, executor."""

import numpy as np
import pandas as pd
import pytest
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.dados import ItemAvaliado, tipo_fonte
from rag2.escada import executar_degrau
from rag2.esquemas import BlocoReranqueado
from rag2.indices import dividir_indice
from rag2.metricas import Casador
from rag2.pool import CachePools
from rag2.recuperadores.base import Recuperador
from rag2.recuperadores.cota import RecuperadorCota
from rag2.recuperadores.fusao import RecuperadorRRF
from rag2.recuperadores.reranqueado import RecuperadorReranker
from rag2.reranker import CacheReranker

Q = "aluguel entre pai e filho"
LEI1, LEI2, LEI3 = "Lei nº 1.txt#0000", "Lei nº 2.txt#0000", "Lei nº 3.txt#0000"
CARF1, CARF2 = "10000_1.txt#0000", "10000_2.txt#0000"


class PontuadorFixo:
    """score fixo por texto do chunk (LEI2 e LEI3 têm o mesmo texto → o mesmo score); conta chamadas/pares para provar que o cache evita o modelo."""

    nome = "fixo"

    def __init__(self, por_texto):
        self.por_texto = por_texto
        self.chamadas = 0
        self.pares = 0

    def pontuar(self, consulta, textos):
        self.chamadas += 1
        self.pares += len(textos)
        return np.array([self.por_texto[t] for t in textos], dtype=np.float32)


def _mapa(indice, por_id):
    """{texto → score} a partir de {chunk_id → score} (textos idênticos colapsam, como no índice real)."""
    return {indice.texto[cid]: s for cid, s in por_id.items()}


def _r3(indice, tmp_path, cotas, pontuador, *, n_final=7, log=lambda s: None):
    subs = dividir_indice(indice)
    por_tipo = {}
    for tipo, cota in cotas.items():
        base = RecuperadorRRF(subs[tipo], CachePools(tmp_path / f"pools_{tipo}.parquet"), VetorizadorHash(dim=16), k_pool=5, log=log)
        por_tipo[tipo] = RecuperadorReranker(base, pontuador, CacheReranker(tmp_path / f"rr_{tipo}.parquet"), n_final=cota, log=log)
    return RecuperadorCota(por_tipo=por_tipo, cotas=cotas, n_final=n_final, log=log)


def test_cotas_e_reindexacao(indice_sintetico, tmp_path):
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r3(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p)
    assert isinstance(rec, Recuperador) and rec.degrau == "R3"
    blocos = rec.recuperar(Q)
    assert [b.chunk_id for b in blocos] == [LEI2, LEI3, CARF1]                 # 2 normas (0,9; empate LEI2<LEI3 por RRF) + 1 CARF, reordenados por score
    assert [b.rerank_score for b in blocos] == [pytest.approx(0.9), pytest.approx(0.9), pytest.approx(0.5)]
    assert [b.combined_rank for b in blocos] == [1, 2, 3] and all(isinstance(b, BlocoReranqueado) for b in blocos)
    assert [tipo_fonte(b.source_doc) for b in blocos] == ["norma", "norma", "acordao_carf"]


def test_reordena_por_score_entre_tipos(indice_sintetico, tmp_path):
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.4, LEI3: 0.4, CARF1: 0.95, CARF2: 0.2}))
    rec = _r3(indice_sintetico, tmp_path, {"norma": 1, "acordao_carf": 1}, p)
    blocos = rec.recuperar(Q)
    assert tipo_fonte(blocos[0].source_doc) == "acordao_carf" and blocos[0].chunk_id == CARF1   # CARF (0,95) antes da norma (0,4): reordenado por score, não por tipo
    assert tipo_fonte(blocos[1].source_doc) == "norma"
    assert [b.combined_rank for b in blocos] == [1, 2]


def test_slot_vazio_nao_compensado(indice_sintetico, tmp_path):
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.3, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r3(indice_sintetico, tmp_path, {"norma": 5, "acordao_carf": 2}, p)   # só há 3 normas
    blocos = rec.recuperar(Q)
    n_norma = sum(tipo_fonte(b.source_doc) == "norma" for b in blocos)
    n_carf = sum(tipo_fonte(b.source_doc) == "acordao_carf" for b in blocos)
    assert (n_norma, n_carf) == (3, 2) and len(blocos) == 5                     # normas não é preenchida a 5 por CARF
    assert [b.combined_rank for b in blocos] == [1, 2, 3, 4, 5]


def test_preaquecer_cache_por_subindice(indice_sintetico, tmp_path):
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r3(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p)
    assert rec.preaquecer([Q, "usufruto"]) == {"norma": 2, "acordao_carf": 2}
    chamadas = p.chamadas
    assert (tmp_path / "rr_norma.parquet").exists() and (tmp_path / "rr_acordao_carf.parquet").exists()
    assert rec.preaquecer([Q, "usufruto"]) == {"norma": 0, "acordao_carf": 0} and p.chamadas == chamadas   # segundo pré-aquecimento não chama o modelo
    rec.recuperar(Q)
    assert p.chamadas == chamadas                                              # recuperar após cache também não chama
    dn = pd.read_parquet(tmp_path / "rr_norma.parquet")
    da = pd.read_parquet(tmp_path / "rr_acordao_carf.parquet")
    assert dn.consulta_hash.nunique() == 2 and da.consulta_hash.nunique() == 2 and len(dn) == 6 and len(da) == 4


def _item(iid, pergunta, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=pergunta, ground_truth="g", has_doc_ref=True), esperados=frozenset(esperados)
    )


def test_executar_degrau_grava_composicao(indice_sintetico, tmp_path):
    logs = []
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r3(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p, log=logs.append)
    itens = [_item("q_0001", Q, {("Lei nº 2.txt", "7")}), _item("q_0002", "lente intraocular", {("Lei nº 1.txt", "1")})]
    res = executar_degrau("R3", rec, itens, Casador(indice_sintetico.texto, indice_sintetico.doc), saida=tmp_path / "R3", b=200)
    assert [x.split(":")[0] for x in logs[:4]] == ["pools", "reranker", "pools", "reranker"]   # pré-aquecimento dos dois subíndices antes do laço
    assert res["degrau"] == "R3" and res["n_chunks_medio"] == 3.0
    servidos = pd.read_parquet(tmp_path / "R3" / "servidos.parquet")
    assert list(servidos.columns) == [
        "item_id", "combined_rank", "chunk_id", "source_doc", "dense_score", "bm25_score", "rrf_score", "rerank_score",
    ]
    assert len(servidos) == 6 and servidos.rerank_score.between(0, 1).all()
    comp = servidos.assign(t=servidos.source_doc.map(tipo_fonte)).groupby("t").size().to_dict()
    assert comp == {"norma": 4, "acordao_carf": 2}                              # 2 normas + 1 CARF por item, 2 itens
