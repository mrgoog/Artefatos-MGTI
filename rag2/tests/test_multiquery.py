"""RecuperadorMultiQuery + R5 via RecuperadorCota sobre o índice sintético, com um reformulador de reformulações fixas. Sem gpu/llm."""

import numpy as np
import pandas as pd
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.dados import ItemAvaliado, tipo_fonte
from rag2.escada import executar_degrau
from rag2.esquemas import BlocoReranqueado
from rag2.indices import dividir_indice
from rag2.metricas import Casador
from rag2.pool import CachePools, hash_consulta
from rag2.recuperadores.base import Recuperador
from rag2.recuperadores.cota import RecuperadorCota
from rag2.recuperadores.fusao import rrf
from rag2.recuperadores.multiquery import RecuperadorMultiQuery
from rag2.recuperadores.reranqueado import RecuperadorReranker
from rag2.reranker import CacheReranker


class PontuadorFixo:
    """score fixo por texto do chunk (idênticos dão o mesmo score); conta chamadas para provar que o cache evita o modelo."""

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
    """{texto -> score} a partir de {chunk_id -> score} (textos idênticos colapsam)."""
    return {indice.texto[cid]: s for cid, s in por_id.items()}


Q = "aluguel entre pai e filho"
LEI1, LEI2, LEI3 = "Lei nº 1.txt#0000", "Lei nº 2.txt#0000", "Lei nº 3.txt#0000"
CARF1, CARF2 = "10000_1.txt#0000", "10000_2.txt#0000"
# reformulações fixas: tokens que casam com chunks distintos, para a união multi-query ter efeito
REFORM = {Q: ("rendimentos de aluguel", "lente intraocular despesa médica", "usufruto de acórdão")}


class ReformuladorFixo:
    """Reformulações de um dicionário; conta gerações para provar 'uma vez, compartilhado'. Mesma interface do Reformulador real."""

    def __init__(self, mapa):
        self.mapa = mapa
        self.model = "fixo"
        self.geracoes = 0
        self._cache = set()

    def preaquecer(self, consultas):
        novas = [c for c in dict.fromkeys(consultas) if c not in self._cache]
        for c in novas:
            self._cache.add(c)
        self.geracoes += len(novas)
        return len(novas)

    def reformular(self, consulta):
        if consulta not in self._cache:
            self.preaquecer([consulta])
        return self.mapa[consulta]


def _base(indice, tmp_path, ref, *, k_pool=5, log=lambda s: None):
    return RecuperadorMultiQuery(indice, CachePools(tmp_path / "pools.parquet"), VetorizadorHash(dim=16), ref, k_pool=k_pool, log=log)


def test_fundir_une_as_quatro_variantes(indice_sintetico, tmp_path):
    subs = dividir_indice(indice_sintetico)
    ref = ReformuladorFixo(REFORM)
    base = _base(subs["norma"], tmp_path, ref)
    base.preaquecer([Q])                                                 # aquece pools da original + 3 reformulações
    variantes = [Q, *REFORM[Q]]
    assert all(v in base.cache for v in variantes) and ref.geracoes == 1
    listas = []
    for v in variantes:
        p = base.cache.pools(v)
        listas += [[c for c, _ in p["denso"]], [c for c, _ in p["bm25"]]]
    assert base.fundir(Q) == rrf(listas, k=base.k_rrf)                   # fundir = RRF sobre as 8 listas


def _r5(indice, tmp_path, cotas, pontuador, ref, *, n_final=7, log=lambda s: None):
    subs = dividir_indice(indice)
    por_tipo = {}
    for tipo, cota in cotas.items():
        base = RecuperadorMultiQuery(subs[tipo], CachePools(tmp_path / f"pools_{tipo}.parquet"), VetorizadorHash(dim=16), ref, k_pool=5, log=log)
        por_tipo[tipo] = RecuperadorReranker(base, pontuador, CacheReranker(tmp_path / f"rr_{tipo}.parquet"), n_final=cota, log=log)
    return RecuperadorCota(por_tipo=por_tipo, cotas=cotas, n_final=n_final, degrau="R5", log=log)


def test_r5_cota_e_reindexacao(indice_sintetico, tmp_path):
    ref = ReformuladorFixo(REFORM)
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r5(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p, ref)
    assert isinstance(rec, Recuperador) and rec.degrau == "R5"
    blocos = rec.recuperar(Q)
    assert [b.chunk_id for b in blocos] == [LEI2, LEI3, CARF1]            # 2 normas (0,9; empate por RRF) + 1 CARF, reordenados por score
    assert [b.combined_rank for b in blocos] == [1, 2, 3] and all(isinstance(b, BlocoReranqueado) for b in blocos)
    assert [tipo_fonte(b.source_doc) for b in blocos] == ["norma", "norma", "acordao_carf"]


def test_reformulador_compartilhado_gera_uma_vez(indice_sintetico, tmp_path):
    ref = ReformuladorFixo(REFORM)
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r5(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p, ref)
    rec.preaquecer([Q])
    assert ref.geracoes == 1                                             # os dois subíndices compartilham a instância
    chamadas = p.chamadas
    rec.recuperar(Q)
    assert p.chamadas == chamadas and ref.geracoes == 1                  # recuperar após aquecer não chama nem o reranker nem o reformulador


def test_pool_cache_tem_as_quatro_variantes(indice_sintetico, tmp_path):
    ref = ReformuladorFixo(REFORM)
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r5(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p, ref)
    rec.preaquecer([Q])
    dn = pd.read_parquet(tmp_path / "pools_norma.parquet")
    assert dn.consulta_hash.nunique() == 4                               # original + 3 reformulações
    assert {hash_consulta(v) for v in [Q, *REFORM[Q]]} == set(dn.consulta_hash.unique())


def _item(iid, pergunta, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=pergunta, ground_truth="g", has_doc_ref=True), esperados=frozenset(esperados)
    )


def test_executar_degrau_r5(indice_sintetico, tmp_path):
    logs = []
    ref = ReformuladorFixo(REFORM | {"lente intraocular": ("lente ocular", "cirurgia de catarata", "prótese ocular")})
    p = PontuadorFixo(_mapa(indice_sintetico, {LEI1: 0.1, LEI2: 0.9, LEI3: 0.9, CARF1: 0.5, CARF2: 0.2}))
    rec = _r5(indice_sintetico, tmp_path, {"norma": 2, "acordao_carf": 1}, p, ref, log=logs.append)
    itens = [_item("q_0001", Q, {("Lei nº 2.txt", "7")}), _item("q_0002", "lente intraocular", {("Lei nº 1.txt", "1")})]
    res = executar_degrau("R5", rec, itens, Casador(indice_sintetico.texto, indice_sintetico.doc), saida=tmp_path / "R5", b=200)
    assert res["degrau"] == "R5" and res["n_chunks_medio"] == 3.0
    servidos = pd.read_parquet(tmp_path / "R5" / "servidos.parquet")
    assert list(servidos.columns) == [
        "item_id", "combined_rank", "chunk_id", "source_doc", "dense_score", "bm25_score", "rrf_score", "rerank_score",
    ]
    assert len(servidos) == 6 and servidos.rerank_score.between(0, 1).all()
    comp = servidos.assign(t=servidos.source_doc.map(tipo_fonte)).groupby("t").size().to_dict()
    assert comp == {"norma": 4, "acordao_carf": 2}                        # 2 normas + 1 CARF por item, 2 itens
