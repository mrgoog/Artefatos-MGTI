"""O índice de produção é lido alinhado e com a composição documentada (DESIGN §1, diagnóstico §3)."""

import pytest

from rag2.indice import carregar_indice


@pytest.fixture(scope="module")
def indice(cfg):
    return carregar_indice(cfg.dir_indice_producao())


def test_tamanhos_e_alinhamento(indice):
    assert len(indice) == 66865
    assert indice.vetores.shape == (66865, 1024)
    assert set(indice.ids_bm25) == set(indice.ids_denso)
    assert indice.remover_stopwords is True


def test_composicao(indice):
    comp = indice.composicao()
    assert comp["documentos"] == {"acordao_carf": 7204, "norma": 478, "pseudo_doc": 568}
    assert comp["chunks"] == {"acordao_carf": 60435, "norma": 5862, "pseudo_doc": 568}


def test_lookup_por_chunk_id(indice):
    cid = "Instrução Normativa RFB nº 2.178.txt#0001"
    assert indice.doc[cid] == "Instrução Normativa RFB nº 2.178.txt"
    assert len(indice.texto[cid]) > 0
