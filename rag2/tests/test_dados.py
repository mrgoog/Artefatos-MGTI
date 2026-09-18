"""Contagens de referência do dataset e das amostras fixas (DESIGN §1, §2.3, §3)."""

import pytest

from rag2.dados import (
    amostra_estratificada,
    carregar_itens,
    carregar_normas,
    contar_palavras,
    ids_amostra_200,
    itens_com_dispositivo,
    tipo_fonte,
)


@pytest.fixture(scope="module")
def itens(cfg):
    return carregar_itens(cfg)


def test_715_itens_com_ids_q_nnnn(itens):
    assert len(itens) == 715
    assert {i.item_id for i in itens} == {f"q_{n:04d}" for n in range(1, 716)}
    assert sum(i.has_doc_ref for i in itens) == 600


def test_556_itens_com_dispositivo(itens):
    avaliados = itens_com_dispositivo(itens)
    assert len(avaliados) == 556
    assert sum(len(a.esperados) for a in avaliados) == 1742
    assert len({arq for a in avaliados for arq, _ in a.esperados}) == 158


def test_amostra_200_reproduz_o_piloto(cfg, itens):
    ids = ids_amostra_200(cfg)
    assert len(ids) == 200
    assert {i.item_id for i in amostra_estratificada(itens, 200)} == set(ids)
    assert len(set(ids) & {a.item_id for a in itens_com_dispositivo(itens)}) == 151


def test_amostra_60_estratificada(itens):
    amostra = amostra_estratificada(itens, 60)
    assert len(amostra) == 60
    assert sum(i.has_doc_ref for i in amostra) == 50
    assert [i.item_id for i in amostra] == [i.item_id for i in amostra_estratificada(itens, 60)]


def test_tipo_fonte():
    assert tipo_fonte("10865722385201112_6949202.txt") == "acordao_carf"
    assert tipo_fonte("Lei nº 9.250.txt") == "norma"
    assert tipo_fonte("disp_q_0001_00") == "pseudo_doc"
    assert tipo_fonte("carf_q_0010_02") == "pseudo_doc"


def test_carregar_normas(cfg):
    normas = carregar_normas(cfg.arquivo_normas())
    assert len(normas) == 478
    nomes = {n["filename"] for n in normas}
    assert "Lei nº 10.406.txt" in nomes
    assert min(contar_palavras(n["filedata"]) for n in normas) == 27  # Medida Provisória nº 252.txt


def test_contar_palavras():
    assert contar_palavras(" a  b\nc ") == 3
    assert contar_palavras("") == 0
