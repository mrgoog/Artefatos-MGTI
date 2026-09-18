"""Métricas canônicas sobre um índice sintético com valores calculáveis à mão."""

import pytest
from ensemble_llm.esquemas import ItemDataset

from rag2.dados import ItemAvaliado
from rag2.metricas import (
    CABECALHO_MARKDOWN,
    Casador,
    avaliar_item,
    avaliar_itens,
    linha_markdown,
    resumo,
    teto_oracular,
)

TEXTO = {
    "Lei A.txt#0000": "Art. 1º Preâmbulo. Art. 2º Corpo.",
    "Lei A.txt#0001": "Art. 25 Texto do vinte e cinco.",
    "Lei A.txt#0002": "Art. 250 Outro.",
    "123_456.txt#0000": "acórdão",
}


def _casador() -> Casador:
    return Casador(TEXTO, {c: c.split("#")[0] for c in TEXTO})


def _item(iid: str, esperados: set[tuple[str, str]]) -> ItemAvaliado:
    base = ItemDataset(item_id=iid, pergunta="p", ground_truth="g", has_doc_ref=True)
    return ItemAvaliado(item=base, esperados=frozenset(esperados))


ITEM1 = _item("q_0001", {("Lei A.txt", "25"), ("Lei A.txt", "1")})
ITEM2 = _item("q_0002", {("Lei A.txt", "2")})
ITEM3 = _item("q_0003", {("Lei A.txt", "99")})


def test_chunks_do_dispositivo():
    c = _casador()
    assert c.chunks_do_dispositivo("Lei A.txt", "25") == {"Lei A.txt#0001"}
    assert c.chunks_do_dispositivo("Lei A.txt", "1") == {"Lei A.txt#0000"}
    assert c.chunks_do_dispositivo("Lei A.txt", "2") == {"Lei A.txt#0000"}
    assert c.chunks_do_dispositivo("Lei A.txt", "99") == frozenset()
    assert c.chunks_do_dispositivo("Inexistente.txt", "1") == frozenset()


def test_arquivo_certo_chunk_errado_nao_conta_passagem():
    r = avaliar_item(ITEM1, ["Lei A.txt#0002", "123_456.txt#0000"], _casador())
    assert (r.hits_arquivo, r.hits_passagem, r.n_chunks, r.palavras) == (2, 0, 2, 4)
    assert r.recall_arquivo == 1.0
    assert r.recall_passagem == 0.0


def test_passagem_conta_quando_chunk_casa_o_artigo():
    r = avaliar_item(ITEM1, ["Lei A.txt#0001"], _casador())
    assert (r.hits_arquivo, r.hits_passagem) == (2, 1)  # os dois esperados são do mesmo arquivo
    assert r.recall_passagem == pytest.approx(0.5)


def test_chunk_desconhecido_e_item_sem_contexto():
    df = avaliar_itens([ITEM1, ITEM2], {"q_0001": ["Lei A.txt#0001", "fantasma#0000"]}, _casador())
    assert list(df["item_id"]) == ["q_0001", "q_0002"]
    assert df.loc[0, "n_chunks"] == 2 and df.loc[0, "palavras"] == 7
    assert df.loc[1, "n_chunks"] == 0 and df.loc[1, "recall_passagem"] == 0.0


def test_resumo_teto_e_markdown():
    c = _casador()
    df = avaliar_itens([ITEM1, ITEM2, ITEM3], {"q_0001": ["Lei A.txt#0001"]}, c)
    r = resumo(df, b=200)
    assert r["recall_passagem"] == pytest.approx(0.5 / 3)
    assert r["recall_arquivo"] == pytest.approx(1 / 3)
    assert r["hit_passagem"] == pytest.approx(1 / 3)
    assert r["frac_zero"] == pytest.approx(2 / 3)
    assert r["n_dispositivos"] == 4 and r["n_itens"] == 3 and r["b_bootstrap"] == 200
    lo, hi = r["ic_recall_passagem"]
    assert lo <= r["recall_passagem"] <= hi
    assert teto_oracular([ITEM1, ITEM2, ITEM3], c) == pytest.approx(0.75)
    linha = linha_markdown("Rx", r)
    assert linha.startswith("| Rx | 0.1667 [") and CABECALHO_MARKDOWN.startswith("| degrau |")
