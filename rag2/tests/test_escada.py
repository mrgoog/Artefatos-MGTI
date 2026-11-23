"""Executor da escada sobre um recuperador falso: artefatos, resumo e regra registrar-uma-vez."""

import pandas as pd
import pytest
from ensemble_llm.esquemas import BlocoRecuperado, ItemDataset

from rag2.dados import ItemAvaliado
from rag2.escada import GoldenDivergente, checar_canonico, executar_degrau, tabela
from rag2.metricas import Casador
from rag2.recuperadores.base import Recuperador, validar_saida

TEXTO = {
    "Lei A.txt#0000": "Art. 1º Preâmbulo.",
    "Lei A.txt#0001": "Art. 25 Texto.",
    "123_456.txt#0000": "acórdão",
}


def _item(iid, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=f"pergunta {iid}", ground_truth="g", has_doc_ref=True),
        esperados=frozenset(esperados),
    )


ITENS = [_item("q_0001", {("Lei A.txt", "25")}), _item("q_0002", {("Lei A.txt", "1")})]


class Falso:
    """Devolve sempre os mesmos chunk_ids, na ordem dada."""

    degrau = "RX"

    def __init__(self, ids):
        self.ids = ids

    def recuperar(self, consulta):
        return [
            BlocoRecuperado(chunk_id=c, text=TEXTO[c], source_doc=c.split("#")[0], combined_rank=i)
            for i, c in enumerate(self.ids, 1)
        ]


def _casador():
    return Casador(TEXTO, {c: c.split("#")[0] for c in TEXTO})


def test_protocolo_e_validacao():
    assert isinstance(Falso(["Lei A.txt#0001"]), Recuperador)
    blocos = Falso(["Lei A.txt#0001", "123_456.txt#0000"]).recuperar("x")
    assert validar_saida(blocos, n_max=2) is blocos
    with pytest.raises(ValueError):
        validar_saida(blocos, n_max=1)
    with pytest.raises(ValueError):
        validar_saida(Falso(["Lei A.txt#0001", "Lei A.txt#0001"]).recuperar("x"))
    blocos[1] = blocos[1].model_copy(update={"combined_rank": 5})
    with pytest.raises(ValueError):
        validar_saida(blocos)


def test_executar_degrau_grava_artefatos(tmp_path):
    saida = tmp_path / "RX"
    res = executar_degrau("RX", Falso(["Lei A.txt#0001", "123_456.txt#0000"]), ITENS, _casador(), saida=saida, b=200)
    assert res["degrau"] == "RX" and res["n_itens"] == 2
    assert res["recall_passagem"] == pytest.approx(0.5)
    assert res["composicao_servido"] == {"acordao_carf": 0.5, "norma": 0.5}
    assert res["teto_oracular"] == 1.0
    servidos = pd.read_parquet(saida / "servidos.parquet")
    assert len(servidos) == 4 and list(servidos.columns[:3]) == ["item_id", "combined_rank", "chunk_id"]
    assert (saida / "por_item.parquet").exists() and (saida / "resumo.md").exists()
    assert "| RX | 0.5000 [" in tabela(tmp_path)


def test_registrar_uma_vez(tmp_path):
    executar_degrau("RX", Falso(["Lei A.txt#0001"]), ITENS, _casador(), saida=tmp_path, b=200)
    mtime = (tmp_path / "resumo.json").stat().st_mtime_ns
    res = executar_degrau("RX", Falso(["Lei A.txt#0001"]), ITENS, _casador(), saida=tmp_path, b=200)
    assert res["delta_vs_golden"] == 0.0
    assert (tmp_path / "resumo.json").stat().st_mtime_ns == mtime
    with pytest.raises(GoldenDivergente):
        executar_degrau("RX", Falso(["123_456.txt#0000"]), ITENS, _casador(), saida=tmp_path, b=200)
    res2 = executar_degrau("RX", Falso(["123_456.txt#0000"]), ITENS, _casador(), saida=tmp_path, b=200, sobrescrever=True)
    assert res2["recall_passagem"] == 0.0


def test_limite_e_checar_canonico(tmp_path):
    res = executar_degrau("RX", Falso(["Lei A.txt#0001"]), ITENS, _casador(), saida=tmp_path, b=200, limite=1)
    assert res["n_itens"] == 1
    assert checar_canonico({"recall_passagem": 0.2355}) == (True, pytest.approx(0.0004))
    assert checar_canonico({"recall_passagem": 0.2400})[0] is False
