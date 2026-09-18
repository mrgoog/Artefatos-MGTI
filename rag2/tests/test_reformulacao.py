"""Reformulador com um ClienteLLM falso: uma chamada por pergunta, cache versionado (registrar uma vez), degradação em falha de parse. Sem gpu/llm."""

import json

import pandas as pd
import pytest
from ensemble_llm.agentes.contratos import ClienteLLM, ResultadoInvocacao

from rag2.pool import hash_consulta
from rag2.reformulacao import (
    COLUNAS,
    CacheReformulacoes,
    Reformulador,
    caminho_reformulacoes,
)

Q1 = "Quem deve declarar o IRPF de 2024?"
Q2 = "Como é tributado o aluguel entre pai e filho?"


class ClienteFalso:
    """Devolve um JSON de reformulação derivado da pergunta; conta invocações para provar que o cache evita o modelo. valid=False emite JSON inválido."""

    def __init__(self, *, valido=True):
        self.model = "falso"
        self.temperature = 0.0
        self.invocacoes = 0
        self.valido = valido

    def validate_model(self):
        return None

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        h = hash_consulta(prompt)[:6]
        if self.valido:
            corpo = json.dumps({"normativa": f"norm {h}", "decomposta": f"a {h} | b {h}", "literal": f"lit {h}"})
        else:
            corpo = "não é json"
        return ResultadoInvocacao(raw_output=corpo, prompt_hash=hash_consulta(prompt), model=self.model)


def test_cliente_falso_satisfaz_protocolo():
    assert isinstance(ClienteFalso(), ClienteLLM)


def test_gera_cacheia_e_persiste(tmp_path):
    cli = ClienteFalso()
    ref = Reformulador(cli, CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None)
    assert ref.preaquecer([Q1, Q2, Q1]) == 2 and cli.invocacoes == 2      # Q1 duplicada conta uma vez
    v = ref.reformular(Q1)
    assert len(v) == 3 and v[0].startswith("norm ") and " | " in v[1] and cli.invocacoes == 2   # já em cache
    d = pd.read_parquet(caminho_reformulacoes(tmp_path))
    assert list(d.columns) == list(COLUNAS) and len(d) == 2 and d.consulta_hash.nunique() == 2
    assert bool(d.valid_json.all()) is True


def test_registrar_uma_vez_novo_processo(tmp_path):
    Reformulador(ClienteFalso(), CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None).preaquecer([Q1])
    cli2 = ClienteFalso()
    ref2 = Reformulador(cli2, CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None)   # relê o parquet
    assert ref2.preaquecer([Q1]) == 0 and cli2.invocacoes == 0
    assert ref2.reformular(Q1)[0].startswith("norm ") and cli2.invocacoes == 0


def test_reformular_sob_demanda(tmp_path):
    cli = ClienteFalso()
    ref = Reformulador(cli, CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None)
    assert Q2 not in ref.cache
    v = ref.reformular(Q2)                                               # gera sob demanda
    assert cli.invocacoes == 1 and Q2 in ref.cache and v == ref.cache.variantes(Q2)


def test_falha_de_parse_degrada_para_original(tmp_path):
    cli = ClienteFalso(valido=False)
    ref = Reformulador(cli, CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None)
    assert ref.preaquecer([Q1]) == 1
    assert ref.reformular(Q1) == (Q1, Q1, Q1)                            # as três variantes viram a pergunta
    d = pd.read_parquet(caminho_reformulacoes(tmp_path))
    assert bool(d.valid_json.iloc[0]) is False


def test_n_reformulacoes_diferente_de_3_e_erro(tmp_path):
    with pytest.raises(ValueError):
        Reformulador(ClienteFalso(), CacheReformulacoes(caminho_reformulacoes(tmp_path)), n_reformulacoes=2)


def test_campo_vazio_degrada(tmp_path):
    class ClienteVazio(ClienteFalso):
        def invoke(self, prompt, *, system_prompt=None):
            self.invocacoes += 1
            corpo = json.dumps({"normativa": "norm", "decomposta": "a | b", "literal": "   "})
            return ResultadoInvocacao(raw_output=corpo, prompt_hash=hash_consulta(prompt), model=self.model)

    ref = Reformulador(ClienteVazio(), CacheReformulacoes(caminho_reformulacoes(tmp_path)), log=lambda s: None)
    assert ref.reformular(Q1) == (Q1, Q1, Q1)                            # JSON válido mas 'literal' vazio -> degrada
    d = pd.read_parquet(caminho_reformulacoes(tmp_path))
    assert bool(d.valid_json.iloc[0]) is False
