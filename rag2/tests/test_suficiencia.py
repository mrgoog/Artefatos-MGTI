"""Prompt, formatação do contexto e parser do juiz de suficiência vêm da origem via importlib (contexto ≡ o do agente; schema ≡ produção); laço por item com cliente falso (contexto vazio → 0 sem chamada; retomada). Sem gpu/llm."""

import json

import pandas as pd
from ensemble_llm.agentes.contratos import ResultadoInvocacao
from ensemble_llm.agentes.prompts_agentes import formatar_chunks_recuperados
from ensemble_llm.esquemas import BlocoRecuperado, ItemDataset

from rag2.constantes import MODELO_JUIZ, RUN_PRODUCAO
from rag2.suficiencia import (
    COLUNAS_SUFICIENCIA,
    formatar_contexto,
    julgar_suficiencia,
    modulo_suficiencia,
    parse_suficiencia,
    prompt_suficiencia,
)


def test_modulo_da_origem_carrega_uma_vez(cfg):
    m = modulo_suficiencia(cfg)
    assert m is modulo_suficiencia() and hasattr(m, "PROMPT_SISTEMA_SUFICIENCIA") and hasattr(m, "TEMPLATE_USUARIO_SUFICIENCIA")
    system, user = prompt_suficiencia(" Q? ", "[DOCUMENTO 1 — X]\ntexto", " GT ")
    assert system.startswith("Você é um auditor de sistemas de recuperação") and "VEREDICTOS" in system
    assert "PERGUNTA:\nQ?\n" in user and "CONTEXTO RECUPERADO (tudo o que o agente recebeu):\n[DOCUMENTO 1 — X]\ntexto" in user
    assert "RESPOSTA DE REFERÊNCIA" in user and "\nGT\n" in user and user.rstrip().endswith("responder à pergunta?")


def test_contexto_igual_ao_do_agente(cfg):
    blocos = [
        BlocoRecuperado(chunk_id="Lei nº 1.txt#0000", text=" Art. 1º Texto. ", source_doc="Lei nº 1.txt", combined_rank=1),
        BlocoRecuperado(chunk_id="10000_1.txt#0000", text="Acórdão.", source_doc="10000_1.txt", combined_rank=2),
    ]
    mapa = {b.chunk_id: {"source_doc": b.source_doc, "text": b.text} for b in blocos}
    assert formatar_contexto([b.chunk_id for b in blocos], mapa) == formatar_chunks_recuperados(blocos)   # o juiz vê o que o agente viu
    assert formatar_contexto([], mapa) == "(Nenhum documento recuperado para esta pergunta.)"


def test_parse_suficiencia(cfg):
    assert parse_suficiencia('{"raciocinio": "ok", "veredicto": 2}') == ("ok", 2)
    assert parse_suficiencia('```json\n{"raciocinio": "x", "veredicto": 1}\n```') == ("x", 1)
    assert parse_suficiencia('{"raciocinio": "cortado') == ("cortado", -1)
    assert parse_suficiencia('{"raciocinio": "x", "veredicto": 5}')[1] == -1


def test_schema_igual_ao_da_producao(cfg):
    prod = pd.read_parquet(cfg.dir_run(RUN_PRODUCAO) / "metrics" / "suficiencia_contexto.parquet")
    assert list(prod.columns) == list(COLUNAS_SUFICIENCIA) and len(prod) == 715 and set(prod.veredicto_suficiencia.unique()) <= {0, 1, 2, -1}


class ClienteSuficienciaFalso:
    """Veredicto fixo; mesma interface que o harness espera de ClienteOpenRouter."""

    def __init__(self, veredicto=1):
        self.model, self.temperature, self.max_tokens, self.veredicto, self.invocacoes = MODELO_JUIZ, 0.0, 4096, veredicto, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "r", "veredicto": self.veredicto}), prompt_hash="h", model=self.model, cost_usd=0.002)


def test_julgar_suficiencia_vazio_e_retomada(cfg, tmp_path):
    itens = [
        ItemDataset(item_id="q_0001", pergunta="A?", ground_truth="a", has_doc_ref=True),
        ItemDataset(item_id="q_0002", pergunta="B?", ground_truth="b", has_doc_ref=False),
    ]
    contexto = pd.DataFrame([{"item_id": "q_0001", "combined_rank": 1, "chunk_id": "L.txt#0000", "source_doc": "L.txt", "text": "Art. 1º"}])   # q_0002 sem contexto
    cli = ClienteSuficienciaFalso(2)
    caminho = tmp_path / "suf.parquet"
    df = julgar_suficiencia(itens, contexto, cli, caminho, log=lambda s: None)
    assert cli.invocacoes == 1 and list(df.columns) == list(COLUNAS_SUFICIENCIA) and caminho.exists()
    r = df.set_index("item_id")
    assert r.loc["q_0001"].to_dict() | {"raciocinio_juiz": ""} == {
        "has_doc_ref": True, "n_chunks": 1, "contexto_vazio": False, "veredicto_suficiencia": 2, "raciocinio_juiz": "", "cost_usd": 0.002, "judge_model": MODELO_JUIZ,
    }
    assert r.loc["q_0002", "veredicto_suficiencia"] == 0 and bool(r.loc["q_0002", "contexto_vazio"]) and r.loc["q_0002", "cost_usd"] == 0.0 and r.loc["q_0002", "n_chunks"] == 0
    cli2 = ClienteSuficienciaFalso(0)
    df2 = julgar_suficiencia(itens, contexto, cli2, caminho, log=lambda s: None)   # retomada: nada a julgar
    assert cli2.invocacoes == 0 and df2.veredicto_suficiencia.tolist() == [2, 0] and list(pd.read_parquet(caminho).columns) == list(COLUNAS_SUFICIENCIA)
