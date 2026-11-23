"""Harness com recuperador, agentes, juiz e auditor falsos: schemas, retomada (nenhuma chamada na 2ª execução), z=0 sem juiz em JSON inválido, recusa → auditor, resumo e golden. Sem gpu/llm."""

import json

import pandas as pd
import pytest
from ensemble_llm.agentes.contratos import ResultadoInvocacao
from ensemble_llm.esquemas import BlocoRecuperado, ItemDataset
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.constantes import MODELO_JUIZ
from rag2.escada import GoldenDivergente
from rag2.geracao import (
    COLUNAS_AUDITORIA,
    COLUNAS_CONTEXTO,
    COLUNAS_RESPOSTAS,
    COLUNAS_ROTULOS,
    amostra_smoke,
    executar_variante,
    tabela_prompts,
)
from rag2.pool import hash_consulta
from rag2.prompts import VarianteP0

RECUSA = "Não é possível responder com base no material fornecido."
DISPOSITIVO = '{"Lei nº 1": [{"título": "Lei nº 1", "artigos": [{"artigo": "2º", "incisos": [], "parágrafos": []}], "file": "Lei nº 1.txt"}]}'
"""Anotação do curador no formato que `dispositivos_esperados` do canônico lê (JSON com `file` e `artigos`)."""
ITENS = [
    ItemDataset(item_id="q_0001", pergunta="Quem declara?", ground_truth="Quem ganhou acima do limite.", dispositivos_legais=[DISPOSITIVO], has_doc_ref=True),
    ItemDataset(item_id="q_0002", pergunta="Aluguel entre pai e filho?", ground_truth="Tributa.", ementas_carf=["Acórdão x"], has_doc_ref=True),
    ItemDataset(item_id="q_0003", pergunta="Sem referência?", ground_truth="Sim.", has_doc_ref=False),
]


class RecuperadorFalso:
    degrau = "RX"

    def __init__(self):
        self.chamadas = 0

    def recuperar(self, consulta):
        self.chamadas += 1
        h = hash_consulta(consulta)[:4]
        return [
            BlocoRecuperado(chunk_id=f"Lei nº {h}.txt#0000", text=f"Art. 1º Norma {h} sim.", source_doc=f"Lei nº {h}.txt", combined_rank=1),
            BlocoRecuperado(chunk_id=f"1{h}_1.txt#0000", text=f"Acórdão {h}.", source_doc=f"1{h}_1.txt", combined_rank=2),
        ]


class ClienteFalso:
    """Agente: JSON válido com 'sim' (juiz falso → z=1) salvo `recusa` (texto de recusa) e `invalido` (não é JSON); conta invocações."""

    def __init__(self, model, *, recusa=(), invalido=()):
        self.model, self.temperature, self.max_tokens = model, 0.0, 1024
        self.recusa, self.invalido, self.invocacoes = set(recusa), set(invalido), 0

    def validate_model(self):
        return None

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        pergunta = prompt.split("PERGUNTA:\n")[1].split("\n")[0]
        if pergunta in self.invalido:
            corpo = "não é json"
        elif pergunta in self.recusa:
            corpo = json.dumps({"resposta": RECUSA, "confianca": 0.1})
        else:
            corpo = json.dumps({"resposta": f"sim, {pergunta}", "confianca": 0.9})
        return ResultadoInvocacao(raw_output=corpo, prompt_hash=hash_consulta(prompt), model=self.model)


class JuizFalso:
    """z=1 se a candidata contém 'sim'; mesma interface que ExecutorJuiz espera de ClienteOpenRouter."""

    def __init__(self):
        self.model, self.temperature, self.max_tokens, self.invocacoes = MODELO_JUIZ, 0.0, 1024, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        cand = prompt.split("RESPOSTA CANDIDATA A AVALIAR:\n")[1]
        z = 1 if "sim" in cand else 0
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "análise curta suficiente", "veredicto": z}), prompt_hash=hash_consulta(prompt), model=self.model, cost_usd=0.003)


class AuditorFalso:
    def __init__(self, veredicto=1):
        self.model, self.temperature, self.veredicto, self.invocacoes = MODELO_JUIZ, 0.0, veredicto, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        assert "CONTEXTO RECUPERADO (tudo o que o agente viu):\n[DOCUMENTO 1 — Lei nº" in prompt   # o auditor vê o contexto do agente
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "r", "veredicto": self.veredicto}), prompt_hash="h", model=self.model, cost_usd=0.001)


def _montar(tmp_path, *, recusa=("Aluguel entre pai e filho?",), invalido=("Sem referência?",)):
    agentes = {"a1": ClienteFalso("m1", recusa=recusa, invalido=invalido), "a2": ClienteFalso("m2")}
    juiz = JuizFalso()
    executor = ExecutorJuiz(client=juiz, cache=CacheRotulosJuiz(tmp_path / "judge", MODELO_JUIZ))   # type: ignore[arg-type]
    return RecuperadorFalso(), agentes, juiz, executor, AuditorFalso()


def test_executar_variante_grava_tudo(tmp_path):
    rec, agentes, juiz, executor, auditor = _montar(tmp_path)
    saida = tmp_path / "PX"
    res = executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, executor, auditor, saida=saida, b=200, log=lambda s: None)
    assert rec.chamadas == 3 and agentes["a1"].invocacoes == 3 + 2 and agentes["a2"].invocacoes == 3   # inválido: k_max=3 tentativas
    assert juiz.invocacoes == 4 and auditor.invocacoes == 1   # 6 respostas − 1 JSON inválido − 1 hit (a1 e a2 respondem igual em q_0001: mesmo prompt_hash do juiz); 1 recusa
    r1 = pd.read_parquet(saida / "agent_responses" / "a1" / "responses.parquet")
    assert list(r1.columns) == list(COLUNAS_RESPOSTAS) and len(r1) == 3 and r1.prompt_hash.str.len().eq(64).all()
    assert r1.set_index("item_id").loc["q_0003", "valid_json"] == False and pd.isna(r1.set_index("item_id").loc["q_0003", "response_text"])   # noqa: E712
    c = pd.read_parquet(saida / "contexto.parquet")
    assert list(c.columns) == list(COLUNAS_CONTEXTO) and len(c) == 6 and list(r1.retrieved_chunk_ids.iloc[0]) == list(c[c.item_id == "q_0001"].chunk_id)
    rot = pd.read_parquet(saida / "rotulos.parquet").set_index(["item_id", "agent_id"])
    assert list(rot.reset_index().columns) == list(COLUNAS_ROTULOS) and len(rot) == 6
    assert rot.loc[("q_0003", "a1"), "z_agent"] == 0 and rot.loc[("q_0003", "a1"), "prompt_hash_juiz"] == ""   # sem juiz
    assert rot.loc[("q_0002", "a1"), "z_agent"] == 0 and rot.loc[("q_0001", "a1"), "z_agent"] == 1 and rot.z_agent.sum() == 4
    aud = pd.read_parquet(saida / "auditoria_recusas.parquet")
    assert list(aud.columns) == list(COLUNAS_AUDITORIA) and aud.to_dict("records")[0] | {"raciocinio_auditor": ""} == {"item_id": "q_0002", "agent_id": "a1", "veredicto_recusa": 1, "raciocinio_auditor": "", "cost_usd": 0.001}
    itens = pd.read_parquet(saida / "itens.parquet")
    assert itens.com_dispositivo.tolist() == [True, False, False] and itens.has_doc_ref.tolist() == [True, True, False]   # só q_0001 tem dispositivo casável
    assert res["variante"] == "PX" and res["degrau"] == "RX" and res["n_itens"] == 3 and res["n_respostas"] == 6
    assert res["adequacao_media"] == pytest.approx(4 / 6) and res["por_agente"]["a1"] == {"adequacao": pytest.approx(1 / 3), "n_adequadas": 1, "n_recusas": 1, "n_evasivas": 1, "n_falhas_parse": 1}
    assert res["taxa_recusa"] == pytest.approx(1 / 6) and res["taxa_evasiva"] == pytest.approx(1 / 6) and res["taxa_falha_parse"] == pytest.approx(1 / 6)
    assert res["n_chunks_medio"] == 2.0 and res["palavras_medias"] == pytest.approx(7.0) and len(res["ic_adequacao_media"]) == 2
    assert (saida / "por_item.parquet").exists() and (saida / "resumo.md").exists() and "| PX | RX |" in tabela_prompts(tmp_path)


def test_segunda_execucao_nao_chama_modelo(tmp_path):
    rec, agentes, juiz, executor, auditor = _montar(tmp_path)
    saida = tmp_path / "PX"
    executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, executor, auditor, saida=saida, b=200, log=lambda s: None)
    rec2, agentes2, juiz2, executor2, auditor2 = _montar(tmp_path)                       # novos objetos, mesmo cache do juiz em disco
    res = executar_variante("PX", "RX", rec2, ITENS, VarianteP0(), agentes2, executor2, auditor2, saida=saida, b=200, log=lambda s: None)
    assert rec2.chamadas == 0 and all(a.invocacoes == 0 for a in agentes2.values()) and juiz2.invocacoes == 0 and auditor2.invocacoes == 0
    assert res["delta_vs_golden"] == 0.0 and res["adequacao_media"] == pytest.approx(4 / 6)


def test_retomada_parcial_e_cache_do_juiz(tmp_path):
    rec, agentes, juiz, executor, auditor = _montar(tmp_path)
    saida = tmp_path / "PX"
    executar_variante("PX", "RX", rec, ITENS[:2], VarianteP0(), agentes, executor, auditor, saida=saida, b=200, log=lambda s: None)
    n = {aid: a.invocacoes for aid, a in agentes.items()}
    executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, executor, auditor, saida=saida, b=200, sobrescrever=True, log=lambda s: None)
    assert rec.chamadas == 3 and agentes["a1"].invocacoes == n["a1"] + 3 and agentes["a2"].invocacoes == n["a2"] + 1   # só q_0003 é gerado
    assert juiz.invocacoes == 4 and len(pd.read_parquet(saida / "rotulos.parquet")) == 6   # q_0001: a1 e a2 compartilham o rótulo (hit)
    assert len(executor.cache) == 4 and (tmp_path / "judge" / MODELO_JUIZ.replace("/", "_") / "cache.parquet").exists()


def test_golden_divergente_e_degrau_diferente(tmp_path):
    rec, agentes, juiz, executor, auditor = _montar(tmp_path)
    saida = tmp_path / "PX"
    executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, executor, auditor, saida=saida, b=200, log=lambda s: None)
    with pytest.raises(GoldenDivergente):
        executar_variante("PX", "RY", rec, ITENS, VarianteP0(), agentes, executor, auditor, saida=saida, b=200, log=lambda s: None)
    (saida / "rotulos.parquet").unlink()                                                    # rejulgar com outro juiz muda os z
    class JuizZero(JuizFalso):
        def invoke(self, prompt, *, system_prompt=None):
            return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "análise curta suficiente", "veredicto": 0}), prompt_hash=hash_consulta(prompt) + "x", model=self.model)
    ex0 = ExecutorJuiz(client=JuizZero(), cache=CacheRotulosJuiz(tmp_path / "judge0", MODELO_JUIZ))   # type: ignore[arg-type]
    with pytest.raises(GoldenDivergente):
        executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, ex0, auditor, saida=saida, b=200, log=lambda s: None)
    res = executar_variante("PX", "RX", rec, ITENS, VarianteP0(), agentes, ex0, auditor, saida=saida, b=200, sobrescrever=True, log=lambda s: None)
    assert res["adequacao_media"] == 0.0 and "delta_vs_golden" not in res


def test_amostra_smoke(cfg):
    from rag2.dados import carregar_itens, itens_com_dispositivo

    itens = carregar_itens(cfg)
    a = amostra_smoke(itens)
    ids = [it.item_id for it in a]
    assert len(a) == 20 and ids == sorted(ids) and ids[0] == "q_0054" and ids[-1] == "q_0695"
    assert sum(it.has_doc_ref for it in a) == 17 and len(set(ids) & {x.item_id for x in itens_com_dispositivo(itens)}) == 15
