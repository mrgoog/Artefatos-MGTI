"""Confirmação com recuperador, agentes, juiz, auditor e juiz de suficiência falsos: artefatos, suficiência por item, custo.json, 2ª execução sem chamadas, golden e retomada; os 200 do piloto. Sem gpu/llm."""

import json

import pandas as pd
import pytest
from ensemble_llm.agentes.contratos import ResultadoInvocacao
from ensemble_llm.esquemas import BlocoRecuperado, ItemDataset
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.confirmacao import executar_confirmacao, itens_confirmacao
from rag2.constantes import MODELO_JUIZ
from rag2.escada import GoldenDivergente
from rag2.pool import hash_consulta
from rag2.prompts import VarianteP0
from rag2.suficiencia import COLUNAS_SUFICIENCIA

RECUSA = "Não é possível responder com base no material fornecido."
DISPOSITIVO = '{"Lei nº 1": [{"título": "Lei nº 1", "artigos": [{"artigo": "2º", "incisos": [], "parágrafos": []}], "file": "Lei nº 1.txt"}]}'
ITENS = [
    ItemDataset(item_id="q_0001", pergunta="Quem declara?", ground_truth="Quem ganhou acima do limite.", dispositivos_legais=[DISPOSITIVO], has_doc_ref=True),
    ItemDataset(item_id="q_0002", pergunta="Aluguel entre pai e filho?", ground_truth="Tributa.", ementas_carf=["Acórdão x"], has_doc_ref=True),
    ItemDataset(item_id="q_0003", pergunta="Sem referência?", ground_truth="Sim.", has_doc_ref=False),
]
SUFICIENCIA = {"Quem declara?": 2, "Aluguel entre pai e filho?": 1, "Sem referência?": 0}


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
    """Agente: JSON válido com 'sim' (juiz falso → z=1) salvo `recusa` (texto de recusa) e `invalido` (não é JSON)."""

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
    def __init__(self):
        self.model, self.temperature, self.max_tokens, self.invocacoes = MODELO_JUIZ, 0.0, 1024, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        z = 1 if "sim" in prompt.split("RESPOSTA CANDIDATA A AVALIAR:\n")[1] else 0
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "análise curta suficiente", "veredicto": z}), prompt_hash=hash_consulta(prompt), model=self.model, cost_usd=0.003)


class AuditorFalso:
    def __init__(self):
        self.model, self.temperature, self.invocacoes = MODELO_JUIZ, 0.0, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "r", "veredicto": 1}), prompt_hash="h", model=self.model, cost_usd=0.001)


class SuficienciaFalsa:
    """Veredicto fixo por pergunta; confere que o prompt é o da origem e que o contexto é o do agente."""

    def __init__(self):
        self.model, self.temperature, self.max_tokens, self.invocacoes = MODELO_JUIZ, 0.0, 4096, 0

    def invoke(self, prompt, *, system_prompt=None):
        self.invocacoes += 1
        assert system_prompt.startswith("Você é um auditor de sistemas de recuperação")
        assert "CONTEXTO RECUPERADO (tudo o que o agente recebeu):\n[DOCUMENTO 1 — Lei nº" in prompt
        pergunta = prompt.split("PERGUNTA:\n")[1].split("\n")[0]
        return ResultadoInvocacao(raw_output=json.dumps({"raciocinio": "r", "veredicto": SUFICIENCIA[pergunta]}), prompt_hash="h", model=self.model, cost_usd=0.002)


def _rodar(tmp_path, saida, degrau="RX"):
    agentes = {"a1": ClienteFalso("m1", recusa=("Aluguel entre pai e filho?",), invalido=("Sem referência?",)), "a2": ClienteFalso("m2")}
    rec, juiz, auditor, suf = RecuperadorFalso(), JuizFalso(), AuditorFalso(), SuficienciaFalsa()
    executor = ExecutorJuiz(client=juiz, cache=CacheRotulosJuiz(tmp_path / "judge", MODELO_JUIZ))   # type: ignore[arg-type]
    res = executar_confirmacao(
        degrau, rec, ITENS, VarianteP0(), agentes, executor, auditor, suf, saida=saida, b=200, ambiente={"device": "cpu"}, log=lambda s: None,
    )
    return res, rec, agentes, juiz, auditor, suf


def test_executar_confirmacao_grava_tudo(tmp_path):
    saida = tmp_path / "novo"
    res, rec, agentes, juiz, auditor, suf = _rodar(tmp_path, saida)
    assert rec.chamadas == 3 and juiz.invocacoes == 4 and auditor.invocacoes == 1 and suf.invocacoes == 3   # juiz: 6 − 1 inválido − 1 hit (a1≡a2 em q_0001)
    s = pd.read_parquet(saida / "suficiencia_contexto.parquet")
    assert list(s.columns) == list(COLUNAS_SUFICIENCIA) and s.item_id.tolist() == ["q_0001", "q_0002", "q_0003"]
    assert s.veredicto_suficiencia.tolist() == [2, 1, 0] and s.n_chunks.tolist() == [2, 2, 2] and not s.contexto_vazio.any()
    assert s.judge_model.eq(MODELO_JUIZ).all() and s.cost_usd.sum() == pytest.approx(0.006)
    p = pd.read_parquet(saida / "por_item.parquet")
    assert p.suficiencia.tolist() == [2, 1, 0] and p.z_media.tolist() == [1.0, 0.5, 0.5]
    assert res["adequacao_media"] == pytest.approx(4 / 6) and res["variante"] == "P0" and res["degrau"] == "RX"
    assert res["n_itens"] == 3 and res["n_respostas"] == 6 and res["n_has_doc_ref"] == 2 and res["n_com_dispositivo"] == 1
    sf = res["suficiencia"]
    assert {k: sf[k] for k in ("n_suficiente", "n_parcial", "n_insuficiente", "n_falha_parse")} == {"n_suficiente": 1, "n_parcial": 1, "n_insuficiente": 1, "n_falha_parse": 0}
    assert sf["frac_suficiente"] == pytest.approx(1 / 3) and sf["frac_insuficiente"] == pytest.approx(1 / 3) and len(sf["ic_frac_suficiente"]) == 2
    custo = json.loads((saida / "custo.json").read_text(encoding="utf-8"))
    assert list(custo["etapas"]) == ["contexto", "agente_a1", "agente_a2", "juiz", "auditor", "suficiencia"] and custo["n_itens"] == 3
    e = custo["etapas"]
    assert e["contexto"]["recuperados"] == 3 and e["agente_a1"]["geradas"] == 3 and e["agente_a1"]["retries"] == 3 and e["agente_a1"]["falhas_parse"] == 1   # k_max=3 → n_retries=3 no inválido
    assert e["agente_a2"]["retries"] == 0 and e["juiz"]["chamadas"] == 4 and e["juiz"]["custo_usd"] == pytest.approx(0.012)
    assert e["auditor"]["auditadas"] == 1 and e["auditor"]["custo_usd"] == pytest.approx(0.001)
    assert e["suficiencia"]["julgados"] == 3 and e["suficiencia"]["vazios"] == 0 and e["suficiencia"]["custo_usd"] == pytest.approx(0.006)
    assert all("segundos" in v for v in e.values()) and custo["custo_usd_total"] == pytest.approx(0.019) and custo["ambiente"] == {"device": "cpu"}
    resumo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
    assert "custo" not in resumo and resumo["suficiencia"]["n_suficiente"] == 1 and res["custo"]["custo_usd_total"] == pytest.approx(0.019)
    md = (saida / "resumo.md").read_text(encoding="utf-8")
    assert "| novo | RX |" in md and "| confirmação |" in md and (saida / "itens.parquet").exists()


def test_segunda_execucao_nao_chama_modelo(tmp_path):
    saida = tmp_path / "novo"
    _rodar(tmp_path, saida)
    custo_antes = (saida / "custo.json").read_text(encoding="utf-8")
    res, rec, agentes, juiz, auditor, suf = _rodar(tmp_path, saida)                       # novos objetos, mesmo cache do juiz em disco
    assert rec.chamadas == 0 and all(a.invocacoes == 0 for a in agentes.values()) and juiz.invocacoes == 0 and auditor.invocacoes == 0 and suf.invocacoes == 0
    assert res["delta_vs_golden"] == 0.0 and res["delta_vs_golden_suficiencia"] == 0.0 and res["adequacao_media"] == pytest.approx(4 / 6)
    assert (saida / "custo.json").read_text(encoding="utf-8") == custo_antes and res["custo"]["custo_usd_total"] == 0.0   # medição da 1ª execução intacta


def test_golden_outro_degrau_e_retomada_da_suficiencia(tmp_path):
    saida = tmp_path / "novo"
    _rodar(tmp_path, saida)
    with pytest.raises(GoldenDivergente):
        _rodar(tmp_path, saida, degrau="RY")
    (saida / "suficiencia_contexto.parquet").unlink()                                       # corrida interrompida antes do golden
    (saida / "resumo.json").unlink()
    res, rec, agentes, juiz, auditor, suf = _rodar(tmp_path, saida)
    assert suf.invocacoes == 3 and rec.chamadas == 0 and juiz.invocacoes == 0 and auditor.invocacoes == 0 and all(a.invocacoes == 0 for a in agentes.values())
    assert res["suficiencia"]["n_suficiente"] == 1 and "delta_vs_golden" not in res and (saida / "resumo.json").exists()


def test_itens_confirmacao(cfg):
    from rag2.dados import carregar_itens, itens_com_dispositivo

    itens = carregar_itens(cfg)
    sel = itens_confirmacao(cfg, itens)
    ids = [it.item_id for it in sel]
    assert len(sel) == 200 and ids == sorted(ids) and ids[0] == "q_0004" and ids[-1] == "q_0712"
    assert sum(it.has_doc_ref for it in sel) == 168 and len(set(ids) & {x.item_id for x in itens_com_dispositivo(itens)}) == 151
