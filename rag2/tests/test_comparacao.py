"""Comparação pareada: parear/comparar/golden com fakes; cliente que nunca chama o juiz; produção-200 reconstruída por prompt_hash sem chamadas (adequação 0,2267, B0 qwen); oráculo recall novo ≡ R3; CLI em tmp. CPU; precisa do .env e dos artefatos versionados de M5.1."""

import json

import pandas as pd
import pytest
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz
from ensemble_llm.juiz.executor import ExecutorJuiz

from rag2.comparacao import (
    COLUNAS_COMPARACAO,
    METRICAS_PAREADAS,
    ClienteSomenteCache,
    RotuloAusente,
    comparar,
    melhor_agente_fixo,
    parear,
    por_item_producao,
    recall_pareado,
    registrar_comparacao,
)
from rag2.config import RAIZ
from rag2.constantes import AGENTES, MODELO_JUIZ
from rag2.escada import GoldenDivergente

IDS = ["q_0001", "q_0002", "q_0003"]
REF = [True, True, False]


def _por_item(z: dict, suficiencia: list, palavras: list, recusa: dict | None = None, evasiva: dict | None = None, parse: dict | None = None) -> pd.DataFrame:
    """Frame com as 21 colunas de por_item: z/recusa/evasiva/parse por agente (listas por item) e médias."""
    recusa = recusa or {a: [0, 0, 0] for a in AGENTES}
    evasiva = evasiva or {a: [0, 0, 0] for a in AGENTES}
    parse = parse or {a: [1, 1, 1] for a in AGENTES}
    d = {"item_id": IDS, "has_doc_ref": REF, "n_chunks": [7, 7, 7], "palavras": palavras}
    for a in AGENTES:
        d[f"z_{a}"], d[f"recusa_{a}"], d[f"evasiva_{a}"], d[f"parse_{a}"] = z[a], recusa[a], evasiva[a], parse[a]
    df = pd.DataFrame(d)
    for nome in ("z", "recusa", "evasiva", "parse"):
        df[f"{nome}_media"] = sum(df[f"{nome}_{a}"] for a in AGENTES) / len(AGENTES)
    df["suficiencia"] = suficiencia
    return df


ATUAL = _por_item({"qwen35_9b": [1, 0, 0], "granite41_8b": [0, 0, 0], "gemma4_e4b": [1, 1, 0]}, [2, 1, 0], [100, 200, 300], recusa={"qwen35_9b": [0, 1, 0], "granite41_8b": [0, 0, 0], "gemma4_e4b": [0, 0, 0]}, evasiva={"qwen35_9b": [0, 1, 0], "granite41_8b": [0, 0, 0], "gemma4_e4b": [0, 0, 0]})
NOVO = _por_item({"qwen35_9b": [1, 1, 0], "granite41_8b": [1, 0, 0], "gemma4_e4b": [1, 1, 1]}, [2, 2, 1], [150, 250, 350], parse={"qwen35_9b": [1, 1, 1], "granite41_8b": [1, 1, 0], "gemma4_e4b": [1, 1, 1]})
RECALL = pd.DataFrame({"item_id": ["q_0001"], "recall_passagem_atual": [0.0], "recall_passagem_novo": [1.0], "recall_arquivo_atual": [0.5], "recall_arquivo_novo": [1.0], "n_esperados": [2]})
CUSTO = {"atual": {"agentes": {"chamadas": 9, "custo_usd": 0.0}, "juiz": {"chamadas": 8, "custo_usd": 0.01}, "auditor": {"chamadas": 1, "custo_usd": 0.001}, "suficiencia": {"chamadas": 3, "custo_usd": 0.02}, "segundos": None, "custo_usd_total": 0.031},
         "novo": {"agentes": {"chamadas": 9, "custo_usd": 0.0, "retries": 3}, "juiz": {"chamadas": 9, "custo_usd": 0.012}, "auditor": {"chamadas": 0, "custo_usd": 0.0}, "suficiencia": {"chamadas": 3, "custo_usd": 0.021}, "segundos": {"contexto": 1.0, "agente_qwen35_9b": 2.0, "agente_granite41_8b": 3.0, "agente_gemma4_e4b": 4.0, "juiz": 5.0, "auditor": 0.5, "suficiencia": 6.0}, "custo_usd_total": 0.033}}


def test_cliente_somente_cache_nunca_chama(tmp_path):
    cli = ClienteSomenteCache()
    ExecutorJuiz(client=cli, cache=CacheRotulosJuiz(tmp_path / "judge", MODELO_JUIZ))   # type: ignore[arg-type]  # model bate com o cache
    assert (cli.model, cli.temperature, cli.max_tokens) == (MODELO_JUIZ, 0.0, 1024)   # o prompt_hash da produção
    with pytest.raises(RotuloAusente):
        cli.invoke("pergunta", system_prompt="sistema")


def test_parear_e_comparar_fakes():
    b0 = melhor_agente_fixo(ATUAL)
    assert b0 == "gemma4_e4b"                                                            # 2/3 > qwen 1/3 > granite 0
    par = parear(ATUAL, NOVO, RECALL, b0)
    assert par.item_id.tolist() == IDS and par.com_dispositivo.tolist() == [True, False, False]
    assert par.z_b0_atual.tolist() == [1.0, 1.0, 0.0] and par.z_b0_novo.tolist() == [1.0, 1.0, 1.0]
    assert par.recall_passagem_novo.tolist()[0] == 1.0 and par.recall_passagem_novo.isna().tolist() == [False, True, True]
    assert par.falha_parse_novo.tolist() == pytest.approx([0.0, 0.0, 1 / 3]) and par.suficiente_atual.tolist() == [1.0, 0.0, 0.0] and par.insuficiente_atual.tolist() == [0.0, 0.0, 1.0]
    comp = comparar(par, b=200)
    assert list(comp.columns) == list(COLUNAS_COMPARACAO) and comp.metrica.tolist() == list(METRICAS_PAREADAS)
    c = comp.set_index("metrica")
    assert c.loc["recall_passagem", "n"] == 1 and c.loc["z_media", "n"] == 3 and c.loc["recall_passagem", "delta"] == 1.0
    assert c.loc["z_media", "atual"] == pytest.approx(3 / 9) and c.loc["z_media", "novo"] == pytest.approx(6 / 9) and c.loc["z_media", "delta"] == pytest.approx(3 / 9)
    assert c.loc["evasiva", "atual"] == pytest.approx(1 / 9) and c.loc["palavras", "delta"] == 50.0 and (c.ic_lo <= c.delta + 1e-12).all() and (c.ic_hi >= c.delta - 1e-12).all()
    with pytest.raises(ValueError):
        parear(ATUAL, NOVO.iloc[::-1].reset_index(drop=True), RECALL, b0)                # ordem diferente


def test_melhor_agente_fixo_empate():
    empate = _por_item({"qwen35_9b": [1, 1, 0], "granite41_8b": [1, 1, 0], "gemma4_e4b": [0, 0, 0]}, [1, 1, 1], [1, 1, 1])
    assert melhor_agente_fixo(empate) == "qwen35_9b"   # qwen e granite empatam (2/3); a ordem de AGENTES (qwen 1º) desempata


def test_registrar_golden_e_segunda_execucao(tmp_path):
    res = registrar_comparacao(ATUAL, NOVO, RECALL, CUSTO, tmp_path, nome_novo="PX/RX", nome_atual="prod", b=200)
    for f in ("pareado.parquet", "comparacao.parquet", "comparacao.json", "comparacao.md", "custo.md"):
        assert (tmp_path / f).exists()
    assert res["b0"] == "gemma4_e4b" and res["n_com_dispositivo"] == 1 and res["revisao_b3"]["significativamente_melhor"] is True and "delta_vs_golden" not in res
    j = json.loads((tmp_path / "comparacao.json").read_text(encoding="utf-8"))
    assert [m["metrica"] for m in j["metricas"]] == list(METRICAS_PAREADAS) and list(j["custo"]["novo"]["segundos"]) == list(CUSTO["novo"]["segundos"])   # sem sort_keys
    md = (tmp_path / "comparacao.md").read_text(encoding="utf-8")
    assert "| z_media | 3 |" in md and "significativamente melhor" in md and "| agentes | 9 |" in (tmp_path / "custo.md").read_text(encoding="utf-8")
    antes = (tmp_path / "comparacao.json").read_bytes()
    res2 = registrar_comparacao(ATUAL, NOVO, RECALL, CUSTO, tmp_path, nome_novo="PX/RX", nome_atual="prod", b=200)
    assert res2["delta_vs_golden"] == {"z_media": 0.0, "recall_passagem": 0.0} and (tmp_path / "comparacao.json").read_bytes() == antes
    j["metricas"][2]["delta"] += 0.01                                                    # z_media adulterado
    (tmp_path / "comparacao.json").write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(GoldenDivergente):
        registrar_comparacao(ATUAL, NOVO, RECALL, CUSTO, tmp_path, nome_novo="PX/RX", nome_atual="prod", b=200)
    res3 = registrar_comparacao(ATUAL, NOVO, RECALL, CUSTO, tmp_path, nome_novo="PX/RX", nome_atual="prod", b=200, sobrescrever=True)
    assert "delta_vs_golden" not in res3 and json.loads((tmp_path / "comparacao.json").read_text(encoding="utf-8"))["metricas"][2]["delta"] == pytest.approx(3 / 9)


def test_por_item_producao_sem_chamar_o_juiz(cfg, tmp_path):
    from rag2.confirmacao import itens_confirmacao
    from rag2.dados import carregar_itens

    itens = itens_confirmacao(cfg, carregar_itens(cfg))
    atual = por_item_producao(cfg, itens, tmp_path / "rotulos.parquet", log=lambda s: None)     # um miss levantaria RotuloAusente
    novo = pd.read_parquet(RAIZ / "runs" / "confirmacao" / "novo" / "por_item.parquet")
    assert list(atual.columns) == list(novo.columns) and atual.item_id.tolist() == novo.item_id.tolist()
    rot = pd.read_parquet(tmp_path / "rotulos.parquet")
    assert len(rot) == 600 and int((rot.prompt_hash_juiz == "").sum()) == 5                # 5 JSON inválidos (granite) → z=0 sem juiz
    assert atual.z_media.mean() == pytest.approx(0.2267, abs=1e-4) and melhor_agente_fixo(atual) == "qwen35_9b"
    assert int(sum(atual[f"recusa_{a}"].sum() for a in AGENTES)) == 37 and int(sum(atual[f"evasiva_{a}"].sum() for a in AGENTES)) == 8
    assert int((atual.suficiencia == 2).sum()) == 50 and int((atual.suficiencia == 0).sum()) == 21
    assert atual.n_chunks.mean() == pytest.approx(7.12) and atual.palavras.mean() == pytest.approx(3264.68, abs=0.01)


def test_recall_novo_igual_r3(cfg):
    from rag2.confirmacao import itens_confirmacao
    from rag2.dados import carregar_itens

    itens = itens_confirmacao(cfg, carregar_itens(cfg))
    ctx = pd.read_parquet(RAIZ / "runs" / "confirmacao" / "novo" / "contexto.parquet")
    r = recall_pareado(cfg, itens, ctx)
    assert len(r) == 151 and int(r.n_esperados.sum()) == 459 and r.item_id.is_monotonic_increasing
    r3 = pd.read_parquet(RAIZ / "runs" / "escada" / "R3" / "por_item.parquet").set_index("item_id").loc[r.item_id]
    assert abs(r.recall_passagem_novo.to_numpy() - r3.recall_passagem.to_numpy()).max() == 0.0      # oráculo: Casador só dos servidos ≡ R3
    assert abs(r.recall_arquivo_novo.to_numpy() - r3.recall_arquivo.to_numpy()).max() == 0.0
    assert r.recall_passagem_atual.mean() == pytest.approx(0.2137, abs=1e-4) and r.recall_passagem_novo.mean() == pytest.approx(0.3308, abs=1e-4)


def test_cli_comparar_em_tmp(cfg, capsys, tmp_path):
    from rag2.cli import main

    assert main(["confirmacao", "comparar", "--b", "200", "--saida", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "comparacao: 200 itens (has_doc_ref=168, com dispositivo=151) | atual=phase2_local_full_715 novo=P0/R3 | b0=qwen35_9b" in out
    assert "| recall_passagem | 151 | 0.2137 | 0.3308 | +0.1171 [" in out and "| z_media | 200 | 0.2267 | 0.2300 | +0.0033 [" in out
    assert "custo: atual US$ 2.11 (juiz 595 rótulos/US$ 0.89" in out and "| novo US$ 2.15 (juiz 582/US$ 0.85" in out and "IC inclui 0" in out
    for f in ("pareado.parquet", "comparacao.parquet", "comparacao.json", "comparacao.md", "custo.md", "atual/rotulos.parquet"):
        assert (tmp_path / f).exists()
    assert len(pd.read_parquet(tmp_path / "pareado.parquet")) == 200 and len(pd.read_parquet(tmp_path / "atual" / "rotulos.parquet")) == 600   # --saida: tudo em tmp, o golden de runs/confirmacao/ não é tocado
