"""Números canônicos: formatação pt-BR, regex de cifras, ledger determinístico ≡ versionado, oráculos dos artefatos, vereditos H1–H5, gate de lastro e tabelas verbatim no apêndice, CLI. CPU; lê só artefatos versionados."""

import pytest

from rag2.numeros import (
    ARQUIVO_NUMEROS,
    ARQUIVO_RELATORIO,
    NOMES_TABELAS,
    carregar,
    cifras,
    conferir,
    fmt,
    fmt_ic,
    fmt_int,
    fmt_sinal,
    ledger,
    tabelas_relatorio,
    veredito,
)


def test_formatacao_pt_br():
    assert fmt(0.23549) == "0,2355" and fmt(3264.68, 1) == "3.264,7" and fmt(7.0, 2) == "7,00" and fmt(0.001, 3) == "0,001"
    assert fmt_int(3199.86) == "3.200" and fmt_int(82) == "82" and fmt_int(10000) == "10.000"
    assert fmt_sinal(0.1171) == "+0,1171" and fmt_sinal(-0.01) == "−0,0100" and fmt_sinal(0.0) == "+0,0000"
    assert fmt_ic(0.2072, 0.2658) == "[0,2072; 0,2658]" and fmt_ic(-0.035, 0.0417, sinal=True) == "[−0,0350; +0,0417]"


def test_regex_de_cifras():
    texto = "R0 0,2355 [0,2072; 0,2658]; 0,236 (3 casas, ignorada); 3.264,7 (1 casa, ignorada); +0,1171 e −0,0100; repetida 0,2355"
    assert cifras(texto) == ["0,2355", "0,2072", "0,2658", "0,1171", "0,0100"]


def test_veredito_criterio_6():
    r0 = {"recall_passagem": 0.2355}
    assert veredito({"ic_recall_passagem": [0.2356, 0.30]}, r0) == "confirmada"
    assert veredito({"ic_recall_passagem": [0.2355, 0.30]}, r0) == "não confirmada"     # igual não exclui


def test_ledger_deterministico_e_igual_ao_versionado(cfg):
    a = carregar()
    texto = ledger(a)
    assert texto == ledger(a) and texto == ARQUIVO_NUMEROS.read_text(encoding="utf-8")   # registrar uma vez: regenerar ≡ versionado (sem timestamp de geração)


def test_oraculos_do_ledger_e_tabelas(cfg):
    a = carregar()
    L = ledger(a)
    for x in ("0,2351", "0,2355 [0,2072; 0,2658]", "0,3426 [0,3106; 0,3754]", "0,2300 [0,1883; 0,2733]", "+0,1171 [+0,0712; +0,1661]", "0,9145", "0,9615", "| 3.200", "5.009", "| 82 |", "qwen35_9b", "0,2333"):
        assert x in L, x
    t = tabelas_relatorio(a)
    assert list(t) == list(NOMES_TABELAS) and all(tab in L for tab in t.values())            # §10 embute as cinco
    assert t["escada"].count("\n") == 7 and t["comparacao"].count("\n") == 15 and t["custo"].count("\n") == 7
    vered = [linha.rsplit("| ", 1)[1].rstrip(" |") for linha in t["hipoteses"].split("\n")[2:]]
    assert vered == ["não confirmada", "confirmada", "confirmada", "não testada", "não confirmada"]
    assert "| P0 | R3 | 200 | 0,2300 [0,1883; 0,2733] | 0,2300 | 0,2300 | 0,2300 |" in t["geracao"]
    assert "| palavras/item | 200 | 3.264,6800 | 3.583,8700 | +319,1900 [+149,9739; +492,6401] |" in t["comparacao"]


# O rascunho docs/relatorio_apendice.md não é publicado: a fonte do apêndice é a dissertação (Apêndice B).
@pytest.mark.skipif(not ARQUIVO_RELATORIO.exists(), reason="docs/relatorio_apendice.md não publicado")
def test_apendice_tem_lastro_e_tabelas_verbatim(cfg):
    a = carregar()
    R = ARQUIVO_RELATORIO.read_text(encoding="utf-8")
    r = conferir(R, ledger(a), tabelas_relatorio(a))
    assert r["sem_lastro"] == [] and r["tabelas_ausentes"] == [] and r["cifras"] >= 40      # o gate de arch §5
    assert R.count("Fonte: elaborado pelo autor.") == 8 and "0,2333" not in R                # A1–A8 (A7/A8: braço de API); o smoke não é resultado
    assert "já é defensável recomendar" in R and "Ainda não é defensável recomendar" in R and "não altera a análise confirmatória do corpo da dissertação" in R


def test_cli_numeros(cfg, capsys, tmp_path):
    from rag2.cli import main

    assert main(["numeros", "--saida", str(tmp_path / "n.md")]) == 0
    assert (tmp_path / "n.md").read_text(encoding="utf-8") == ARQUIVO_NUMEROS.read_text(encoding="utf-8")
    assert "numeros: ledger com" in capsys.readouterr().out
    assert main(["numeros", "--tabelas"]) == 0 and "### escada" in capsys.readouterr().out
    falso = tmp_path / "rel.md"
    falso.write_text("cifra inventada 0,9999, a real 0,2355 e colada de rag2 tabela 0.2355\n", encoding="utf-8")   # sem as tabelas; ponto decimal proibido
    assert main(["numeros", "--conferir", str(falso)]) == 1
    out = capsys.readouterr().out
    assert "1 sem lastro, 1 com ponto decimal; 0/5 tabelas" in out and "sem lastro: 0,9999" in out and "ponto decimal (use vírgula): 0.2355" in out
    if ARQUIVO_RELATORIO.exists():
        assert main(["numeros", "--conferir", str(ARQUIVO_RELATORIO)]) == 0 and "0 sem lastro, 0 com ponto decimal; 5/5 tabelas" in capsys.readouterr().out
