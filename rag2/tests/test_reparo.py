"""Reparo do corpus: extração de HTML sintético, regras de status, JSON determinístico e CLI offline sem rede."""

import hashlib
import json

from rag2 import canonico
from rag2.cli import main
from rag2.dados import carregar_normas
from rag2.reparo import (
    ALVOS,
    CAMPOS_PROIBIDOS,
    FALHA,
    FONTES,
    REPARADO,
    SEM_FONTE,
    SEM_GANHO,
    Fonte,
    avaliar_reparo,
    compartilha_trecho,
    escrever_json,
    extrair_texto,
    reparar_registros,
)

HTML = (
    "<html><head><title>L0001</title><style>p{}</style></head>\n"
    "<body><script>var x=1;</script><p>Presid&ecirc;ncia da Rep&uacute;blica</p>\n"
    "<p>Art. 1<sup>&ordm;</sup>&nbsp;&nbsp;Texto   do&#160;artigo.</p><p>Art. 2º Fim.</p></body></html>"
)


def _fonte() -> Fonte:
    return Fonte(url="https://exemplo.gov.br/x.htm")


def test_extrair_texto_remove_head_script_style_e_decodifica():
    t = extrair_texto(HTML)
    assert t == "Presidência da República Art. 1 º Texto do artigo. Art. 2º Fim."
    assert "L0001" not in t and "var x" not in t and "p{}" not in t
    assert canonico.regex_artigo("1").search(t) and canonico.regex_artigo("2").search(t)


def test_compartilha_trecho():
    original = " ".join(f"p{i}" for i in range(40))
    assert compartilha_trecho(original, "x y " + " ".join(f"p{i}" for i in range(10, 30)) + " z")
    assert not compartilha_trecho(original, " ".join(f"p{i}" for i in range(0, 40, 2)))
    assert compartilha_trecho("a b c", "z a b c z")  # original menor que a janela: exige o original inteiro
    assert not compartilha_trecho("", "a b c")


def test_avaliar_reparo_status():
    original = "Presidência da República LEI Nº 1 Art. 1º Início " + " ".join(f"w{i}" for i in range(20))
    completo = original + " Art. 2º Meio Art. 25 Fim"
    r = avaliar_reparo("Lei nº 1.txt", original, completo, ["1", "2", "2", "25"], _fonte())
    assert (r.status, r.trecho_comum, r.palavras_antes, r.palavras_depois) == (REPARADO, True, 29, 35)
    assert (r.artigos_antes, r.artigos_depois, r.artigos_total, r.pares_ganhos) == (1, 3, 3, 3)
    assert avaliar_reparo("Lei nº 1.txt", original, original, ["1"], _fonte()).status == SEM_GANHO
    assert avaliar_reparo("Lei nº 1.txt", original, "outro documento " * 30, ["1"], _fonte()).status == FALHA
    assert avaliar_reparo("Lei nº 1.txt", original, completo + chr(0xFFFD), ["1"], _fonte()).status == FALHA
    assert avaliar_reparo("Lei nº 1.txt", original, None, ["1"], _fonte()).status == FALHA
    r = avaliar_reparo("IN.txt", original, None, ["1"], Fonte(url=None, nota="sem HTML"))
    assert (r.status, r.fonte, r.nota, r.palavras_depois) == (SEM_FONTE, "", "sem HTML", 29)


def test_reparar_registros_preserva_ordem_filenames_e_campos():
    registros = [
        {"filename": "Lei nº 1.txt", "filedata": "Presidência da República Art. 1º " + " ".join(f"w{i}" for i in range(20))},
        {"filename": "Outro.txt", "filedata": "sem mudança"},
        {"filename": "Lei nº 2.txt", "filedata": "Presidência da República Art. 1º " + " ".join(f"v{i}" for i in range(20))},
    ]
    novo = registros[0]["filedata"] + " Art. 2º Texto novo"
    fontes = {"Lei nº 1.txt": _fonte(), "Lei nº 2.txt": _fonte()}
    esperados = {"Lei nº 1.txt": ["1", "2", "2"], "Lei nº 2.txt": ["3"]}
    saida, resultados = reparar_registros(
        registros, {"Lei nº 1.txt": novo, "Lei nº 2.txt": None}, ("Lei nº 1.txt", "Lei nº 2.txt"), fontes, esperados
    )
    assert [r["filename"] for r in saida] == [r["filename"] for r in registros]
    assert saida[0]["filedata"] == novo and saida[0]["reparado"] is True
    assert (saida[0]["palavras_antes"], saida[0]["palavras_depois"], saida[0]["fonte"]) == (25, 29, _fonte().url)
    assert saida[1] == {"filename": "Outro.txt", "filedata": "sem mudança", "reparado": False,
                        "palavras_antes": 2, "palavras_depois": 2, "fonte": ""}
    assert saida[2]["reparado"] is False and saida[2]["filedata"] == registros[2]["filedata"]
    assert registros[0]["filedata"] != novo and "reparado" not in registros[0]  # entrada não é mutada
    assert [r.status for r in resultados] == [REPARADO, FALHA]
    assert resultados[0].pares_ganhos == 2  # "2" esperado duas vezes passa a casar; "1" já casava
    assert not (set(saida[0]) & CAMPOS_PROIBIDOS)


def test_escrever_json_deterministico(tmp_path):
    registros = [{"filename": "Lei nº 1.txt", "filedata": "Presidência ção", "reparado": False,
                  "palavras_antes": 2, "palavras_depois": 2, "fonte": ""}]
    h1 = escrever_json(registros, tmp_path / "a.json")
    h2 = escrever_json(registros, tmp_path / "b.json")
    assert h1 == h2 == hashlib.sha256((tmp_path / "a.json").read_bytes()).hexdigest()
    bruto = (tmp_path / "a.json").read_text(encoding="utf-8")
    assert "Presidência ção" in bruto and (chr(92) + "u") not in bruto and bruto.endswith("\n")
    assert json.loads(bruto) == registros


def test_alvos_e_fontes_declarados():
    assert len(ALVOS) == 11 and len(set(ALVOS)) == 11 and set(FONTES) == set(ALVOS)
    assert sum(f.url is None for f in FONTES.values()) == 2
    assert all(f.url is None or f.url.startswith("https://www.planalto.gov.br/ccivil_03/") for f in FONTES.values())
    assert all(f.url is not None or f.nota for f in FONTES.values())


def test_cli_offline_so_um_alvo(tmp_path, capsys, cfg):
    alvo = "Medida Provisória nº 252.txt"
    original = next(d["filedata"] for d in carregar_normas(cfg.arquivo_normas()) if d["filename"] == alvo)
    (tmp_path / "reparo").mkdir()
    (tmp_path / "reparo" / alvo).write_text(
        original + " Art. 37 Texto sintético do artigo trinta e sete.", encoding="utf-8"
    )
    assert main(["reparar-corpus", "--offline", "--so", alvo,
                 "--dir-reparo", str(tmp_path / "reparo"), "--saida", str(tmp_path / "reparado")]) == 0
    out = capsys.readouterr().out
    assert f"REPARADO  {alvo} | 27 → 36 palavras | trecho=sim | artigos 0/1 → 1/1" in out
    assert "reparados=1 sem_ganho=0 sem_fonte=0 falhas=0 | pares esperados que passam a casar: 1 | registros=478" in out
    dados = json.loads(
        (tmp_path / "reparado" / "referred_legal_documents_QA_2024_v1.1_reparado.json").read_text(encoding="utf-8")
    )
    assert len(dados) == 478 and [d["reparado"] for d in dados].count(True) == 1
    assert [d["filename"] for d in dados] == [d["filename"] for d in carregar_normas(cfg.arquivo_normas())]
    assert (tmp_path / "reparado" / "relatorio_reparo.md").read_text(encoding="utf-8").count("| REPARADO |") == 1
