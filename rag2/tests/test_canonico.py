"""As funções canônicas importadas da origem se comportam como o canônico documenta."""

from rag2.canonico import dispositivos_esperados, normalizar_artigo, regex_artigo


def test_normalizar_artigo():
    assert normalizar_artigo("1º") == "1"
    assert normalizar_artigo("12-A") == "12-A"
    assert normalizar_artigo(" 25 ") == "25"


def test_regex_artigo_casa_ordinal_e_nao_casa_prefixo():
    assert regex_artigo("25").search("... Art. 25 - Texto ...")
    assert regex_artigo("25").search("Art. 250 ...") is None
    assert regex_artigo("5").search("Art. 5º Fica isento")
    assert regex_artigo("12-A").search("Artigo 12-A. Dispõe")


def test_dispositivos_esperados_extrai_pares_e_ignora_texto_livre():
    bruto = [
        '{"Lei nº 9.250": [{"título": "Lei nº 9.250", "artigos": [{"artigo": "25"}, {"artigo": "1º"}], '
        '"file": "Lei nº 9.250.txt"}]}',
        "Acórdão CARF nº 2202-009.000 — texto livre, não é JSON",
    ]
    assert dispositivos_esperados(bruto) == {("Lei nº 9.250.txt", "25"), ("Lei nº 9.250.txt", "1")}
