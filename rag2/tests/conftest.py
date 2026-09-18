"""Fixtures compartilhadas: configuração do ambiente e itens carregados uma vez por sessão."""

import pytest

from rag2.config import ConfigRag2


@pytest.fixture(scope="session")
def cfg() -> ConfigRag2:
    """Configuração lida de .env; os testes de dados exigem o ambiente completo."""
    return ConfigRag2()


NORMAS_SINTETICAS = [
    {"filename": "Lei nº 1.txt", "filedata": "Art. 1º Despesas médicas com lente intraocular são dedutíveis."},
    {"filename": "Lei nº 2.txt", "filedata": "Art. 7º Rendimentos de aluguel entre pai e filho."},
    {"filename": "Lei nº 3.txt", "filedata": "Art. 7º Rendimentos de aluguel entre pai e filho."},   # idêntico à Lei 2: empate exato nos dois ramos
]
CARF_SINTETICO = [
    {"filename": "10000_1.txt", "filedata": "Acórdão sobre lente intraocular e despesa médica."},
    {"filename": "10000_2.txt", "filedata": "Acórdão sobre usufruto de aluguel."},
]


@pytest.fixture
def indice_sintetico(tmp_path):
    """Índice híbrido de 5 chunks (2 CARF, depois 3 normas) construído por construir_indice_reparado com VetorizadorHash(dim=16)."""
    import json

    from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

    from rag2.indice import carregar_indice
    from rag2.indices import construir_indice_reparado

    n, c = tmp_path / "normas.json", tmp_path / "carf.json"
    n.write_text(json.dumps(NORMAS_SINTETICAS, ensure_ascii=False), encoding="utf-8")
    c.write_text(json.dumps(CARF_SINTETICO, ensure_ascii=False), encoding="utf-8")
    construir_indice_reparado(
        ConfigRag2(), saida=tmp_path / "idx", arquivo_normas=n, arquivo_carf=c,
        vetorizador=VetorizadorHash(dim=16), log=lambda s: None,
    )
    return carregar_indice(tmp_path / "idx")
