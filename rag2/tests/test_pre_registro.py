"""docs/pre-registro.md e src/rag2/constantes.py declaram os mesmos hiperparâmetros."""

import re

from rag2 import constantes
from rag2.config import RAIZ

PADRAO_LINHA = re.compile(r"^\|\s*([A-Z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*$")


def _tabela_pre_registro() -> dict[str, str]:
    """Só a tabela da seção 4 (a da escada, na seção 3, também tem nomes em maiúsculas)."""
    texto = (RAIZ / "docs" / "pre-registro.md").read_text(encoding="utf-8")
    texto = texto.split("## 4. Hiperparâmetros")[1].split("## 5. Métricas")[0]
    return {m.group(1): m.group(2) for linha in texto.splitlines() if (m := PADRAO_LINHA.match(linha))}


def test_constantes_batem_com_o_pre_registro():
    tabela = _tabela_pre_registro()
    assert len(tabela) >= 22
    for nome, valor in tabela.items():
        assert hasattr(constantes, nome), f"{nome} não existe em constantes.py"
        assert str(getattr(constantes, nome)) == valor, f"{nome}: pre-registro={valor} constantes={getattr(constantes, nome)}"


def test_pre_registro_declara_secoes_obrigatorias():
    texto = (RAIZ / "docs" / "pre-registro.md").read_text(encoding="utf-8")
    for secao in ("## 3. Escada", "## 4. Hiperparâmetros", "## 5. Métricas", "## 6. Critérios de decisão", "## 7. Vazamento"):
        assert secao in texto
    assert "linked_questions" in texto and "dupla" in texto
