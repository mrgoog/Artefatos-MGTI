"""Definições canônicas de dispositivo esperado e casamento artigo↔chunk, importadas da origem."""

from __future__ import annotations

import importlib.util
import re
import sys
from types import ModuleType

from rag2.config import ConfigRag2

_NOME = "_recall_passagem_origem"
_MODULO: ModuleType | None = None


def modulo_canonico(cfg: ConfigRag2 | None = None) -> ModuleType:
    """Carrega uma única vez scripts/recall_passagem.py do repositório de origem como módulo."""
    global _MODULO
    if _MODULO is None:
        caminho = (cfg or ConfigRag2()).script_recall_passagem()
        spec = importlib.util.spec_from_file_location(_NOME, caminho)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"recall_passagem.py não encontrado em {caminho}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_NOME] = mod
        spec.loader.exec_module(mod)
        _MODULO = mod
    return _MODULO


def dispositivos_esperados(bruto: list[str]) -> set[tuple[str, str]]:
    """Pares (arquivo, artigo normalizado) anotados pelo curador — definição do canônico."""
    return modulo_canonico().dispositivos_esperados(bruto)


def regex_artigo(artigo: str) -> re.Pattern[str]:
    """Regex que casa o cabeçalho 'Art. N' no texto de um chunk — definição do canônico."""
    return modulo_canonico()._regex_artigo(artigo)


def normalizar_artigo(artigo: str) -> str:
    """'1º' → '1', '12-A' → '12-A', ' 25 ' → '25' — definição do canônico."""
    return modulo_canonico()._normalizar_artigo(artigo)
