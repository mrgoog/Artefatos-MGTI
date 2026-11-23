"""Auditor de recusas: regex PADRAO_RECUSA, prompts e parser importados de scripts/auditar_recusas.py da origem (nunca reescritos)."""

from __future__ import annotations

import importlib.util
import re
import sys
from types import ModuleType

from rag2.config import ConfigRag2

_NOME = "_auditar_recusas_origem"
_MODULO: ModuleType | None = None


def modulo_auditor(cfg: ConfigRag2 | None = None) -> ModuleType:
    """Carrega uma única vez scripts/auditar_recusas.py da origem (só definições; main() não roda)."""
    global _MODULO
    if _MODULO is None:
        caminho = (cfg or ConfigRag2()).origem / "scripts" / "auditar_recusas.py"
        spec = importlib.util.spec_from_file_location(_NOME, caminho)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"auditar_recusas.py não encontrado em {caminho}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_NOME] = mod
        spec.loader.exec_module(mod)
        _MODULO = mod
    return _MODULO


def padrao_recusa() -> re.Pattern[str]:
    """`PADRAO_RECUSA` da origem — a definição de recusa textual (arch §3)."""
    return modulo_auditor().PADRAO_RECUSA


def eh_recusa(texto: str | None) -> bool:
    """True se o texto da resposta casa PADRAO_RECUSA (None/NaN/vazio → False; o parquet relê response_text nulo como NaN)."""
    return isinstance(texto, str) and bool(texto) and padrao_recusa().search(texto) is not None


def prompt_auditor(pergunta: str, contexto: str, ground_truth: str, resposta_agente: str) -> tuple[str, str]:
    """(system, user) do auditor com o template da origem; `contexto` é o texto que o agente viu."""
    m = modulo_auditor()
    usuario = m.TEMPLATE_USUARIO_AUDITOR.format(
        pergunta=pergunta.strip(), contexto=contexto, ground_truth=ground_truth.strip(), resposta_agente=resposta_agente.strip()
    )
    return m.PROMPT_SISTEMA_AUDITOR, usuario


def parse_auditor(bruto: str) -> tuple[str, int]:
    """(raciocinio, veredicto) com o parser tolerante a truncamento da origem; veredicto -1 se irrecuperável."""
    return modulo_auditor()._parse(bruto)
