"""Variantes de prompt da frente B: P0 é o prompt atual da origem, sem alteração; P1–P3 registram-se em VARIANTES."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ensemble_llm.agentes.prompts_agentes import (
    DICA_SCHEMA_AGENTE,
    construir_prompt_agente,
    formatar_chunks_recuperados,
)
from ensemble_llm.esquemas import BlocoRecuperado, RespostaAgente
from pydantic import BaseModel


@runtime_checkable
class VariantePrompt(Protocol):
    """Uma variante da frente B: monta (system, user) e diz como o contexto foi apresentado ao agente.

    `schema` precisa manter `resposta` e `confianca` obrigatórios (arch §4); campos extras são ignorados pelo juiz.
    """

    nome: str
    schema: type[BaseModel]
    schema_hint: str

    def construir(self, pergunta: str, blocos: list[BlocoRecuperado]) -> tuple[str, str]: ...

    def contexto(self, blocos: list[BlocoRecuperado]) -> str: ...


class VarianteP0:
    """P0 — o par system/user de `prompts_agentes.py`, byte a byte o da produção (mesmo prompt_hash)."""

    nome = "P0"
    schema: type[BaseModel] = RespostaAgente
    schema_hint = DICA_SCHEMA_AGENTE

    def construir(self, pergunta: str, blocos: list[BlocoRecuperado]) -> tuple[str, str]:
        """Delega à origem."""
        return construir_prompt_agente(pergunta, blocos)

    def contexto(self, blocos: list[BlocoRecuperado]) -> str:
        """O contexto exatamente como entrou no prompt — é o que o auditor de recusas deve ver."""
        return formatar_chunks_recuperados(blocos)


VARIANTES: dict[str, type] = {"P0": VarianteP0}
"""Variante → classe; M4.2–M4.4 registram P1–P3 aqui."""
