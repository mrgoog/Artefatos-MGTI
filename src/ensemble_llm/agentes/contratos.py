"""Protocolos e tipos comuns para clientes LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ResultadoInvocacao:
    """Resultado bruto de uma chamada LLM: texto gerado, hash do prompt, modelo e custo."""

    raw_output: str
    prompt_hash: str
    model: str
    cost_usd: float = 0.0
    n_tokens_input: int = 0
    n_tokens_output: int = 0
    finish_reason: str = "stop"


@dataclass(frozen=True)
class InvocacaoEstruturada:
    """Resultado de uma invocação com tentativas de parsing JSON: objeto parseado ou None."""

    parsed: object | None
    raw_output: str
    prompt_hash: str
    model: str
    n_retries: int
    valid_json: bool
    cost_usd: float = 0.0


@runtime_checkable
class ClienteLLM(Protocol):
    """Protocolo mínimo para clientes LLM: invoke + validate_model (OpenRouter ou Ollama)."""

    model: str
    temperature: float

    def invoke(self, prompt: str, *, system_prompt: str | None = None) -> ResultadoInvocacao: ...

    def validate_model(self) -> None: ...
