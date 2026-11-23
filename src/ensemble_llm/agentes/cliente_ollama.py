"""Cliente Ollama para LLMs locais."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ensemble_llm.agentes.contratos import ResultadoInvocacao


class ErroOllama(RuntimeError):
    """Erro base para falhas do cliente Ollama."""

    pass


class ErroOllamaTransitorio(ErroOllama):
    """Erro transitório com retry automático (timeout, conexão recusada, 5xx)."""

    pass


class ErroOllamaPermanente(ErroOllama):
    """Erro permanente sem retry (4xx, modelo indisponível no servidor Ollama)."""

    pass


def _hash_prompt(
    system_prompt: str | None,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Gera SHA-256 hex do payload de invocação; garante determinismo do cache."""
    payload = json.dumps(
        {
            "system": system_prompt or "",
            "user": user_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ClienteOllama:
    """
    Cliente para LLMs locais via Ollama.

    Implementa o mesmo protocolo ClienteLLM do ClienteOpenRouter.
    O campo ``host`` corresponde ao valor de ``ConfigAgente.host`` no YAML;
    se omitido no YAML, o orquestrador usa o fallback de ``Segredos.ollama_host_gpu0``.

    Diferenças em relação ao OpenRouter:
    - Sem API key.
    - Endpoint de chat: ``POST {host}/api/chat``.
    - Listagem de modelos: ``GET {host}/api/tags``.
    - ``cost_usd`` sempre 0.0 (modelo local).
    - ``ConnectError`` é transitório (servidor Ollama pode estar iniciando).
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 42,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._chat_url = f"{self.host}/api/chat"
        self._tags_url = f"{self.host}/api/tags"

    def validate_model(self) -> None:
        """Verifica se ``self.model`` está disponível no servidor Ollama."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(self._tags_url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ErroOllama(
                f"Falha ao consultar {self._tags_url} para validar {self.model!r}: {e}"
            ) from e

        available = {m["name"] for m in data.get("models", [])}
        if self.model not in available:
            similares = sorted(m for m in available if self.model.split(":")[0] in m)[:5]
            raise ErroOllamaPermanente(
                f"Modelo {self.model!r} não disponível em {self.host}. "
                f"Modelos similares: {similares or sorted(available)[:5]}"
            )

    def invoke(self, prompt: str, *, system_prompt: str | None = None) -> ResultadoInvocacao:
        """Calcula hash do prompt e delega a chamada HTTP ao método com retry."""
        prompt_hash = _hash_prompt(
            system_prompt, prompt, self.model, self.temperature, self.max_tokens
        )
        return self._invocar_com_retry(prompt, system_prompt, prompt_hash)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(ErroOllamaTransitorio),
        reraise=True,
    )
    def _invocar_com_retry(
        self,
        user_prompt: str,
        system_prompt: str | None,
        prompt_hash: str,
    ) -> ResultadoInvocacao:
        """Executa POST para o endpoint /api/chat do Ollama com backoff exponencial."""
        mensagens: list[dict[str, str]] = []
        if system_prompt:
            mensagens.append({"role": "system", "content": system_prompt})
        mensagens.append({"role": "user", "content": user_prompt})

        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if self.seed is not None:
            options["seed"] = self.seed

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mensagens,
            "stream": False,
            "think": False,
            "options": options,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self._chat_url, json=payload)
        except httpx.TimeoutException as e:
            raise ErroOllamaTransitorio(f"Timeout: {e}") from e
        except httpx.ConnectError as e:
            raise ErroOllamaTransitorio(f"Erro de conexão com {self.host}: {e}") from e
        except httpx.HTTPError as e:
            raise ErroOllamaTransitorio(f"HTTP error: {e}") from e

        if 500 <= resp.status_code < 600:
            raise ErroOllamaTransitorio(f"Server error {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ErroOllamaPermanente(f"Client error {resp.status_code}: {resp.text[:500]}")

        return self._parse_resposta(resp.json(), prompt_hash)

    def _parse_resposta(self, data: dict[str, Any], prompt_hash: str) -> ResultadoInvocacao:
        """Extrai conteúdo e contagens de tokens do JSON de resposta do Ollama."""
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise ErroOllamaPermanente(
                f"Resposta Ollama sem 'message.content': {data}"
            ) from e

        done_reason = str(data.get("done_reason") or "stop")
        n_in = int(data.get("prompt_eval_count") or 0)
        n_out = int(data.get("eval_count") or 0)

        return ResultadoInvocacao(
            raw_output=content,
            prompt_hash=prompt_hash,
            model=self.model,
            cost_usd=0.0,
            n_tokens_input=n_in,
            n_tokens_output=n_out,
            finish_reason=done_reason,
        )
