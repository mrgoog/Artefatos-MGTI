"""Cliente llama-swap para LLMs locais via API OpenAI-compatível."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ensemble_llm.agentes.contratos import ResultadoInvocacao


class ErroLlamaSwap(RuntimeError):
    """Erro base para falhas do cliente llama-swap."""

    pass


class ErroLlamaSwapTransitorio(ErroLlamaSwap):
    """Erro transitório com retry automático (timeout, conexão recusada, 5xx)."""

    pass


class ErroLlamaSwapPermanente(ErroLlamaSwap):
    """Erro permanente sem retry (4xx, modelo indisponível no host)."""

    pass


def _eh_erro_contexto_excedido(resp: httpx.Response) -> bool:
    """Detecta erro de janela de contexto excedida no backend llama-swap."""
    if resp.status_code != 500:
        return False
    texto = resp.text.lower()
    return "context size has been exceeded" in texto


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


class ClienteLlamaSwap:
    """Cliente HTTP para llama-swap em endpoint OpenAI-compatível."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:8080",
        temperature: float = 0.0,
        max_tokens: int = 2048,
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
        self._chat_url = f"{self.host}/v1/chat/completions"
        self._models_url = f"{self.host}/v1/models"

    def validate_model(self) -> None:
        """Verifica se ``self.model`` está disponível no servidor llama-swap."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(self._models_url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ErroLlamaSwap(
                f"Falha ao consultar {self._models_url} para validar {self.model!r}: {e}"
            ) from e

        available = {m["id"] for m in data.get("data", [])}
        if self.model not in available:
            prefix = self.model.split("-", 1)[0]
            similares = sorted(m for m in available if prefix in m)[:5]
            raise ErroLlamaSwapPermanente(
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
        retry=retry_if_exception_type(ErroLlamaSwapTransitorio),
        reraise=True,
    )
    def _invocar_com_retry(
        self,
        user_prompt: str,
        system_prompt: str | None,
        prompt_hash: str,
    ) -> ResultadoInvocacao:
        """Executa POST para /v1/chat/completions com backoff exponencial."""
        mensagens: list[dict[str, str]] = []
        if system_prompt:
            mensagens.append({"role": "system", "content": system_prompt})
        mensagens.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mensagens,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Disable thinking para todos os modelos.
        payload["enable_thinking"] = False
        
        if self.seed is not None:
            payload["seed"] = self.seed

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self._chat_url, json=payload)
        except httpx.TimeoutException as e:
            raise ErroLlamaSwapTransitorio(f"Timeout: {e}") from e
        except httpx.ConnectError as e:
            raise ErroLlamaSwapTransitorio(f"Erro de conexão com {self.host}: {e}") from e
        except httpx.HTTPError as e:
            raise ErroLlamaSwapTransitorio(f"HTTP error: {e}") from e

        if _eh_erro_contexto_excedido(resp):
            raise ErroLlamaSwapPermanente(
                f"Janela de contexto excedida no modelo {self.model!r}: {resp.text[:300]}"
            )
        if 500 <= resp.status_code < 600:
            raise ErroLlamaSwapTransitorio(f"Server error {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ErroLlamaSwapPermanente(f"Client error {resp.status_code}: {resp.text[:500]}")

        return self._parse_resposta(resp.json(), prompt_hash)

    def _parse_resposta(self, data: dict[str, Any], prompt_hash: str) -> ResultadoInvocacao:
        """Extrai conteúdo e contagens de tokens do JSON de resposta OpenAI-compatível."""
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = str(choice.get("finish_reason") or "stop")
        except (KeyError, IndexError, TypeError) as e:
            raise ErroLlamaSwapPermanente(
                f"Resposta llama-swap sem 'choices[0].message.content': {data}"
            ) from e

        usage = data.get("usage", {})
        n_in = int(usage.get("prompt_tokens") or 0)
        n_out = int(usage.get("completion_tokens") or 0)

        return ResultadoInvocacao(
            raw_output=content,
            prompt_hash=prompt_hash,
            model=self.model,
            cost_usd=0.0,
            n_tokens_input=n_in,
            n_tokens_output=n_out,
            finish_reason=finish,
        )
