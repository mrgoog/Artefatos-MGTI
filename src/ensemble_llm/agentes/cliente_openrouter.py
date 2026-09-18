"""Cliente OpenRouter."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ensemble_llm.agentes.contratos import ResultadoInvocacao

logger = logging.getLogger(__name__)


class ErroOpenRouter(RuntimeError):
    """Erro base para falhas do cliente OpenRouter."""

    pass


class ErroOpenRouterTransitorio(ErroOpenRouter):
    """Erro transitório com retry automático (timeout, rate-limit 429, erros 5xx)."""

    pass


class ErroOpenRouterPermanente(ErroOpenRouter):
    """Erro permanente sem retry (4xx, modelo inexistente)."""

    pass


def _hash_prompt(
    system_prompt: str | None,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    """Gera SHA-256 hex do payload de invocação; usado como chave de cache do juiz."""
    payload = json.dumps(
        {
            "system": system_prompt or "",
            "user": user_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ClienteOpenRouter:
    """Cliente HTTP para o gateway OpenRouter com retry exponencial para erros transitórios."""

    def __init__(
        self,
        model: str,
        api_key: str,
        url: str = "https://openrouter.ai/api/v1/chat/completions",
        referer: str = "http://localhost",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 42,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.url = url
        self.referer = referer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._base_url = url.rsplit("/chat/completions", 1)[0]

    def validate_model(self) -> None:
        models_url = f"{self._base_url}/models"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(models_url, headers={"Authorization": f"Bearer {self.api_key}"})
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ErroOpenRouter(f"Falha ao validar modelo {self.model!r}: {e}") from e

        available = {m["id"] for m in data.get("data", [])}
        if self.model not in available:
            similares = sorted(m for m in available if self.model.split("/")[0] in m)[:5]
            raise ErroOpenRouterPermanente(
                f"Modelo {self.model!r} não disponível no OpenRouter. Modelos similares: {similares}"
            )

    def invoke(self, prompt: str, *, system_prompt: str | None = None) -> ResultadoInvocacao:
        """Calcula hash do prompt e delega a chamada HTTP ao método com retry."""
        prompt_hash = _hash_prompt(
            system_prompt,
            prompt,
            self.model,
            self.temperature,
            self.max_tokens,
            self.reasoning_effort,
        )
        return self._invocar_com_retry(prompt, system_prompt, prompt_hash)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(ErroOpenRouterTransitorio),
        reraise=True,
    )
    def _invocar_com_retry(
        self,
        user_prompt: str,
        system_prompt: str | None,
        prompt_hash: str,
    ) -> ResultadoInvocacao:
        """Executa POST para a API OpenRouter com backoff exponencial em erros transitórios."""
        t0 = time.perf_counter()
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
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.seed is not None:
            payload["seed"] = self.seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": "ensemble-llm",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.url, headers=headers, json=payload)
        except httpx.TimeoutException as e:
            raise ErroOpenRouterTransitorio(f"Timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ErroOpenRouterTransitorio(f"HTTP error: {e}") from e

        if resp.status_code == 429:
            raise ErroOpenRouterTransitorio(f"Rate limited: {resp.text[:200]}")
        if 500 <= resp.status_code < 600:
            raise ErroOpenRouterTransitorio(f"Server error {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ErroOpenRouterPermanente(f"Client error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        resultado = self._parse_resposta(data, prompt_hash)
        duracao = time.perf_counter() - t0
        logger.info(
            "OpenRouter model=%s concluído em %.2fs (prompt=%d tok, completion=%d tok, cost=$%.6f, finish=%s)",
            self.model,
            duracao,
            resultado.n_tokens_input,
            resultado.n_tokens_output,
            resultado.cost_usd,
            resultado.finish_reason,
        )
        return resultado

    def _parse_resposta(self, data: dict[str, Any], prompt_hash: str) -> ResultadoInvocacao:
        """Extrai conteúdo, tokens e custo do JSON de resposta da API OpenRouter."""
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason", "stop")
        except (KeyError, IndexError) as e:
            raise ErroOpenRouterPermanente(
                f"Resposta OpenRouter sem 'choices[0].message.content': {data}"
            ) from e

        usage = data.get("usage", {})
        n_in = int(usage.get("prompt_tokens", 0))
        n_out = int(usage.get("completion_tokens", 0))
        cost = float(usage.get("cost", 0.0) or 0.0)

        return ResultadoInvocacao(
            raw_output=content,
            prompt_hash=prompt_hash,
            model=self.model,
            cost_usd=cost,
            n_tokens_input=n_in,
            n_tokens_output=n_out,
            finish_reason=finish,
        )
