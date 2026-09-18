"""
Validação prévia da configuração do experimento.

Roda checks rápidos e gratuitos ANTES de qualquer chamada paga, para falhar
cedo se algum slug de modelo não existe no backend configurado.

- OpenRouter: GET /models (uma chamada, cobre todos os agentes OR + juiz)
- Ollama: GET /api/tags por host único (uma chamada por host distinto)
- llama-swap: GET /v1/models por host único (uma chamada por host distinto)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import httpx

from ensemble_llm.configuracao import ConfigExperimento, Segredos

logger = logging.getLogger(__name__)


class ErroValidacao(RuntimeError):
    """Falha de validação prévia: slug inexistente no backend ou backend inacessível."""


def _validar_openrouter(
    models: list[tuple[str, str]],  # [(role, slug), ...]
    api_key: str,
    base_url: str,
) -> list[str]:
    """Verifica slugs OpenRouter via GET /models; retorna lista de problemas (vazia = OK)."""
    models_url = base_url.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(models_url, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return [f"Falha ao consultar OpenRouter /models: {e}"]

    available = {m["id"] for m in data.get("data", [])}
    logger.info("OpenRouter disponibiliza %d modelos.", len(available))

    problems: list[str] = []
    for role, slug in models:
        if slug not in available:
            similar = sorted(m for m in available if slug.split("/")[0] in m)[:5]
            problems.append(f"{role}: modelo {slug!r} NÃO disponível. Similares: {similar}")
        else:
            logger.info("✓ %s: %s disponível", role, slug)
    return problems


def _validar_ollama_host(
    host: str,
    models: list[tuple[str, str]],  # [(role, slug), ...]
) -> list[str]:
    """Verifica slugs Ollama via GET /api/tags em um host; retorna lista de problemas."""
    tags_url = host.rstrip("/") + "/api/tags"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(tags_url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return [f"Falha ao consultar {tags_url}: {e}"]
    except httpx.ConnectError as e:
        return [f"Não foi possível conectar ao Ollama em {host}: {e}"]

    available = {m["name"] for m in data.get("models", [])}
    logger.info("Ollama em %s disponibiliza %d modelos.", host, len(available))

    problems: list[str] = []
    for role, slug in models:
        if slug not in available:
            similar = sorted(m for m in available if slug.split(":")[0] in m)[:5]
            problems.append(
                f"{role}: modelo {slug!r} NÃO disponível em {host}. "
                f"Similares: {similar or sorted(available)[:5]}"
            )
        else:
            logger.info("✓ %s: %s disponível em %s", role, slug, host)
    return problems


def _validar_llama_swap_host(
    host: str,
    models: list[tuple[str, str]],  # [(role, slug), ...]
) -> list[str]:
    """Verifica slugs llama-swap via GET /v1/models em um host; retorna problemas."""
    models_url = host.rstrip("/") + "/v1/models"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(models_url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return [f"Falha ao consultar {models_url}: {e}"]
    except httpx.ConnectError as e:
        return [f"Não foi possível conectar ao llama-swap em {host}: {e}"]

    available = {m["id"] for m in data.get("data", [])}
    logger.info("llama-swap em %s disponibiliza %d modelos.", host, len(available))

    problems: list[str] = []
    for role, slug in models:
        if slug not in available:
            prefix = slug.split("-", 1)[0]
            similar = sorted(m for m in available if prefix in m)[:5]
            problems.append(
                f"{role}: modelo {slug!r} NÃO disponível em {host}. "
                f"Similares: {similar or sorted(available)[:5]}"
            )
        else:
            logger.info("✓ %s: %s disponível em %s", role, slug, host)
    return problems


def validar_config_experimento(
    config: ConfigExperimento,
    secrets: Segredos,
) -> None:
    """
    Valida que todos os modelos referenciados existem nos respectivos backends.

    - OpenRouter: uma chamada GET /models cobre todos os slugs OR.
    - Ollama: uma chamada GET /api/tags por host distinto.
    - llama-swap: uma chamada GET /v1/models por host distinto.

    Raises:
        ErroValidacao: lista todos os problemas encontrados.
    """
    openrouter_models: list[tuple[str, str]] = []
    # host → [(role, slug)]
    ollama_por_host: dict[str, list[tuple[str, str]]] = defaultdict(list)
    llama_swap_por_host: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for agent in config.agents:
        role = f"agent[{agent.agent_id}]"
        if agent.backend == "openrouter":
            openrouter_models.append((role, agent.model))
        elif agent.backend == "ollama":
            host = agent.host or secrets.ollama_host_gpu0
            ollama_por_host[host].append((role, agent.model))
        elif agent.backend == "llama-swap":
            host = agent.host or secrets.llama_swap_host
            llama_swap_por_host[host].append((role, agent.model))

    if config.judge.backend == "openrouter":
        openrouter_models.append((f"judge[{config.judge.judge_id}]", config.judge.model))

    problems: list[str] = []

    if openrouter_models:
        problems += _validar_openrouter(
            openrouter_models,
            api_key=secrets.openrouter_api_key,
            base_url=secrets.openrouter_url,
        )

    for host, models in ollama_por_host.items():
        problems += _validar_ollama_host(host, models)

    for host, models in llama_swap_por_host.items():
        problems += _validar_llama_swap_host(host, models)

    if not openrouter_models and not ollama_por_host and not llama_swap_por_host:
        logger.info("Nenhum modelo a validar.")
        return

    if problems:
        msg = "Problemas de validação:\n" + "\n".join(f"  - {p}" for p in problems)
        raise ErroValidacao(msg)

    total = (
        len(openrouter_models)
        + sum(len(v) for v in ollama_por_host.values())
        + sum(len(v) for v in llama_swap_por_host.values())
    )
    logger.info("Validação de configuração OK: %d modelos verificados.", total)
