"""Wrapper estruturado para invocações de LLM."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ensemble_llm.agentes.contratos import ClienteLLM, InvocacaoEstruturada

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_PADRAO_OBJETO_JSON = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extrair_json(texto: str | None) -> str | None:
    """Extrai o primeiro objeto JSON válido de texto com possível markdown ou texto livre."""
    if texto is None:
        return None
    texto_limpo = re.sub(r"```(?:json)?\s*", "", texto)
    texto_limpo = re.sub(r"```\s*$", "", texto_limpo).strip()
    try:
        json.loads(texto_limpo)
        return texto_limpo
    except json.JSONDecodeError:
        pass

    match = _PADRAO_OBJETO_JSON.search(texto_limpo)
    if match:
        candidato = match.group(0)
        try:
            json.loads(candidato)
            return candidato
        except json.JSONDecodeError:
            return None
    return None


_extract_json = _extrair_json


def _canonizar_chave(chave: str) -> str:
    """Normaliza chave textual para comparação case/acento-insensível."""
    decomposed = unicodedata.normalize("NFKD", chave)
    sem_acentos = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return sem_acentos.casefold().strip()


def _normalizar_chaves_para_schema(payload: object, schema: type[T]) -> object:
    """
    Mapeia chaves de nível raiz para campos do schema por equivalência canônica.

    Exemplo: ``confiança`` / ``CONFIANÇA`` / ``Confianca`` -> ``confianca``.
    """
    if not isinstance(payload, dict):
        return payload

    aliases: dict[str, str] = {}
    for field_name in schema.model_fields:
        aliases[_canonizar_chave(field_name)] = field_name

    normalizado: dict[object, object] = dict(payload)
    for chave_original in list(payload.keys()):
        if not isinstance(chave_original, str):
            continue
        chave_canonica = _canonizar_chave(chave_original)
        destino = aliases.get(chave_canonica)
        if destino is None or destino == chave_original:
            continue
        # Prioriza sempre o nome canônico do schema quando ambos aparecerem.
        if destino not in normalizado:
            normalizado[destino] = normalizado[chave_original]
        del normalizado[chave_original]
    return normalizado


def _construir_prompt_tentativa(
    prompt_original: str, output_anterior: str | None, dica_schema: str
) -> str:
    """Monta prompt de reintento mostrando o erro anterior e forçando formato JSON exato."""
    output_referencia = (output_anterior or "")[:500]
    return (
        "Sua resposta anterior não pôde ser parseada como JSON válido.\n\n"
        f"RESPOSTA ANTERIOR (para referência):\n{output_referencia}\n\n"
        "INSTRUÇÃO REVISADA:\n"
        "Responda APENAS com JSON no formato exato abaixo. "
        "Não inclua texto antes ou depois. Não use blocos de código markdown.\n\n"
        f"FORMATO EXATO:\n{dica_schema}\n\n"
        f"REQUISIÇÃO ORIGINAL:\n{prompt_original}"
    )


def invocar_estruturado(
    client: ClienteLLM,
    user_prompt: str,
    schema: type[T],
    *,
    system_prompt: str | None = None,
    schema_hint: str,
    k_max: int = 3,
) -> InvocacaoEstruturada:
    """Invoca o LLM com até k_max tentativas até obter JSON válido conforme o schema T."""
    prompt_atual = user_prompt
    ultimo_output = ""
    custo_total = 0.0
    prompt_hash = ""

    for tentativa in range(k_max):
        result = client.invoke(prompt_atual, system_prompt=system_prompt)
        ultimo_output = result.raw_output or ""
        custo_total += result.cost_usd
        if tentativa == 0:
            prompt_hash = result.prompt_hash

        texto_json = _extrair_json(ultimo_output)
        if texto_json is None:
            prompt_atual = _construir_prompt_tentativa(user_prompt, ultimo_output, schema_hint)
            continue

        try:
            payload = json.loads(texto_json)
            payload = _normalizar_chaves_para_schema(payload, schema)
            parsed = schema.model_validate(payload)
            return InvocacaoEstruturada(
                parsed=parsed,
                raw_output=ultimo_output,
                prompt_hash=prompt_hash,
                model=result.model,
                n_retries=tentativa,
                valid_json=True,
                cost_usd=custo_total,
            )
        except ValidationError:
            prompt_atual = _construir_prompt_tentativa(user_prompt, ultimo_output, schema_hint)

    return InvocacaoEstruturada(
        parsed=None,
        raw_output=ultimo_output,
        prompt_hash=prompt_hash,
        model=client.model,
        n_retries=k_max,
        valid_json=False,
        cost_usd=custo_total,
    )
