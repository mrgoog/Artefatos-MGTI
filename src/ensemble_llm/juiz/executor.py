"""Executor do juiz: cache + invocação + parsing estruturado."""

from __future__ import annotations

import logging
import time

from ensemble_llm.agentes.cliente_openrouter import ClienteOpenRouter
from ensemble_llm.agentes.estruturado import invocar_estruturado
from ensemble_llm.esquemas import ItemDataset, RespostaJuiz, RotuloJuiz
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz, agora_iso
from ensemble_llm.juiz.rubrica import DICA_SCHEMA_JUIZ, construir_prompt_juiz

logger = logging.getLogger(__name__)


class ExecutorJuiz:
    """Orquestra cache de rótulos + invocação estruturada + fallback para falha de parse do juiz."""

    def __init__(
        self,
        client: ClienteOpenRouter,
        cache: CacheRotulosJuiz,
        k_max: int = 3,
    ) -> None:
        if client.model != cache.judge_model:
            raise ValueError(
                f"Inconsistência: client.model={client.model!r} != "
                f"cache.judge_model={cache.judge_model!r}"
            )
        self.client = client
        self.cache = cache
        self.k_max = k_max

    def julgar(
        self,
        item: ItemDataset,
        resposta_candidata: str,
        fonte_resposta: str,
    ) -> RotuloJuiz:
        """Avalia resposta candidata; retorna rótulo do cache se já foi avaliada antes."""
        t0 = time.perf_counter()
        prompt_sistema, prompt_usuario = construir_prompt_juiz(
            question=item.pergunta,
            ground_truth=item.ground_truth,
            dispositivos_legais=item.dispositivos_legais,
            ementas_carf=item.ementas_carf,
            candidate_response=resposta_candidata,
        )

        from ensemble_llm.agentes.cliente_openrouter import _hash_prompt

        prompt_hash = _hash_prompt(
            prompt_sistema,
            prompt_usuario,
            self.client.model,
            self.client.temperature,
            self.client.max_tokens,
        )

        cached = self.cache.obter(prompt_hash)
        if cached is not None:
            logger.info(
                "Juiz cache hit: item=%s fonte=%s em %.2fs.",
                item.item_id,
                fonte_resposta,
                time.perf_counter() - t0,
            )
            return cached
        logger.info("Juiz cache miss: item=%s fonte=%s.", item.item_id, fonte_resposta)

        result = invocar_estruturado(
            self.client,
            user_prompt=prompt_usuario,
            schema=RespostaJuiz,
            system_prompt=prompt_sistema,
            schema_hint=DICA_SCHEMA_JUIZ,
            k_max=self.k_max,
        )

        if not result.valid_json or result.parsed is None:
            rotulo = RotuloJuiz(
                item_id=item.item_id,
                response_source=fonte_resposta,
                prompt_hash=prompt_hash,
                judge_model=self.client.model,
                z=0,
                raciocinio=f"[FALHA DE PARSE] Output bruto: {result.raw_output[:300]}",
                cost_usd=result.cost_usd,
                timestamp=agora_iso(),
            )
        else:
            resposta_juiz: RespostaJuiz = result.parsed  # type: ignore[assignment]
            rotulo = RotuloJuiz(
                item_id=item.item_id,
                response_source=fonte_resposta,
                prompt_hash=prompt_hash,
                judge_model=self.client.model,
                z=resposta_juiz.veredicto,
                raciocinio=resposta_juiz.raciocinio,
                cost_usd=result.cost_usd,
                timestamp=agora_iso(),
            )

        self.cache.adicionar(rotulo)
        logger.info(
            "Juiz persistiu rótulo: item=%s fonte=%s z=%d json_ok=%s em %.2fs.",
            item.item_id,
            fonte_resposta,
            int(rotulo.z),
            result.valid_json,
            time.perf_counter() - t0,
        )
        return rotulo
