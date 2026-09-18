"""Cache de rótulos do juiz, indexado por hash de prompt completo."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ensemble_llm.esquemas import RotuloJuiz

logger = logging.getLogger(__name__)


class CacheRotulosJuiz:
    """Cache persistido em parquet, com escrita em batches."""

    def __init__(self, cache_dir: Path, judge_model: str, flush_every: int = 10) -> None:
        sanitizado = judge_model.replace("/", "_")
        self.dir = Path(cache_dir) / sanitizado
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.dir / "cache.parquet"
        self.judge_model = judge_model
        self.flush_every = flush_every

        self._lock = threading.RLock()
        self._pendentes: list[dict] = []
        self._df = self._carregar()

    def _carregar(self) -> pd.DataFrame:
        if self.cache_file.exists():
            df = pd.read_parquet(self.cache_file)
            logger.info("Cache do juiz carregado: %d entradas em %s", len(df), self.cache_file)
            return df
        return pd.DataFrame(
            columns=[
                "prompt_hash",
                "item_id",
                "response_source",
                "judge_model",
                "z",
                "raciocinio",
                "cost_usd",
                "timestamp",
            ]
        )

    def flush(self) -> None:
        """Grava registros pendentes em disco e limpa o buffer em memória."""
        with self._lock:
            if not self._pendentes:
                return
            novas = pd.DataFrame(self._pendentes)
            self._df = pd.concat([self._df, novas], ignore_index=True)
            self._df.to_parquet(self.cache_file, index=False)
            self._pendentes = []

    def obter(self, prompt_hash: str) -> RotuloJuiz | None:
        """Busca rótulo por prompt_hash; verifica pendentes em memória antes do parquet."""
        with self._lock:
            for entrada in self._pendentes:
                if entrada["prompt_hash"] == prompt_hash:
                    return RotuloJuiz(**entrada)
            matches = self._df[self._df["prompt_hash"] == prompt_hash]
            if len(matches) == 0:
                return None
            return RotuloJuiz(**matches.iloc[0].to_dict())

    def adicionar(self, rotulo: RotuloJuiz) -> None:
        """Adiciona rótulo ao buffer de pendentes; idempotente por prompt_hash."""
        with self._lock:
            existente = self.obter(rotulo.prompt_hash)
            if existente is not None:
                return
            self._pendentes.append(rotulo.model_dump())

        if len(self._pendentes) >= self.flush_every:
            self.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._df) + len(self._pendentes)

    def custo_total(self) -> float:
        """Soma custo acumulado em USD de todos os rótulos no cache (persiste + pendentes)."""
        with self._lock:
            custo_df = float(self._df["cost_usd"].sum()) if len(self._df) else 0.0
            custo_pend = sum(float(e["cost_usd"]) for e in self._pendentes)
            return custo_df + custo_pend


def agora_iso() -> str:
    """Retorna timestamp UTC atual em formato ISO 8601."""
    return datetime.now(UTC).isoformat()
