"""Configuração tipada do experimento via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Segredos(BaseSettings):
    """Variáveis sensíveis carregadas do .env (API keys, hosts, seed global)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions", alias="OPENROUTER_URL")
    openrouter_referer: str = Field(default="http://localhost:2718", alias="OPENROUTER_REFERER")
    ollama_host_gpu0: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST_GPU0")
    ollama_host_gpu1: str = Field(default="http://localhost:11435", alias="OLLAMA_HOST_GPU1")
    llama_swap_host: str = Field(default="http://localhost:8080", alias="LLAMA_SWAP_HOST")
    runs_base_dir: Path = Field(default=Path("./runs"), alias="RUNS_BASE_DIR")
    experiment_seed: int = Field(default=42, alias="EXPERIMENT_SEED")


class ConfigAgente(BaseModel):
    """Parâmetros de um agente LLM (backend, modelo, temperatura, seed)."""

    agent_id: str = Field(..., description="ID lógico estável; ex: 'phi4_mini'")
    backend: Literal["openrouter", "ollama", "llama-swap"]
    model: str = Field(..., description="Slug do modelo no backend")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64)
    seed: int = Field(default=42)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    host: str | None = None


class ConfigJuiz(BaseModel):
    """Parâmetros do juiz frontier (backend OpenRouter, modelo e temperatura)."""

    judge_id: str = Field(default="judge_main")
    backend: Literal["openrouter"] = "openrouter"
    model: str = Field(..., description="Slug do modelo juiz; ex: 'openai/gpt-5.4-mini'")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64)
    seed: int = Field(default=42)


class ConfigRetrieval(BaseModel):
    """Parâmetros do pipeline RAG (embedder, tamanho de chunk, top-k denso e BM25)."""

    embedder_model: str = Field(default="BAAI/bge-m3")
    embedder_device: str | None = Field(
        default=None,
        description="Dispositivo explícito do embedder; ex: 'cuda:0', 'cuda:1' ou 'cpu'.",
    )
    embedder_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size do embedder para indexação e consultas densas.",
    )
    persisted_index_dir: Path | None = Field(
        default=None,
        description=(
            "Diretório compartilhado do índice RAG offline. "
            "Quando definido, as execuções carregam o índice daqui em vez de reconstruí-lo em runs/<experiment_id>/."
        ),
    )
    chunk_max_tokens: int = Field(default=512, ge=64)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    dense_top_k: int = Field(default=5, ge=1)
    bm25_top_k: int = Field(default=5, ge=1)
    merge_strategy: Literal["union_dedup", "rrf"] = "union_dedup"


class ConfigParticao(BaseModel):
    """Parâmetros do Stratified K-fold tri-partite (n_splits, n_val_folds, estratificação)."""

    n_splits: int = Field(default=10, ge=2)
    n_val_folds: int = Field(default=2, ge=1)
    stratify_by: str = Field(default="has_doc_ref")

    @field_validator("n_val_folds")
    @classmethod
    def val_less_than_total(cls, v: int, info) -> int:  # noqa: ANN001
        n_splits = info.data.get("n_splits", 10)
        if v >= n_splits:
            raise ValueError(f"n_val_folds ({v}) must be < n_splits ({n_splits})")
        return v


class ConfigDecisao(BaseModel):
    """Parâmetros da política seletiva: grade de C_min e regra de desempate."""

    c_min_grid: list[float] = Field(
        default_factory=lambda: [0.5, 0.7, 0.8, 0.9],
        min_length=1,
        description="Grade de cobertura mínima; varredura para análise de sensibilidade.",
    )
    tiebreak_strategy: Literal["highest_weight", "lexicographic"] = "highest_weight"
    method: Literal["weighted_calibrated", "mean_raw"] = "weighted_calibrated"
    """Cabeça de decisão. 'weighted_calibrated': s_i = w_i·ĉ_i,
    vencedor = argmax s_i, η = 1 − max_i s_i. 'mean_raw' (pipeline minimalista):
    vencedor = argmax c_bruta_i, η = 1 − média_i(c_bruta_i); ignora pesos e calibração."""

    @field_validator("c_min_grid")
    @classmethod
    def _validar_grade(cls, v: list[float]) -> list[float]:
        for c in v:
            if not (0.0 < c <= 1.0):
                raise ValueError(f"Cada valor de c_min_grid deve estar em (0,1], recebeu {c}")
        return sorted(set(v))


class ConfigBootstrap(BaseModel):
    """Parâmetros do bootstrap BCa estratificado (B=10.000 reamostras, seed, nível de confiança)."""

    n_resamples: int = Field(default=10000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: Literal["BCa", "percentile"] = "BCa"
    seed: int = Field(default=42)


class ConfigDataset(BaseModel):
    """Arquivos físicos que compõem um dataset compatível com o contrato canônico do pipeline."""

    dataset_id: str = Field(..., description="ID lógico estável do dataset; ex: 'brtaxq_2024_v1_1'")
    format: Literal["brtaxqa_compatible"] = Field(
        default="brtaxqa_compatible",
        description="Formato lógico esperado pelo loader; desacoplado do nome do dataset.",
    )
    root_dir: Path = Field(..., description="Diretório raiz onde vivem os arquivos do dataset.")
    questions_file: Path = Field(..., description="Arquivo principal com as questões.")
    corpus_files: list[Path] = Field(
        default_factory=list,
        description="Arquivos adicionais usados para montar o corpus RAG.",
    )

    @field_validator("corpus_files")
    @classmethod
    def _remover_duplicatas_preservando_ordem(cls, v: list[Path]) -> list[Path]:
        vistos: set[Path] = set()
        resultado: list[Path] = []
        for path in v:
            if path in vistos:
                continue
            vistos.add(path)
            resultado.append(path)
        return resultado

    def resolver_questions_path(self) -> Path:
        """Resolve o caminho do arquivo principal de questões."""
        return self.root_dir / self.questions_file

    def resolver_corpus_paths(self) -> list[Path]:
        """Resolve os caminhos dos arquivos extras do corpus RAG."""
        return [self.root_dir / path for path in self.corpus_files]


class ConfigExperimento(BaseModel):
    """Configuração declarativa completa do experimento, carregada de um YAML versionado."""

    experiment_id: str = Field(..., description="ID estável do experimento; usado em runs/<experiment_id>/")
    phase: Literal["phase1_api_pilot", "phase2_local_full"]
    sample_size: int | None = Field(default=None, description="None = dataset completo; senão amostra aleatória deste tamanho")
    parallel_agents: bool = Field(default=False, description="Se True, agentes rodam em paralelo (Fase 2 com 2 GPUs)")

    dataset: ConfigDataset
    agents: list[ConfigAgente] = Field(..., min_length=2)
    judge: ConfigJuiz
    retrieval: ConfigRetrieval = ConfigRetrieval()
    partition: ConfigParticao = ConfigParticao()
    decision: ConfigDecisao = ConfigDecisao()
    bootstrap: ConfigBootstrap = ConfigBootstrap()

    @field_validator("agents")
    @classmethod
    def unique_agent_ids(cls, v: list[ConfigAgente]) -> list[ConfigAgente]:
        ids = [a.agent_id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent_id duplicado em agents: {ids}")
        return v


class ConfigRagOffline(BaseModel):
    """Configuração mínima para build e inspeção de um índice RAG compartilhado."""

    dataset: ConfigDataset
    retrieval: ConfigRetrieval


def carregar_config_experimento(path: str | Path) -> ConfigExperimento:
    """Lê e valida ConfigExperimento a partir de um arquivo YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigExperimento.model_validate(data)


def carregar_config_rag_offline(path: str | Path) -> ConfigRagOffline:
    """Lê e valida ConfigRagOffline a partir de um arquivo YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigRagOffline.model_validate(data)
