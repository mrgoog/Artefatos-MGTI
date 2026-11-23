"""Esquemas Pydantic — contratos de dados do pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ItemDataset(BaseModel):
    """Questão canônica do dataset BR-TaxQA-R (pergunta + ground truth + referências legais)."""

    model_config = ConfigDict(frozen=True)
    item_id: str = Field(..., description="ID estável do item no BR-TaxQA-R")
    pergunta: str
    ground_truth: str = Field(..., description="Resposta de referência oficial")
    dispositivos_legais: list[str] = Field(default_factory=list)
    ementas_carf: list[str] = Field(default_factory=list)
    has_doc_ref: bool = Field(..., description="True se possui dispositivos legais OU ementas CARF associados")


class BlocoDocumento(BaseModel):
    """Chunk de texto de um documento legal, produzido pela segmentação do corpus RAG."""

    model_config = ConfigDict(frozen=True)
    chunk_id: str
    text: str
    source_doc: str = Field(..., description="ID do documento de origem")
    section_path: str | None = None
    n_tokens: int


class BlocoRecuperado(BaseModel):
    """Chunk retornado pelo recuperador híbrido com escores denso e BM25."""

    chunk_id: str
    text: str
    source_doc: str
    dense_score: float | None = None
    bm25_score: float | None = None
    combined_rank: int = Field(..., description="Posição final após merge (1 = top)")


class RespostaAgente(BaseModel):
    """Saída JSON do agente: resposta textual + confiança autodeclarada ∈ [0, 1]."""

    resposta: str = Field(..., min_length=1, max_length=4000)
    confianca: float = Field(..., ge=0.0, le=1.0)


class InvocacaoAgente(BaseModel):
    """Registro completo de uma invocação de agente: prompt, resposta, retries e custo."""

    item_id: str
    agent_id: str
    prompt_hash: str = Field(..., description="SHA-256 hex do prompt completo")
    response: RespostaAgente | None = None
    n_retries: int = Field(0, ge=0)
    valid_json: bool
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    timestamp: str
    raw_output: str = Field("", description="Output bruto da última tentativa, para auditoria")


class RespostaJuiz(BaseModel):
    """Saída JSON do juiz: raciocínio chain-of-thought + veredicto binário (0=inadequada, 1=adequada)."""

    raciocinio: str = Field(..., min_length=10, max_length=4000)
    veredicto: int = Field(..., ge=0, le=1, description="0=inadequada, 1=adequada")


class RotuloJuiz(BaseModel):
    """Rótulo binário persistido no cache do juiz, indexado por prompt_hash."""

    item_id: str
    response_source: str
    prompt_hash: str
    judge_model: str
    z: int = Field(..., ge=0, le=1)
    raciocinio: str
    cost_usd: float = Field(0.0, ge=0.0)
    timestamp: str


class ParticaoTri(BaseModel):
    """Divisão fit/val/test de uma execução k do Stratified K-fold."""

    model_config = ConfigDict(frozen=True)
    k: int = Field(..., ge=0, description="Índice da execução / fold de teste")
    fit_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.fit_ids) + len(self.val_ids) + len(self.test_ids)


class PesosAgentes(BaseModel):
    """Pesos w_i normalizados por adequação validada em D_fit."""

    weights: dict[str, float]
    alphas_raw: dict[str, float] = Field(..., description="Frações de adequação não-normalizadas em D_fit")
    fit_size: int


class ArtefatoIsotonico(BaseModel):
    """Metadados diagnósticos do calibrador isotônico ajustado para um agente."""

    agent_id: str
    fit_size: int
    x_min: float
    x_max: float
    n_breakpoints: int


class LimiarSeletivo(BaseModel):
    """Limiar τ* ótimo da política seletiva e diagnóstico da restrição C_min."""

    tau_star: float = Field(..., ge=0.0, le=1.0)
    c_min: float = Field(..., gt=0.0, le=1.0)
    coverage_emp_val: float = Field(..., ge=0.0, le=1.0)
    risk_sel_emp_val: float = Field(..., ge=0.0, le=1.0)
    val_size: int
    constraint_satisfied: bool


class DecisaoSistema(BaseModel):
    """Decisão completa do sistema para um item: agente vencedor, η, abstenção e rótulos."""

    item_id: str
    fold_k: int
    confidences_raw: dict[str, float]
    confidences_calibrated: dict[str, float]
    weights: dict[str, float]
    scores: dict[str, float]
    agent_chosen: str
    eta: float = Field(..., ge=0.0, le=1.0)
    abstained: bool
    response_text: str | None = None
    z_per_agent: dict[str, int | None] = Field(default_factory=dict)
    z_system: int | None = None


class ICBootstrap(BaseModel):
    """Intervalo de confiança bootstrap (BCa ou percentil) com estimativa pontual e diagnóstico."""

    point: float
    ci_lo: float
    ci_hi: float
    confidence_level: float = 0.95
    method: Literal["BCa", "percentile"] = "BCa"
    n_resamples: int
    n_dropped: int = 0


class ResultadoMetrica(BaseModel):
    """Métrica escalar com IC bootstrap opcional e rótulo de estrato."""

    name: str
    value: float
    ci: ICBootstrap | None = None
    description: str = ""
    stratum: str | None = None


class MetricasExperimento(BaseModel):
    """Métricas consolidadas de um experimento completo (risco, cobertura, ECE, diversidade)."""

    risk_selective: list[ResultadoMetrica]
    coverage: list[ResultadoMetrica]
    ece_per_agent_raw: dict[str, ResultadoMetrica]
    ece_per_agent_calibrated: dict[str, ResultadoMetrica]
    weights_stability: dict[str, dict[str, float]]
    tau_star_stability: dict[str, float]
    diversity_matrix: dict[str, dict[str, float]]
