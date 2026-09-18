"""Helpers de caminho para artefatos em runs/<experiment_id>/."""

from __future__ import annotations

from pathlib import Path


def diretorio_execucao(base_dir: Path, experiment_id: str) -> Path:
    """Raiz do diretório de artefatos de um experimento em runs/<experiment_id>/."""
    return base_dir / experiment_id


def caminho_manifesto(base_dir: Path, experiment_id: str) -> Path:
    """Caminho do manifest.yaml com snapshot de reprodutibilidade do experimento."""
    return diretorio_execucao(base_dir, experiment_id) / "manifest.yaml"


def diretorio_rag(base_dir: Path, experiment_id: str) -> Path:
    """Diretório do índice RAG persistido (denso + BM25)."""
    return diretorio_execucao(base_dir, experiment_id) / "rag_index"


def caminho_respostas_agente(base_dir: Path, experiment_id: str, agent_id: str) -> Path:
    """Parquet com respostas e metadados de um agente para todos os itens (Nível 1 de cache)."""
    return diretorio_execucao(base_dir, experiment_id) / "agent_responses" / agent_id / "responses.parquet"


def diretorio_cache_juiz(base_dir: Path, experiment_id: str) -> Path:
    """Diretório dos caches de rótulos do juiz por modelo (Nível 2 de cache)."""
    return diretorio_execucao(base_dir, experiment_id) / "judge_labels"


def diretorio_fold(base_dir: Path, experiment_id: str, fold_k: int, c_min: float) -> Path:
    """Diretório de artefatos de um fold específico (pesos, limiar, decisões de teste)."""
    return diretorio_execucao(base_dir, experiment_id) / "folds" / f"fold_k={fold_k:02d}" / f"cmin={c_min:.1f}"


def caminho_decisoes_teste_fold(base_dir: Path, experiment_id: str, fold_k: int, c_min: float) -> Path:
    """Parquet com decisões out-of-sample do conjunto de teste de um fold."""
    return diretorio_fold(base_dir, experiment_id, fold_k, c_min) / "test_decisions.parquet"


def caminho_pesos_fold(base_dir: Path, experiment_id: str, fold_k: int, c_min: float) -> Path:
    """JSON com pesos w_i ajustados em D_fit de um fold.

    O arquivo é idêntico entre os subdiretórios `cmin=X.X/` do mesmo fold.
    A duplicação é deliberada para manter a estrutura de caminhos simples.
    """
    return diretorio_fold(base_dir, experiment_id, fold_k, c_min) / "weights.json"


def caminho_limiar_fold(base_dir: Path, experiment_id: str, fold_k: int, c_min: float) -> Path:
    """JSON com τ* selecionado em D_val de um fold."""
    return diretorio_fold(base_dir, experiment_id, fold_k, c_min) / "threshold.json"


def caminho_teste_consolidado(
    base_dir: Path,
    experiment_id: str,
    c_min: float | None = None,
    baseline: str = "full",
    agent_id: str | None = None,
) -> Path:
    """Parquet consolidado por (baseline, c_min).

    Para baseline="full", c_min é obrigatório (cada valor da grade gera um arquivo).
    Para baselines B1/B2/B3, c_min é ignorado (não há abstenção).
    Para baseline="Bsolo", `agent_id` e `c_min` são obrigatórios.
    """
    if baseline == "full":
        if c_min is None:
            raise ValueError("c_min é obrigatório para baseline='full'")
        nome = f"consolidated_test_full_cmin={c_min:.1f}.parquet"
    elif baseline == "Bsolo":
        if c_min is None:
            raise ValueError("c_min é obrigatório para baseline='Bsolo'")
        if agent_id is None:
            raise ValueError("agent_id é obrigatório para baseline='Bsolo'")
        nome = f"consolidated_test_Bsolo_agent={agent_id}_cmin={c_min:.1f}.parquet"
    elif baseline in ("B1", "B2", "B3"):
        nome = f"consolidated_test_{baseline}.parquet"
    else:
        raise ValueError(f"baseline desconhecido: {baseline!r}")
    return diretorio_execucao(base_dir, experiment_id) / nome


def diretorio_metricas(base_dir: Path, experiment_id: str) -> Path:
    """Diretório de métricas e resumos estatísticos do experimento."""
    return diretorio_execucao(base_dir, experiment_id) / "metrics"


def diretorio_reliability(base_dir: Path, experiment_id: str, fold_k: int) -> Path:
    """Diretório dos parquets de reliability diagram para um fold específico."""
    return diretorio_metricas(base_dir, experiment_id) / f"reliability_k={fold_k:02d}"


def caminho_curva_risco_cobertura(base_dir: Path, experiment_id: str) -> Path:
    """Parquet com a curva (τ, cobertura, risco_seletivo, n_respondidos)."""
    return diretorio_metricas(base_dir, experiment_id) / "risk_coverage_curve.parquet"


def caminho_pontos_tau_estrela(base_dir: Path, experiment_id: str) -> Path:
    """Parquet com os pontos (c_min, tau_star_medio) para marcar na curva."""
    return diretorio_metricas(base_dir, experiment_id) / "tau_star_points.parquet"
