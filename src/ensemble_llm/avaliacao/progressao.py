"""Avaliação offline da progressão B1 → B2 → B3 → Full com ICs bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado
from ensemble_llm.avaliacao.metricas import cobertura_empirica, risco_seletivo
from ensemble_llm.configuracao import ConfigExperimento, carregar_config_experimento


@dataclass(frozen=True)
class ResultadoProgressaoBootstrap:
    """Artefato materializado da análise bootstrap da progressão."""

    run_dir: Path
    summary_path: Path


def _resolver_run_dir(run: str | Path, base_dir: Path) -> Path:
    run_path = Path(run)
    if run_path.exists():
        return run_path
    return base_dir / run


def _carregar_parquet_obrigatorio(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Consolidado não encontrado: {path}")
    return pd.read_parquet(path)


def _normalizar_item_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["item_id"] = out["item_id"].astype(str)
    return out


def _validar_sem_duplicatas(nome: str, df: pd.DataFrame) -> None:
    duplicados = int(df["item_id"].duplicated().sum())
    if duplicados > 0:
        raise ValueError(f"{nome} contém {duplicados} item_ids duplicados")


def carregar_adequacao_por_item(run_dir: Path, *, c_min: float) -> pd.DataFrame:
    """Materializa tabela pareada item-a-item para a progressão B1 → Full."""
    b1 = _normalizar_item_id(
        _carregar_parquet_obrigatorio(run_dir / "consolidated_test_B1.parquet")
    )
    b2 = _normalizar_item_id(
        _carregar_parquet_obrigatorio(run_dir / "consolidated_test_B2.parquet")
    )
    b3 = _normalizar_item_id(
        _carregar_parquet_obrigatorio(run_dir / "consolidated_test_B3.parquet")
    )
    full = _normalizar_item_id(
        _carregar_parquet_obrigatorio(run_dir / f"consolidated_test_full_cmin={c_min:.1f}.parquet")
    )

    for nome, df in (("B1", b1), ("B2", b2), ("B3", b3), ("Full", full)):
        _validar_sem_duplicatas(nome, df)

    ids_b1 = set(b1["item_id"])
    for nome, df in (("B2", b2), ("B3", b3), ("Full", full)):
        ids_df = set(df["item_id"])
        if ids_df != ids_b1:
            discrepantes = len(ids_b1.symmetric_difference(ids_df))
            raise ValueError(
                f"Conjuntos de item_id divergem entre B1 e {nome}: "
                f"{discrepantes} itens discrepantes"
            )

    paired = b1[["item_id", "z_winner"]].rename(columns={"z_winner": "z_B1"})
    paired = paired.merge(
        b2[["item_id", "z_winner"]].rename(columns={"z_winner": "z_B2"}),
        on="item_id",
        validate="one_to_one",
    )
    paired = paired.merge(
        b3[["item_id", "z_winner"]].rename(columns={"z_winner": "z_B3"}),
        on="item_id",
        validate="one_to_one",
    )

    full_adeq = full[["item_id"]].copy()
    full_adeq["z_Full"] = np.where(
        ~full["abstained"].astype(bool).to_numpy(),
        pd.to_numeric(full["z_winner"], errors="coerce").fillna(0).astype(int).to_numpy(),
        0,
    )
    paired = paired.merge(full_adeq, on="item_id", validate="one_to_one")
    paired = paired.merge(
        full[["item_id", "has_doc_ref"]],
        on="item_id",
        validate="one_to_one",
    )

    if len(paired) != len(ids_b1):
        raise RuntimeError(
            f"Pareamento incompleto: esperado {len(ids_b1)} itens, obteve {len(paired)}"
        )

    for coluna in ("z_B1", "z_B2", "z_B3", "z_Full"):
        paired[coluna] = pd.to_numeric(paired[coluna], errors="raise").astype(int)
    paired["has_doc_ref"] = paired["has_doc_ref"].astype(bool)
    return paired.sort_values("item_id", kind="stable").reset_index(drop=True)


def bootstrap_delta_pareado(
    paired: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    config: ConfigExperimento,
    stratify_col: str = "has_doc_ref",
) -> dict[str, Any]:
    """Computa IC bootstrap BCa para Δ = mean(col_b - col_a)."""
    dados = paired.copy()
    diff_col = f"_diff_{col_a}_{col_b}"
    dados[diff_col] = dados[col_b].astype(int) - dados[col_a].astype(int)

    result = ic_bootstrap_estratificado(
        dados,
        lambda df: float(df[diff_col].mean()),
        coluna_estratificacao=stratify_col,
        n_reamostras=config.bootstrap.n_resamples,
        nivel_confianca=config.bootstrap.confidence_level,
        metodo=config.bootstrap.method,
        seed=config.bootstrap.seed,
    )
    return {
        "point": result.point,
        "ci_lo": result.ci_lo,
        "ci_hi": result.ci_hi,
        "confidence_level": result.confidence_level,
        "method": result.method,
        "n_resamples": result.n_resamples,
        "n_dropped": result.n_dropped,
        "includes_zero": bool(result.ci_lo <= 0.0 <= result.ci_hi),
    }


def _bootstrap_media_coluna(
    paired: pd.DataFrame,
    coluna: str,
    *,
    config: ConfigExperimento,
) -> dict[str, Any]:
    result = ic_bootstrap_estratificado(
        paired,
        lambda df: float(df[coluna].mean()),
        coluna_estratificacao="has_doc_ref",
        n_reamostras=config.bootstrap.n_resamples,
        nivel_confianca=config.bootstrap.confidence_level,
        metodo=config.bootstrap.method,
        seed=config.bootstrap.seed,
    )
    return result.model_dump()


def _precision_abstention_full(dados: pd.DataFrame) -> float:
    """Precisão da abstenção no Full via `z_winner` nos itens abstidos."""
    abstidos = dados[dados["abstained"]]
    if len(abstidos) == 0:
        return float("nan")
    return float((1 - abstidos["z_winner"].astype(int)).mean())


def _ic_metrica_full(
    full: pd.DataFrame,
    *,
    nome_metrica: str,
    config: ConfigExperimento,
) -> dict[str, Any] | None:
    funcoes = {
        "risk_selective": risco_seletivo,
        "coverage": cobertura_empirica,
        "precision_abstention": _precision_abstention_full,
    }
    if nome_metrica not in funcoes:
        raise ValueError(f"Métrica desconhecida: {nome_metrica}")
    try:
        return ic_bootstrap_estratificado(
            full,
            funcoes[nome_metrica],
            coluna_estratificacao="has_doc_ref",
            n_reamostras=config.bootstrap.n_resamples,
            nivel_confianca=config.bootstrap.confidence_level,
            metodo=config.bootstrap.method,
            seed=config.bootstrap.seed,
        ).model_dump()
    except (RuntimeError, ValueError):
        return None


def gerar_resumo_progressao_bootstrap(
    *,
    run: str | Path,
    config_path: str | Path,
    base_dir: Path = Path("runs"),
    primary_c_min: float = 0.7,
    output_json: Path | None = None,
) -> ResultadoProgressaoBootstrap:
    """Gera JSON com ICs bootstrap da progressão B1 → Full e trade-off por C_min."""
    config = carregar_config_experimento(config_path)
    run_dir = _resolver_run_dir(run, base_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run não encontrada: {run_dir}")

    paired = carregar_adequacao_por_item(run_dir, c_min=primary_c_min)
    transicoes = [
        ("B1→B2", "z_B1", "z_B2"),
        ("B2→B3", "z_B2", "z_B3"),
        ("B3→Full", "z_B3", "z_Full"),
        ("B1→Full", "z_B1", "z_Full"),
    ]

    progression = {
        "paired_n": int(len(paired)),
        "stratify_col": "has_doc_ref",
        "deltas": {
            nome: bootstrap_delta_pareado(paired, col_a, col_b, config=config)
            for nome, col_a, col_b in transicoes
        },
        "absolutes": {
            nome: _bootstrap_media_coluna(paired, coluna, config=config)
            for nome, coluna in (
                ("B1", "z_B1"),
                ("B2", "z_B2"),
                ("B3", "z_B3"),
                ("Full", "z_Full"),
            )
        },
    }

    cmin_metrics: dict[str, dict[str, Any] | None] = {}
    for c_min in config.decision.c_min_grid:
        full = _carregar_parquet_obrigatorio(
            run_dir / f"consolidated_test_full_cmin={c_min:.1f}.parquet"
        ).copy()
        full["has_doc_ref"] = full["has_doc_ref"].astype(bool)
        full["abstained"] = full["abstained"].astype(bool)
        if "z_system" in full.columns:
            full["z_system"] = pd.to_numeric(full["z_system"], errors="coerce")
        full["z_winner"] = pd.to_numeric(full["z_winner"], errors="coerce").fillna(0).astype(int)

        cmin_metrics[f"full_cmin={c_min:.1f}"] = {
            "coverage": _ic_metrica_full(full, nome_metrica="coverage", config=config),
            "risk_selective": _ic_metrica_full(full, nome_metrica="risk_selective", config=config),
            "precision_abstention": _ic_metrica_full(
                full,
                nome_metrica="precision_abstention",
                config=config,
            ),
        }

    summary = {
        "run": run_dir.name,
        "config_path": str(Path(config_path)),
        "primary_c_min": primary_c_min,
        "bootstrap": {
            "n_resamples": config.bootstrap.n_resamples,
            "confidence_level": config.bootstrap.confidence_level,
            "method": config.bootstrap.method,
            "seed": config.bootstrap.seed,
        },
        "progression": progression,
        "cmin_metrics": cmin_metrics,
    }

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_json
        if output_json is not None
        else metrics_dir / f"progressao_bootstrap_cmin={primary_c_min:.1f}.json"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return ResultadoProgressaoBootstrap(run_dir=run_dir, summary_path=summary_path)
