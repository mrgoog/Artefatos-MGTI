"""Avaliação offline do baseline B_solo a partir de uma run existente."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado
from ensemble_llm.avaliacao.metricas import cobertura_empirica, risco_seletivo
from ensemble_llm.calibracao.isotonica import ajustar_calibrador_isotonico
from ensemble_llm.configuracao import ConfigExperimento, carregar_config_experimento
from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r
from ensemble_llm.dados.particionamento import criar_particoes, validar_cobertura_completa
from ensemble_llm.decisao import baselines as bl
from ensemble_llm.persistencia.caminhos import caminho_teste_consolidado
from ensemble_llm.persistencia.manifesto import ler_manifesto
from ensemble_llm.persistencia.tabelas import escrever_parquet


@dataclass(frozen=True)
class ResultadoBSoloOffline:
    """Resultado materializado da avaliação offline do baseline B_solo."""

    run_dir: Path
    summary_path: Path
    item_level_csv_path: Path
    parquets_por_agente_cmin: dict[tuple[str, float], Path]


def _risk_selective_bsolo(dados: pd.DataFrame) -> float:
    respondidos = dados[~dados["abstained"]]
    if len(respondidos) == 0:
        return float("nan")
    return float((1 - respondidos["z_agent"]).mean())


def _risk_selective_from_col(dados: pd.DataFrame, z_col: str) -> float:
    respondidos = dados[~dados["abstained"]]
    if len(respondidos) == 0:
        return float("nan")
    return float((1 - respondidos[z_col]).mean())


def _adequacy_useful(dados: pd.DataFrame, z_col: str) -> float:
    if len(dados) == 0:
        return float("nan")
    respondidos = pd.to_numeric(dados.loc[~dados["abstained"], z_col], errors="coerce").fillna(0.0)
    return float(respondidos.sum() / len(dados))


def _precision_abstention(dados: pd.DataFrame, z_col: str) -> float:
    abstidos = dados[dados["abstained"]]
    if len(abstidos) == 0:
        return float("nan")
    return float((1 - abstidos[z_col].astype(int)).mean())


def _base_rate_inadequacy(dados: pd.DataFrame, z_col: str) -> float:
    if len(dados) == 0:
        return float("nan")
    return float((1 - dados[z_col].astype(int)).mean())


def _lift_pp(dados: pd.DataFrame, z_col: str) -> float:
    precisao = _precision_abstention(dados, z_col)
    if not np.isfinite(precisao):
        return float("nan")
    return float(precisao - _base_rate_inadequacy(dados, z_col))


def _auroc(dados: pd.DataFrame, z_col: str, eta_col: str | None) -> float:
    if eta_col is None or eta_col not in dados.columns:
        return float("nan")
    eta = pd.to_numeric(dados[eta_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(eta).all():
        return float("nan")
    y_true = 1 - dados[z_col].astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, eta))


def _n_abstentions(dados: pd.DataFrame) -> float:
    return float(dados["abstained"].sum())


def _compactar_df_bootstrap(
    dados: pd.DataFrame,
    *,
    bool_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    float_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Converte colunas para dtypes densos do numpy antes do bootstrap."""
    out = dados.copy()
    for col in bool_cols or []:
        if col in out.columns:
            out[col] = out[col].astype(bool)
    for col in int_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def _bootstrap_metric(
    dados: pd.DataFrame,
    fn_estatistica: Any,
    *,
    config: ConfigExperimento,
) -> dict[str, Any] | None:
    try:
        return ic_bootstrap_estratificado(
            dados,
            fn_estatistica,
            coluna_estratificacao="has_doc_ref",
            n_reamostras=config.bootstrap.n_resamples,
            nivel_confianca=config.bootstrap.confidence_level,
            metodo=config.bootstrap.method,
            seed=config.bootstrap.seed,
        ).model_dump()
    except (RuntimeError, ValueError):
        return None


def _summarize_points(
    dados: pd.DataFrame,
    *,
    z_col: str,
    eta_col: str | None,
    risk_fn: Any,
) -> dict[str, float | int]:
    return {
        "coverage": cobertura_empirica(dados),
        "risk_selective": risk_fn(dados),
        "adequacy_useful": _adequacy_useful(dados, z_col),
        "precision_abstention": _precision_abstention(dados, z_col),
        "base_rate_inadequacy": _base_rate_inadequacy(dados, z_col),
        "lift_pp": _lift_pp(dados, z_col),
        "auroc": _auroc(dados, z_col, eta_col),
        "n_abstentions": int(dados["abstained"].sum()),
        "n_items": int(len(dados)),
    }


def _summarize_with_ci(
    dados: pd.DataFrame,
    *,
    z_col: str,
    eta_col: str | None,
    risk_fn: Any,
    config: ConfigExperimento,
) -> dict[str, Any]:
    fns = {
        "coverage": cobertura_empirica,
        "risk_selective": risk_fn,
        "adequacy_useful": lambda x: _adequacy_useful(x, z_col),
        "precision_abstention": lambda x: _precision_abstention(x, z_col),
        "base_rate_inadequacy": lambda x: _base_rate_inadequacy(x, z_col),
        "lift_pp": lambda x: _lift_pp(x, z_col),
        "n_abstentions": _n_abstentions,
    }
    if eta_col is not None:
        fns["auroc"] = lambda x: _auroc(x, z_col, eta_col)
    return {
        "point": _summarize_points(dados, z_col=z_col, eta_col=eta_col, risk_fn=risk_fn),
        "ci": {
            nome: _bootstrap_metric(dados, fn, config=config)
            for nome, fn in fns.items()
        },
    }


def _resolver_run_dir(run: str | Path, base_dir: Path) -> Path:
    run_path = Path(run)
    if run_path.exists():
        return run_path
    return base_dir / run


def _carregar_z_lookup(
    run_dir: Path,
    by_agent: dict[str, pd.DataFrame],
    judge_model: str,
    c_min_grid: list[float],
) -> dict[tuple[str, str], int]:
    judge_dir = run_dir / "judge_labels" / judge_model.replace("/", "_")
    cache_path = judge_dir / "cache.parquet"
    if cache_path.exists():
        judge_df = pd.read_parquet(cache_path)
        z_lookup = {
            (str(row.item_id), str(row.response_source)): int(row.z)
            for row in judge_df.itertuples(index=False)
        }
    else:
        z_lookup = {}
        consolidado_ref = pd.read_parquet(
            run_dir / f"consolidated_test_full_cmin={c_min_grid[0]:.1f}.parquet"
        )
        for aid in by_agent:
            coluna = f"z_agent_{aid}"
            if coluna not in consolidado_ref.columns:
                continue
            for row in consolidado_ref[["item_id", coluna]].itertuples(index=False):
                z_lookup[(str(row[0]), aid)] = int(row[1])
    # Falhas de JSON não passam pelo juiz; contam como z_agent=0.
    for aid, table in by_agent.items():
        for row in table.itertuples():
            z_lookup.setdefault((str(row.Index), aid), 0)
    return z_lookup


def _carregar_bsolo_por_run(
    *,
    run_dir: Path,
    config: ConfigExperimento,
    root_seed: int,
) -> tuple[dict[tuple[str, float], pd.DataFrame], pd.DataFrame]:
    items = carregar_br_taxqa_r(
        config.dataset.resolver_questions_path(),
        tamanho_amostra=config.sample_size,
        semente_amostra=root_seed,
    )
    item_ids = [it.item_id for it in items]
    strat_labels = [1 if it.has_doc_ref else 0 for it in items]
    partitions, _ = criar_particoes(
        item_ids=item_ids,
        rotulos_estratificacao=strat_labels,
        n_splits=config.partition.n_splits,
        n_folds_val=config.partition.n_val_folds,
        seed=root_seed,
    )
    validar_cobertura_completa(partitions)

    by_agent = {
        agent.agent_id: pd.read_parquet(
            run_dir / "agent_responses" / agent.agent_id / "responses.parquet"
        ).set_index("item_id")
        for agent in config.agents
    }
    z_lookup = _carregar_z_lookup(
        run_dir,
        by_agent,
        config.judge.model,
        list(config.decision.c_min_grid),
    )
    has_doc_ref_lookup = {it.item_id: bool(it.has_doc_ref) for it in items}
    c_min_ref = config.decision.c_min_grid[0]
    full_07 = pd.read_parquet(run_dir / f"consolidated_test_full_cmin={c_min_ref:.1f}.parquet")
    retrieval_recall_lookup = {
        str(row.item_id): float(row.retrieval_recall)
        for row in full_07.itertuples(index=False)
    }

    artefatos_bsolo: dict[tuple[str, float], list[dict[str, Any]]] = {
        (agent.agent_id, c): [] for agent in config.agents for c in config.decision.c_min_grid
    }
    for p in partitions:
        calibradores = {}
        for aid, table in by_agent.items():
            fit_conf = []
            fit_z = []
            for iid in p.fit_ids:
                raw_conf = table.loc[iid, "confidence_raw"]
                conf = float(raw_conf) if pd.notna(raw_conf) else 0.0
                fit_conf.append(conf)
                fit_z.append(z_lookup[(iid, aid)])
            calibradores[aid] = ajustar_calibrador_isotonico(fit_conf, fit_z)
        for aid in by_agent:
            rows_por_cmin = bl.bsolo_agente_calibrado_com_abstencao(
                agent_id=aid,
                calibrador=calibradores[aid],
                by_agent=by_agent,
                val_ids=list(p.val_ids),
                test_ids=list(p.test_ids),
                z_lookup=z_lookup,
                has_doc_ref_lookup=has_doc_ref_lookup,
                retrieval_recall_lookup=retrieval_recall_lookup,
                fold_k=p.k,
                c_min_grid=list(config.decision.c_min_grid),
            )
            for c_min, rows in rows_por_cmin.items():
                artefatos_bsolo[(aid, c_min)].extend(rows)

    dataframes = {
        key: pd.DataFrame(rows)
        for key, rows in artefatos_bsolo.items()
    }
    item_level = pd.concat(
        [df for df in dataframes.values()],
        ignore_index=True,
    ).sort_values(["agent", "c_min", "fold_k", "item_id"], kind="stable")
    return dataframes, item_level


def gerar_avaliacao_bsolo_offline(
    *,
    run: str | Path,
    config_path: str | Path,
    base_dir: Path = Path("runs"),
    primary_c_min: float = 0.7,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    write_parquets: bool = True,
) -> ResultadoBSoloOffline:
    """Gera métricas e tabelas do baseline B_solo a partir de uma run existente."""
    config = carregar_config_experimento(config_path)
    run_dir = _resolver_run_dir(run, base_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run não encontrada: {run_dir}")

    manifesto = ler_manifesto(run_dir / "manifest.yaml")
    root_seed = int(manifesto.root_seed)
    bsolo_por_agente_cmin, item_level = _carregar_bsolo_por_run(
        run_dir=run_dir,
        config=config,
        root_seed=root_seed,
    )

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_json
        if output_json is not None
        else metrics_dir / f"bsolo_summary_cmin={primary_c_min:.1f}.json"
    )
    item_level_csv_path = (
        output_csv
        if output_csv is not None
        else metrics_dir / "bsolo_item_level.csv"
    )

    parquets_por_agente_cmin: dict[tuple[str, float], Path] = {}
    if write_parquets:
        for (agent_id, c_min), df in bsolo_por_agente_cmin.items():
            path = caminho_teste_consolidado(
                base_dir,
                run_dir.name,
                c_min=c_min,
                baseline="Bsolo",
                agent_id=agent_id,
            )
            escrever_parquet(df, path)
            parquets_por_agente_cmin[(agent_id, c_min)] = path

    summary: dict[str, Any] = {
        "run": run_dir.name,
        "config_path": str(Path(config_path)),
        "root_seed": root_seed,
        "primary_c_min": primary_c_min,
        "cmin_primary": {},
        "supplemental_points": {},
    }

    full_primary = _compactar_df_bootstrap(
        pd.read_parquet(run_dir / f"consolidated_test_full_cmin={primary_c_min:.1f}.parquet")[
            ["has_doc_ref", "abstained", "z_system", "z_winner", "eta"]
        ],
        bool_cols=["has_doc_ref", "abstained"],
        int_cols=["z_system", "z_winner"],
        float_cols=["eta"],
    )
    summary["cmin_primary"]["Full"] = _summarize_with_ci(
        full_primary,
        z_col="z_winner",
        eta_col="eta",
        risk_fn=risco_seletivo,
        config=config,
    )

    b1_df = pd.read_parquet(run_dir / "consolidated_test_B1.parquet").copy()
    b1_df["abstained"] = False
    summary["cmin_primary"]["B1"] = {
        "point": _summarize_points(
            b1_df[["abstained", "z_winner"]].copy(),
            z_col="z_winner",
            eta_col=None,
            risk_fn=lambda x: _risk_selective_from_col(x, "z_winner"),
        )
    }

    b3_df = pd.read_parquet(run_dir / "consolidated_test_B3.parquet").copy()
    b3_df["abstained"] = False
    summary["cmin_primary"]["B3"] = {
        "point": _summarize_points(
            b3_df[["abstained", "z_winner", "eta"]].copy(),
            z_col="z_winner",
            eta_col="eta",
            risk_fn=lambda x: _risk_selective_from_col(x, "z_winner"),
        )
    }

    for agent in config.agents:
        df_boot = _compactar_df_bootstrap(
            bsolo_por_agente_cmin[(agent.agent_id, primary_c_min)][
                ["has_doc_ref", "abstained", "z_agent", "eta_solo"]
            ],
            bool_cols=["has_doc_ref", "abstained"],
            int_cols=["z_agent"],
            float_cols=["eta_solo"],
        )
        summary["cmin_primary"][f"{agent.agent_id}_solo"] = _summarize_with_ci(
            df_boot,
            z_col="z_agent",
            eta_col="eta_solo",
            risk_fn=_risk_selective_bsolo,
            config=config,
        )

    for c_min in config.decision.c_min_grid:
        full_df = pd.read_parquet(run_dir / f"consolidated_test_full_cmin={c_min:.1f}.parquet")
        summary["supplemental_points"][f"full_cmin={c_min:.1f}"] = _summarize_points(
            full_df,
            z_col="z_winner",
            eta_col="eta",
            risk_fn=risco_seletivo,
        )
        for agent in config.agents:
            df = bsolo_por_agente_cmin[(agent.agent_id, c_min)]
            summary["supplemental_points"][f"{agent.agent_id}_solo_cmin={c_min:.1f}"] = (
                _summarize_points(
                    df,
                    z_col="z_agent",
                    eta_col="eta_solo",
                    risk_fn=_risk_selective_bsolo,
                )
            )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    item_level_csv_path.parent.mkdir(parents=True, exist_ok=True)
    item_level.to_csv(item_level_csv_path, index=False)

    return ResultadoBSoloOffline(
        run_dir=run_dir,
        summary_path=summary_path,
        item_level_csv_path=item_level_csv_path,
        parquets_por_agente_cmin=parquets_por_agente_cmin,
    )
