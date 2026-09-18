"""Geração reprodutível das figuras."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado
from ensemble_llm.avaliacao.diversidade import fracao_erro_todos_agentes
from ensemble_llm.avaliacao.metricas import (
    dados_diagrama_confiabilidade,
    erro_calibracao_esperado,
)

logger = logging.getLogger(__name__)

COR_AZUL = "#1f4e79"
COR_AZUL_CLARO = "#5b8db8"
COR_CINZA = "#7a8793"
COR_VERMELHO = "#b03a2e"
COR_LARANJA = "#d97706"
COR_VERDE = "#2e7d32"
COR_TEXTO = "#243447"
COR_FAIXA = "#f4f6f8"
CORES_ESTRATOS = {
    "sem_doc_ref": "#8da0cb",
    "recall=0": "#fc8d62",
    "0<recall<0.5": "#ffd166",
    "recall>=0.5": "#66c2a5",
}
ORDEM_ESTRATOS = ["sem_doc_ref", "recall=0", "0<recall<0.5", "recall>=0.5"]
ROTULOS_ESTRATOS = {
    "sem_doc_ref": "sem\ndoc ref",
    "recall=0": "recall\n= 0",
    "0<recall<0.5": "0 < recall\n< 0.5",
    "recall>=0.5": "recall\n≥ 0.5",
}

# Rótulos visíveis das figuras, por idioma. As figuras NÃO carregam título nem número
# interno: a legenda (e a numeração) vivem no manuscrito, o que elimina por construção
# a inconsistência entre numeração interna e externa.
TEXTOS: dict[str, dict[str, str]] = {
    "pt": {
        "risco_sem_abstencao": "Risco sem abstenção",
        "cobertura": "Cobertura",
        "risco_seletivo": "Risco seletivo",
        "pontos_cmin": r"Pontos = escolhas operacionais de $C_{\min}$",
        "bruto": "Bruta",
        "calibrado": "Calibrada",
        "ece_caixa": "ECE bruto = {raw:.3f}\nECE calib. = {cal:.3f}",
        "confianca": "Confiança",
        "fracao_adequacao": "Fração empírica de adequação",
        "rodape_fold": (
            "Fold representativo k={fold:02d}, escolhido por proximidade à mediana "
            "do ECE bruto médio entre folds (mediana = {mediana:.3f})."
        ),
        "rodape_barras": (
            "Barras claras: frequência relativa por intervalo, normalizada pelo "
            "intervalo mais frequente de cada painel."
        ),
        "itens_inadequados": "Itens inadequados (z=0)",
        "itens_adequados": "Itens adequados (z=1)",
        "densidade": "Densidade",
        "zona_abstencao": "Região sombreada = zona de abstenção",
        "suf_insuficiente": "insuficiente",
        "suf_parcial": "parcial",
        "suf_suficiente": "suficiente",
        "adequacao_vencedor": "Adequação do vencedor (z)",
        "cobertura_fracao": "Cobertura (fração respondida)",
        "suficiencia_xlabel": "Suficiência do contexto recuperado (juiz LLM)",
        "adequacao_ylabel": "Adequação (z do vencedor)",
        "percentual": "Percentual",
        "rodape_suficiencia": (
            "Suficiência julgada por LLM a partir do contexto recuperado, da questão e "
            "da resposta de referência. "
            "IC95% percentil (B={b}), estratificado por presença de referência documental."
        ),
        "precisao_abstencao": "Precisão da abstenção",
        "precisao": "Precisão",
        "volume_abstencoes": "Volume de abstenções",
        "n_abstencoes": "Nº de abstenções",
        "agente_i": "Agente i",
        "agente_j": "Agente j",
        "cbar_diversidade": "Off-diagonal: P(j erra | i errou)\nDiagonal: taxa marginal de erro",
        "rodape_diversidade": (
            "Todos erram juntos / esperado sob independência = {ratio:.3f}"
        ),
    },
    "en": {
        "risco_sem_abstencao": "Risk without abstention (B3)",
        "cobertura": "Coverage",
        "risco_seletivo": "Selective risk",
        "pontos_cmin": r"Points = operational choices of $C_{\min}$",
        "bruto": "Raw",
        "calibrado": "Calibrated",
        "ece_caixa": "Raw ECE = {raw:.3f}\nCalib. ECE = {cal:.3f}",
        "confianca": "Confidence",
        "fracao_adequacao": "Empirical adequacy fraction",
        "rodape_fold": (
            "Representative fold k={fold:02d}, chosen for proximity to the median "
            "of the mean raw ECE across folds (median = {mediana:.3f})."
        ),
        "rodape_barras": (
            "Light bars: relative frequency by bin, normalized by the most frequent "
            "bin in each panel."
        ),
        "itens_inadequados": "Inadequate items (z=0)",
        "itens_adequados": "Adequate items (z=1)",
        "densidade": "Density",
        "zona_abstencao": "Shaded region = abstention zone",
        "suf_insuficiente": "insufficient",
        "suf_parcial": "partial",
        "suf_suficiente": "sufficient",
        "adequacao_vencedor": "Winner adequacy (z)",
        "cobertura_fracao": "Coverage (fraction answered)",
        "suficiencia_xlabel": "Sufficiency of the retrieved context (LLM judge)",
        "adequacao_ylabel": "Adequacy (winner's z)",
        "percentual": "Percentage",
        "rodape_suficiencia": (
            "Sufficiency judged by an LLM from the retrieved context, the question, and "
            "the reference answer. 95% percentile CIs (B={b}), stratified by presence "
            "of a documentary reference."
        ),
        "precisao_abstencao": "Abstention precision",
        "precisao": "Precision",
        "volume_abstencoes": "Abstention volume",
        "n_abstencoes": "Number of abstentions",
        "agente_i": "Agent i",
        "agente_j": "Agent j",
        "cbar_diversidade": "Off-diagonal: P(j errs | i errs)\nDiagonal: marginal error rate",
        "rodape_diversidade": (
            "All err jointly / expected under independence = {ratio:.3f}"
        ),
    },
}


def _t(lang: str, chave: str, **fmt: Any) -> str:
    """Resolve um rótulo do dicionário de textos, com formatação opcional."""
    texto = TEXTOS[lang][chave]
    return texto.format(**fmt) if fmt else texto


@dataclass(frozen=True)
class ContextoRun:
    """Artefatos necessários para geração das figuras de um run."""

    base_dir: Path
    experiment_id: str
    run_dir: Path
    output_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    curve: pd.DataFrame
    tau_points: pd.DataFrame
    cmins: list[float]
    cmin_ref: float
    consolidated_by_cmin: dict[float, pd.DataFrame]
    baseline_b3: pd.DataFrame
    agent_ids: list[str]
    agent_labels: dict[str, str]
    n_items: int
    suficiencia: pd.DataFrame | None

    @classmethod
    def carregar(
        cls,
        base_dir: Path,
        experiment_id: str,
        *,
        output_dir: Path | None = None,
        cmin_ref: float | None = None,
    ) -> ContextoRun:
        """Carrega os artefatos persistidos de um run."""
        run_dir = base_dir / experiment_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run não encontrado: {run_dir}")

        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
        curve = pd.read_parquet(run_dir / "metrics" / "risk_coverage_curve.parquet")
        tau_points = pd.read_parquet(run_dir / "metrics" / "tau_star_points.parquet")
        baseline_b3 = pd.read_parquet(run_dir / "consolidated_test_B3.parquet")

        consolidated_by_cmin: dict[float, pd.DataFrame] = {}
        cmins: list[float] = []
        for path in sorted(run_dir.glob("consolidated_test_full_cmin=*.parquet")):
            valor = float(path.stem.split("cmin=")[1])
            cmins.append(valor)
            consolidated_by_cmin[valor] = pd.read_parquet(path)
        if not cmins:
            raise ValueError(f"Nenhum consolidated_test_full_cmin=*.parquet em {run_dir}")

        cmin_escolhido = resolver_cmin_referencia(cmins, cmin_ref)
        df_ref = consolidated_by_cmin[cmin_escolhido]
        agent_ids = descobrir_agentes(df_ref)
        manifest_agents = manifest.get("agents", {})
        agent_labels = {
            aid: formatar_rotulo_modelo(str(manifest_agents.get(aid, aid)))
            for aid in agent_ids
        }

        suf_path = run_dir / "metrics" / "suficiencia_contexto.parquet"
        suficiencia = pd.read_parquet(suf_path) if suf_path.exists() else None

        out_dir = output_dir or run_dir / "figures" / "article_v3"
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            base_dir=base_dir,
            experiment_id=experiment_id,
            run_dir=run_dir,
            output_dir=out_dir,
            manifest=manifest,
            summary=summary,
            curve=curve,
            tau_points=tau_points,
            cmins=sorted(cmins),
            cmin_ref=cmin_escolhido,
            consolidated_by_cmin=consolidated_by_cmin,
            baseline_b3=baseline_b3,
            agent_ids=agent_ids,
            agent_labels=agent_labels,
            n_items=len(df_ref),
            suficiencia=suficiencia,
        )


@dataclass(frozen=True)
class ResultadoFigura:
    """Metadados de saída de uma figura gerada."""

    nome: str
    caminhos: list[Path]


def aplicar_estilo_artigo() -> None:
    """Aplica um estilo gráfico sóbrio para o artigo."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "grid.color": "#d0d7de",
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "savefig.dpi": 300,
        }
    )


def descobrir_agentes(df: pd.DataFrame) -> list[str]:
    """Extrai IDs de agentes a partir das colunas de confiança bruta."""
    prefixo = "confidence_raw_"
    agentes = sorted(col[len(prefixo) :] for col in df.columns if col.startswith(prefixo))
    if not agentes:
        raise ValueError("Nenhuma coluna confidence_raw_<aid> encontrada.")
    return agentes


def resolver_cmin_referencia(cmins_disponiveis: Sequence[float], cmin_ref: float | None) -> float:
    """Escolhe o C_min de referência para figuras que usam um consolidado específico."""
    if cmin_ref is not None:
        if cmin_ref not in cmins_disponiveis:
            raise ValueError(
                f"cmin_ref={cmin_ref} não encontrado; disponíveis: {list(cmins_disponiveis)}"
            )
        return cmin_ref
    if 0.7 in cmins_disponiveis:
        return 0.7
    return sorted(cmins_disponiveis)[0]


def formatar_rotulo_modelo(slug: str) -> str:
    """Normaliza o slug do modelo para uso em rótulos curtos."""
    valor = slug.split("/")[-1]
    rotulos = {
        "gemma4-e4b": "Gemma 4 E4B",
        "granite4.1-8b": "Granite 4.1 8B",
        "qwen3.5-9b": "Qwen 3.5 9B",
    }
    return rotulos.get(valor, valor.replace("-", " "))


def classificar_estrato_retrieval(has_doc_ref: bool, retrieval_recall: float) -> str:
    """Classifica o item no estrato operacional de recuperação."""
    if not has_doc_ref:
        return "sem_doc_ref"
    if pd.isna(retrieval_recall) or retrieval_recall == 0:
        return "recall=0"
    if retrieval_recall < 0.5:
        return "0<recall<0.5"
    return "recall>=0.5"


def resumir_estratos_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas operacionais por estrato de recuperação."""
    dados = df.copy()
    dados["estrato_retrieval"] = [
        classificar_estrato_retrieval(bool(has_ref), float(recall))
        for has_ref, recall in zip(dados["has_doc_ref"], dados["retrieval_recall"], strict=True)
    ]

    linhas: list[dict[str, float | str | int]] = []
    for estrato in ORDEM_ESTRATOS:
        sub = dados[dados["estrato_retrieval"] == estrato]
        if sub.empty:
            continue
        responded = sub[~sub["abstained"]]
        abstained = sub[sub["abstained"]]
        linhas.append(
            {
                "estrato": estrato,
                "n": int(len(sub)),
                "taxa_abstencao": float(sub["abstained"].mean()),
                "precisao_abstencao": (
                    float((abstained["z_winner"] == 0).mean())
                    if not abstained.empty
                    else float("nan")
                ),
                "risco_seletivo": (
                    float((responded["z_winner"] == 0).mean())
                    if not responded.empty
                    else float("nan")
                ),
                "adequacao_respondidas": (
                    float(responded["z_system"].mean()) if not responded.empty else float("nan")
                ),
            }
        )
    return pd.DataFrame(linhas)


def selecionar_fold_representativo(
    df: pd.DataFrame,
    agent_ids: Sequence[str],
    *,
    n_bins: int = 15,
) -> tuple[int, pd.DataFrame]:
    """Escolhe o fold cujo ECE bruto médio é mais próximo da mediana entre folds."""
    linhas: list[dict[str, float | int]] = []
    for fold_k, sub in df.groupby("fold_k"):
        linha: dict[str, float | int] = {"fold_k": int(fold_k)}
        eces: list[float] = []
        for aid in agent_ids:
            ece = erro_calibracao_esperado(
                sub[f"confidence_raw_{aid}"].to_numpy(),
                sub[f"z_agent_{aid}"].to_numpy(),
                n_bins=n_bins,
            )
            linha[f"ece_raw_{aid}"] = ece
            eces.append(ece)
        linha["ece_raw_mean"] = float(np.mean(eces))
        linhas.append(linha)

    stats = pd.DataFrame(linhas).sort_values("fold_k").reset_index(drop=True)
    mediana = float(stats["ece_raw_mean"].median())
    stats["dist_to_median"] = (stats["ece_raw_mean"] - mediana).abs()
    escolhido = int(stats.sort_values(["dist_to_median", "fold_k"]).iloc[0]["fold_k"])
    return escolhido, stats


def salvar_figura(
    fig: matplotlib.figure.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
) -> list[Path]:
    """Salva uma figura em múltiplos formatos."""
    caminhos: list[Path] = []
    for fmt in formats:
        caminho = output_dir / f"{stem}.{fmt}"
        fig.savefig(caminho)
        caminhos.append(caminho)
    plt.close(fig)
    return caminhos


def _ponto_curva_em_tau(df_curva: pd.DataFrame, tau: float) -> pd.Series:
    """Encontra o ponto da curva com tau mais próximo do solicitado."""
    idx = (df_curva["tau"] - tau).abs().idxmin()
    return df_curva.loc[idx]


def _configurar_percentual(ax: plt.Axes, *, ymax: float | None = None) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if ymax is not None:
        ax.set_ylim(0, ymax)


def gerar_figura_risco_cobertura(
    contexto: ContextoRun,
    *,
    comparacao: ContextoRun | None,
    formats: Iterable[str],
    lang: str = "pt",
    main_label: str | None = None,
    comparison_label: str | None = None,
) -> ResultadoFigura:
    """Gera a curva risco-cobertura com comparação opcional."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    curva_main = contexto.curve[contexto.curve["estrato"] == "all"]
    curva_main = curva_main.dropna(subset=["risco_seletivo"])
    curva_main = curva_main.sort_values("cobertura")
    ax.plot(
        curva_main["cobertura"],
        curva_main["risco_seletivo"],
        color=COR_AZUL,
        linewidth=2.2,
        label=main_label or f"{contexto.experiment_id} (N={contexto.n_items})",
    )

    if comparacao is not None:
        curva_cmp = comparacao.curve[comparacao.curve["estrato"] == "all"].dropna(
            subset=["risco_seletivo"]
        )
        curva_cmp = curva_cmp.sort_values("cobertura")
        ax.plot(
            curva_cmp["cobertura"],
            curva_cmp["risco_seletivo"],
            color=COR_CINZA,
            linewidth=1.8,
            linestyle="--",
            label=comparison_label or f"{comparacao.experiment_id} (N={comparacao.n_items})",
        )

    risco_b3 = float((contexto.baseline_b3["z_winner"] == 0).mean())
    ax.axhline(
        risco_b3,
        color=COR_LARANJA,
        linestyle=":",
        linewidth=1.8,
        label=_t(lang, "risco_sem_abstencao"),
    )

    deslocamentos_por_cmin = {
        0.5: (8, 16),
        0.7: (10, -24),
        0.8: (10, 16),
        0.9: (10, -24),
    }
    for row in contexto.tau_points.sort_values("c_min").itertuples(index=False):
        cmin = float(row.c_min)
        decisoes = contexto.consolidated_by_cmin[cmin]
        respondidas = ~decisoes["abstained"].astype(bool)
        cobertura_ponto = float(respondidas.mean())
        risco_ponto = float((decisoes.loc[respondidas, "z_winner"] == 0).mean())
        ax.scatter(
            [cobertura_ponto],
            [risco_ponto],
            color="black",
            s=50,
            zorder=5,
            edgecolor="white",
            linewidth=0.9,
        )
        dx, dy = deslocamentos_por_cmin.get(round(cmin, 1), (8, 12))
        ax.annotate(
            rf"$C_{{\min}}={cmin:.1f}$",
            xy=(cobertura_ponto, risco_ponto),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=COR_TEXTO,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(_t(lang, "cobertura"))
    ax.set_ylabel(_t(lang, "risco_seletivo"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.text(
        0.99,
        0.02,
        _t(lang, "pontos_cmin"),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=COR_CINZA,
    )
    fig.tight_layout()
    return ResultadoFigura(
        nome="f1_risk_coverage",
        caminhos=salvar_figura(fig, contexto.output_dir, "f1_risk_coverage", formats),
    )


def gerar_figura_reliability(
    contexto: ContextoRun,
    *,
    formats: Iterable[str],
    lang: str = "pt",
) -> ResultadoFigura:
    """Gera os reliability diagrams antes/depois da calibração."""
    df_ref = contexto.consolidated_by_cmin[contexto.cmin_ref]
    fold_k, stats = selecionar_fold_representativo(df_ref, contexto.agent_ids)
    fold_df = df_ref[df_ref["fold_k"] == fold_k].copy()

    fig, axes = plt.subplots(
        1,
        len(contexto.agent_ids),
        figsize=(4.4 * len(contexto.agent_ids), 5.2),
    )
    if len(contexto.agent_ids) == 1:
        axes = np.array([axes])

    for ax, aid in zip(axes, contexto.agent_ids, strict=True):
        ax_hist = ax.twinx()
        rotulo = contexto.agent_labels[aid]
        bins_raw = dados_diagrama_confiabilidade(
            fold_df[f"confidence_raw_{aid}"].to_numpy(),
            fold_df[f"z_agent_{aid}"].to_numpy(),
            n_bins=15,
        )
        bins_cal = dados_diagrama_confiabilidade(
            fold_df[f"confidence_calib_{aid}"].to_numpy(),
            fold_df[f"z_agent_{aid}"].to_numpy(),
            n_bins=15,
        )

        centros = (bins_raw["bin_lo"] + bins_raw["bin_hi"]) / 2
        largura = float((bins_raw["bin_hi"] - bins_raw["bin_lo"]).iloc[0] * 0.45)
        escala_hist = max(float(bins_raw["count"].max()), float(bins_cal["count"].max()), 1.0)
        ax_hist.bar(
            centros - largura / 2,
            bins_raw["count"] / escala_hist,
            width=largura,
            color=COR_VERMELHO,
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )
        ax_hist.bar(
            centros + largura / 2,
            bins_cal["count"] / escala_hist,
            width=largura,
            color=COR_AZUL,
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )
        ax_hist.set_ylim(0, 1.05)
        ax_hist.set_yticks([])
        ax_hist.grid(False)

        raw_valid = bins_raw.dropna(subset=["avg_confidence", "accuracy"])
        cal_valid = bins_cal.dropna(subset=["avg_confidence", "accuracy"])
        ax.plot([0, 1], [0, 1], linestyle="--", color=COR_CINZA, linewidth=1.0)
        ax.plot(
            raw_valid["avg_confidence"],
            raw_valid["accuracy"],
            color=COR_VERMELHO,
            marker="o",
            linewidth=1.5,
            markersize=4.2,
            label=_t(lang, "bruto"),
        )
        ax.plot(
            cal_valid["avg_confidence"],
            cal_valid["accuracy"],
            color=COR_AZUL,
            marker="o",
            linewidth=1.5,
            markersize=4.2,
            label=_t(lang, "calibrado"),
        )

        ece_raw = erro_calibracao_esperado(
            fold_df[f"confidence_raw_{aid}"].to_numpy(),
            fold_df[f"z_agent_{aid}"].to_numpy(),
        )
        ece_cal = erro_calibracao_esperado(
            fold_df[f"confidence_calib_{aid}"].to_numpy(),
            fold_df[f"z_agent_{aid}"].to_numpy(),
        )
        ece_texto = _t(lang, "ece_caixa", raw=ece_raw, cal=ece_cal)
        if lang == "pt":
            ece_texto = ece_texto.replace(".", ",")
        ax.text(
            0.98,
            0.03,
            ece_texto,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de"},
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if lang == "pt":
            decimal_pt = FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ","))
            ax.xaxis.set_major_formatter(decimal_pt)
            ax.yaxis.set_major_formatter(decimal_pt)
        ax.set_title(rotulo)
        ax.set_xlabel(_t(lang, "confianca"))
        if aid == contexto.agent_ids[0]:
            ax.set_ylabel(_t(lang, "fracao_adequacao"))
        ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    mediana_ece = float(stats["ece_raw_mean"].median())
    rodape_fold = _t(
        lang,
        "rodape_fold",
        fold=fold_k,
        mediana=mediana_ece,
    )
    if lang == "pt":
        rodape_fold = rodape_fold.replace(
            f"{mediana_ece:.3f}", f"{mediana_ece:.3f}".replace(".", ",")
        )
    fig.text(
        0.5,
        0.035,
        rodape_fold,
        ha="center",
        fontsize=8.5,
        color=COR_TEXTO,
    )
    fig.text(
        0.5,
        0.012,
        _t(lang, "rodape_barras"),
        ha="center",
        fontsize=8.5,
        color=COR_TEXTO,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return ResultadoFigura(
        nome="f2_reliability",
        caminhos=salvar_figura(fig, contexto.output_dir, "f2_reliability", formats),
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#4a5568",
    fontsize: int = 10,
    pad: float = 0.02,
    linestyle: str = "-",
) -> None:
    # `pad` expande a caixa em `pad` para CADA lado: com pad=0.02 e vãos de 0.02 entre
    # caixas, elas encostam e a fila vira uma barra contínua. Quem posicionar caixas
    # justas deve reduzir o pad.
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size=0.015",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COR_TEXTO,
        wrap=True,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linestyle: str = "-",
    color: str = COR_TEXTO,
) -> None:
    seta = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        linestyle=linestyle,
        color=color,
    )
    ax.add_patch(seta)


TEXTOS_ARQUITETURA: dict[str, dict[str, str]] = {
    "pt": {
        "faixa1": (
            "1 | AJUSTE E AVALIAÇÃO — offline, com rótulos do instrumento de adequação."
        ),
        "juiz": (
            "Instrumento de adequação\n(offline)\n"
            r"$z \in \{0,1\}$" "  de cada resposta"
        ),
        "dfit": r"$\mathcal{D}_{\mathrm{fit}}$" "\npesos " r"$w_i$" "   calibradores " r"$f_i$",
        "dval": r"$\mathcal{D}_{\mathrm{val}}$" "\nlimiar " r"$\tau^*$" "\n(curva risco–cobertura)",
        "dtest": (
            r"$\mathcal{D}_{\mathrm{test}}$"
            "\navaliação fora da amostra\n(rótulos apenas offline)"
        ),
        "artefatos": "artefatos persistidos  ↓",
        "faixa2": "2 | INFERÊNCIA  —  online, apenas com sinais endógenos.",
        "questao": "Questão " r"$q$",
        "rag": "RAG híbrido\n(denso + BM25)",
        "agentes": (
            r"$N=3$" " agentes\nmesmo contexto\nresposta " r"$r_i$" " + confiança " r"$c_i$"
        ),
        "selecao": "SELEÇÃO\n" r"$i^* = \arg\max_i(\cdot)$",
        "agregacao": "AGREGAÇÃO\n" r"$\eta = 1 - (\cdot)$",
        "duas_operacoes": (
            "A seleção escolhe a resposta candidata; a agregação produz a incerteza.\n"
            "A0 usa máximo ponderado calibrado; A5 usa média das confianças brutas."
        ),
        "liberar": "LIBERAR RESPOSTA?\n" r"$\eta \leq \tau^*$",
        "responder": "Responder\n" r"$r_{i^*}$",
        "abster": "Abster\nrevisão humana futura",
        "nao": "não",
        "sim": "sim",
        "wi_fi_decisao": r"$w_i, f_i$" " → decisão apenas em A0",
        "wi_fi_minima": r"A5: $w_i$ diagnóstico · $f_i$ reporte",
        "cabeca": (
            "Cabeça de decisão  " r"$(\cdot)$" " — chave " r"$\mathtt{decision.method}$" "\n"
            r"A0: $w_i \cdot \hat{c}_i$; $\eta = 1 - \max_i w_i \hat{c}_i$"
            "\n" r"A5: $c_i$ bruta; $\eta = 1 - \mathrm{média}_i\, c_i$"
        ),
    },
    "en": {
        "faixa1": (
            "1 | FITTING AND EVALUATION — offline, with labels from the adequacy instrument."
        ),
        "juiz": (
            "Frontier judge (offline)\nlabels the adequacy\n"
            r"$z \in \{0,1\}$" "  of each answer"
        ),
        "dfit": r"$\mathcal{D}_{\mathrm{fit}}$" "\nweights " r"$w_i$" "   calibrators " r"$f_i$",
        "dval": (
            r"$\mathcal{D}_{\mathrm{val}}$" "\nthreshold " r"$\tau^*$"
            "\n(risk–coverage curve)"
        ),
        "dtest": (
            r"$\mathcal{D}_{\mathrm{test}}$"
            "\nout-of-sample evaluation\n(labels offline only)"
        ),
        "artefatos": "persisted artefacts  ↓",
        "faixa2": "2 | INFERENCE  —  online, endogenous signals only.",
        "questao": "Question " r"$q$",
        "rag": "Hybrid RAG\n(dense + BM25)",
        "agentes": (
            r"$N=3$" " agents\nsame context\nanswer " r"$r_i$" " + confidence " r"$c_i$"
        ),
        "selecao": "SELECTION\n" r"$i^* = \arg\max_i(\cdot)$",
        "agregacao": "AGGREGATION\n" r"$\eta = 1 - (\cdot)$",
        "duas_operacoes": (
            "Selection chooses the candidate answer; aggregation produces uncertainty.\n"
            "A0 uses the calibrated weighted maximum; A5 uses mean raw confidence."
        ),
        "liberar": "RELEASE ANSWER?\n" r"$\eta \leq \tau^*$",
        "responder": "Answer\n" r"$r_{i^*}$",
        "abster": "Abstain\nfuture human review",
        "nao": "no",
        "sim": "yes",
        "wi_fi_decisao": r"$w_i, f_i$" " → decision only in A0",
        "wi_fi_minima": r"A5: $w_i$ diagnostic · $f_i$ reporting",
        "cabeca": (
            "Decision head  " r"$(\cdot)$" " — key " r"$\mathtt{decision.method}$" "\n"
            r"A0: $w_i \cdot \hat{c}_i$; $\eta = 1 - \max_i w_i \hat{c}_i$"
            "\n" r"A5: raw $c_i$; $\eta = 1 - \mathrm{mean}_i\, c_i$"
        ),
    },
}


def gerar_figura_arquitetura(
    contexto: ContextoRun,
    *,
    formats: Iterable[str],
    lang: str = "pt",
) -> ResultadoFigura:
    """Diagrama da arquitetura, em duas faixas temporais.

    A figura precisa deixar três coisas visíveis, porque o texto depende delas:

    1. **Os dois regimes.** O ajuste é offline e é o único lugar onde o juiz atua; a
       inferência é online e opera apenas com sinais endógenos. Desenhá-los na mesma
       faixa faz o juiz parecer parte do caminho de decisão.
    2. **As duas operações.** As N confianças alimentam operações de naturezas distintas:
       o ``argmax`` **seleciona** a resposta, a média **agrega** a incerteza. É a distinção
       que sustenta a leitura teórica — teoremas de ensemble governam a segunda e não a
       primeira.
    3. **A cabeça como slot.** ``decision.method`` é uma bifurcação única: a arquitetura
       proposta e a cabeça minimalista diferem em uma caixa, e o resto do pipeline é
       idêntico. Mostrar as duas variantes evita antecipar o resultado da avaliação.

    Convenção de traço: linha cheia escura = fluxo de dados; linha cinza fina = parâmetro
    alimentando o caminho de decisão; tracejado = só vale para a variante proposta.

    O que atravessa para a inferência é apenas $\\tau^*$ (parâmetro da política) e as
    respostas dos agentes. Os pesos $w_i$ e os calibradores $f_i$ só entram na *decisão* na
    variante proposta; na cabeça mínima não decidem — $w_i$ sobrevive como diagnóstico (a
    razão entre adequações) e $f_i$ como confiança reportada ao operador humano. A figura
    marca essa distinção para não sugerir que os dois mecanismos refutados decidem.
    """
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cinza = "#8a94a6"
    txt = TEXTOS_ARQUITETURA[lang]

    # ------------------------------------------------------------------ faixa 1: ajuste
    ax.text(
        0.01, 0.955,
        txt["faixa1"],
        fontsize=10.5, fontweight="bold", color=COR_TEXTO,
    )
    ya, ha = 0.745, 0.145
    _box(
        ax, (0.02, ya), 0.18, ha,
        txt["juiz"],
        facecolor="#fcebea", fontsize=8.5, pad=0.008,
    )
    _box(
        ax, (0.26, ya), 0.18, ha,
        txt["dfit"],
        facecolor="#fdf3e7", fontsize=8.5, pad=0.008,
    )
    _box(
        ax, (0.51, ya), 0.18, ha,
        txt["dval"],
        facecolor="#fbe6cf", fontsize=8.5, pad=0.008,
    )
    _box(
        ax, (0.76, ya), 0.21, ha,
        txt["dtest"],
        facecolor="#f3edf8", fontsize=8.5, pad=0.008,
    )
    _arrow(ax, (0.205, ya + ha / 2), (0.255, ya + ha / 2))
    _arrow(ax, (0.445, ya + ha / 2), (0.505, ya + ha / 2))
    _arrow(ax, (0.695, ya + ha / 2), (0.755, ya + ha / 2))

    # divisória entre os regimes
    ax.plot([0.01, 0.99], [0.665, 0.665], linestyle=(0, (6, 4)), color=cinza, linewidth=1.1)
    ax.text(0.99, 0.678, txt["artefatos"], fontsize=8.5, color=cinza, ha="right")

    # --------------------------------------------------------------- faixa 2: inferência
    ax.text(
        0.01, 0.628,
        txt["faixa2"],
        fontsize=10.5, fontweight="bold", color=COR_TEXTO,
    )
    yb, hb = 0.335, 0.135
    _box(ax, (0.02, yb), 0.10, hb, txt["questao"], facecolor="#e8f1f8", pad=0.008)
    _box(
        ax, (0.15, yb), 0.13, hb, txt["rag"],
        facecolor="#f0ecf8", fontsize=9, pad=0.008,
    )
    _box(
        ax, (0.31, yb), 0.16, hb,
        txt["agentes"],
        facecolor="#eef5ec", fontsize=8.5, pad=0.008,
    )
    _arrow(ax, (0.128, yb + hb / 2), (0.145, yb + hb / 2))
    _arrow(ax, (0.288, yb + hb / 2), (0.305, yb + hb / 2))

    # As confianças abrem em DUAS operações de naturezas distintas.
    _box(
        ax, (0.50, 0.475), 0.15, 0.115,
        txt["selecao"],
        facecolor="#eef5ec", fontsize=9.5, pad=0.008,
    )
    _box(
        ax, (0.50, 0.175), 0.15, 0.115,
        txt["agregacao"],
        facecolor="#fff7d6", fontsize=9.5, pad=0.008,
    )
    _arrow(ax, (0.478, 0.440), (0.495, 0.515))
    _arrow(ax, (0.478, 0.365), (0.495, 0.250))
    ax.text(
        0.02, 0.245,
        txt["duas_operacoes"],
        fontsize=8.5, color=cinza, style="italic",
    )

    _box(
        ax, (0.70, 0.335), 0.12, 0.12, txt["liberar"],
        facecolor="#e4f4ea", fontsize=9, pad=0.008,
    )
    _arrow(ax, (0.658, 0.5325), (0.695, 0.420))
    _arrow(ax, (0.658, 0.2325), (0.695, 0.370))

    _box(
        ax, (0.85, 0.455), 0.135, 0.155, txt["responder"],
        facecolor="#eef5ec", fontsize=9.5, pad=0.008,
    )
    _box(
        ax, (0.85, 0.035), 0.135, 0.135, txt["abster"],
        facecolor="#fde8e8", fontsize=8.5, pad=0.008, linestyle=(0, (4, 3)),
    )
    # Seleção e agregação convergem na política; só ela libera a resposta.
    _arrow(ax, (0.828, 0.415), (0.843, 0.500))
    _arrow(ax, (0.828, 0.365), (0.843, 0.140), linestyle=(0, (4, 3)))
    caixa = {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
    ax.text(0.842, 0.432, txt["sim"], fontsize=9, color=COR_TEXTO,
            ha="center", va="center", bbox=caixa)
    ax.text(0.842, 0.250, txt["nao"], fontsize=9, color=COR_TEXTO,
            ha="center", va="center", bbox=caixa)

    # --------------------------------- artefatos do ajuste descendo para a inferência
    _arrow(ax, (0.60, ya - 0.010), (0.76, 0.465), color=cinza)
    ax.text(0.625, 0.690, r"$\tau^*$", fontsize=10, color=cinza)
    # w_i e f_i só entram NA DECISÃO na variante proposta; na cabeça mínima não decidem —
    # w_i sobrevive como diagnóstico e f_i como confiança reportada ao operador.
    _arrow(ax, (0.35, ya - 0.010), (0.52, 0.600), linestyle=(0, (4, 3)), color=cinza)
    ax.text(0.36, 0.700, txt["wi_fi_decisao"],
            fontsize=8.5, color=cinza)
    ax.text(0.36, 0.678, txt["wi_fi_minima"],
            fontsize=7.8, color=cinza)

    # ------------------------------------------------------ a cabeça de decisão é um slot
    _box(
        ax, (0.02, 0.035), 0.42, 0.135,
        txt["cabeca"],
        facecolor="#f4f6f8", fontsize=9, pad=0.008, linestyle=(0, (4, 3)),
    )

    fig.tight_layout()
    return ResultadoFigura(
        nome="f3_architecture",
        caminhos=salvar_figura(fig, contexto.output_dir, "f3_architecture", formats),
    )


def gerar_figura_eta(
    contexto: ContextoRun,
    *,
    cmin_principal: float,
    cmin_secundario: float | None,
    formats: Iterable[str],
    lang: str = "pt",
) -> ResultadoFigura:
    """Gera a distribuição de η(q) por adequação."""
    df = contexto.consolidated_by_cmin[contexto.cmin_ref]
    eta_inadequado = df.loc[df["z_winner"] == 0, "eta"]
    eta_adequado = df.loc[df["z_winner"] == 1, "eta"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    # Range adaptativo: η depende da cabeça de decisão (weighted_calibrated ~[0.7,1.0];
    # mean_raw = 1 − média das confianças cruas, tipicamente ~[0, 0.5]).
    todos = df["eta"].to_numpy()
    pad = 0.02
    x_lo = max(0.0, float(np.floor(todos.min() * 20) / 20) - pad)
    x_hi = min(1.0, float(np.ceil(todos.max() * 20) / 20) + pad)
    bins = np.linspace(x_lo, x_hi, 22)
    ax.hist(
        eta_inadequado,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COR_VERMELHO,
        label=_t(lang, "itens_inadequados"),
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        eta_adequado,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COR_AZUL,
        label=_t(lang, "itens_adequados"),
        edgecolor="white",
        linewidth=0.4,
    )

    tau_lookup = {
        float(row.c_min): float(row.tau_star_mean)
        for row in contexto.tau_points.itertuples(index=False)
    }
    tau_principal = tau_lookup[cmin_principal]
    ax.axvspan(tau_principal, x_hi, color=COR_LARANJA, alpha=0.08)
    ax.axvline(
        tau_principal,
        color=COR_LARANJA,
        linestyle="--",
        linewidth=1.8,
        label=fr"$\tau^*$ para $C_{{min}}={cmin_principal:.1f}$",
    )
    if cmin_secundario is not None and cmin_secundario in tau_lookup:
        ax.axvline(
            tau_lookup[cmin_secundario],
            color=COR_VERDE,
            linestyle=":",
            linewidth=1.8,
            label=fr"$\tau^*$ para $C_{{min}}={cmin_secundario:.1f}$",
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"$\eta(q)$")
    ax.set_ylabel(_t(lang, "densidade"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.text(
        0.98,
        0.92,
        _t(lang, "zona_abstencao"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COR_CINZA,
    )
    fig.tight_layout()
    return ResultadoFigura(
        nome="f4_eta_distribution",
        caminhos=salvar_figura(fig, contexto.output_dir, "f4_eta_distribution", formats),
    )


def gerar_figura_suficiencia(
    contexto: ContextoRun,
    *,
    formats: Iterable[str],
    lang: str = "pt",
    n_reamostras: int = 10_000,
) -> ResultadoFigura:
    """Adequação e cobertura por nível de SUFICIÊNCIA de contexto.

    A suficiência é julgada por LLM a partir do contexto recuperado, da questão e
    da resposta de referência.
    """
    if contexto.suficiencia is None:
        raise ValueError(
            "suficiencia_contexto.parquet ausente em metrics/; "
            "rode scripts/juiz_suficiencia_contexto.py antes de gerar a F5."
        )
    df = contexto.consolidated_by_cmin[contexto.cmin_ref]
    suf = contexto.suficiencia[["item_id", "veredicto_suficiencia"]]
    m = df.merge(suf, on="item_id", how="inner")
    m = m[m["veredicto_suficiencia"] >= 0]

    niveis = [
        (0, _t(lang, "suf_insuficiente")),
        (1, _t(lang, "suf_parcial")),
        (2, _t(lang, "suf_suficiente")),
    ]
    xs: list[int] = []
    adeq: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    cobertura: list[float] = []
    ns: list[int] = []
    for valor, _ in niveis:
        sub = m[m["veredicto_suficiencia"] == valor]
        if sub.empty:
            continue
        ic = ic_bootstrap_estratificado(
            sub,
            lambda d: float(d["z_winner"].mean()),
            coluna_estratificacao="has_doc_ref",
            n_reamostras=n_reamostras,
            metodo="percentile",
        )
        xs.append(valor)
        adeq.append(ic.point)
        err_lo.append(max(0.0, ic.point - ic.ci_lo))
        err_hi.append(max(0.0, ic.ci_hi - ic.point))
        cobertura.append(float((~sub["abstained"]).mean()))
        ns.append(int(len(sub)))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    h1 = ax.errorbar(
        xs, adeq, yerr=[err_lo, err_hi], marker="o", markersize=7, linewidth=1.8,
        color=COR_AZUL, capsize=4, label=_t(lang, "adequacao_vencedor"),
    )
    for x, a, n in zip(xs, adeq, ns, strict=True):
        percentual = f"{a:.1%}"
        if lang == "pt":
            percentual = percentual.replace(".", ",")
        ax.annotate(
            f"{percentual}\n(n={n})", (x, a), textcoords="offset points",
            xytext=(0, 13), ha="center", fontsize=8,
        )
    (h2,) = ax.plot(
        xs, cobertura, marker="s", linestyle="--", color=COR_LARANJA, linewidth=1.6,
        label=_t(lang, "cobertura_fracao"),
    )

    ax.set_xticks([v for v, _ in niveis])
    ax.set_xticklabels([r for _, r in niveis])
    ax.set_xlabel(_t(lang, "suficiencia_xlabel"))
    ax.set_ylabel(_t(lang, "percentual"))
    ax.set_ylim(0, 1.03)
    _configurar_percentual(ax)
    ax.legend(handles=[h1, h2], loc="upper left", frameon=True, framealpha=0.95)
    fig.text(
        0.5, -0.02,
        _t(lang, "rodape_suficiencia", b=f"{n_reamostras:,}".replace(",", " ")),
        ha="center", fontsize=8, color=COR_CINZA,
    )
    fig.tight_layout()
    return ResultadoFigura(
        nome="f5_suficiencia",
        caminhos=salvar_figura(fig, contexto.output_dir, "f5_suficiencia", formats),
    )


def gerar_figura_tradeoff_cmin(
    contexto: ContextoRun,
    *,
    formats: Iterable[str],
    lang: str = "pt",
) -> ResultadoFigura:
    """Gera o trade-off operacional entre precisão da abstenção e volume."""
    linhas: list[dict[str, float]] = []
    for cmin in contexto.cmins:
        df = contexto.consolidated_by_cmin[cmin]
        abst = df[df["abstained"]]
        linhas.append(
            {
                "c_min": cmin,
                "precisao_abstencao": (
                    float((abst["z_winner"] == 0).mean()) if not abst.empty else float("nan")
                ),
                "n_abstencoes": float(df["abstained"].sum()),
                "cobertura": float((~df["abstained"]).mean()),
                "risco_seletivo": (
                    float((df.loc[~df["abstained"], "z_winner"] == 0).mean())
                    if (~df["abstained"]).any()
                    else float("nan")
                ),
            }
        )
    resumo = pd.DataFrame(linhas).sort_values("c_min")

    largura = 6.4 if len(resumo) <= 2 else 7.4
    altura = 5.8 if len(resumo) <= 2 else 6.6
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(largura, altura), sharex=True)
    xs = resumo["c_min"].to_numpy(dtype=float)

    ax_top.plot(xs, resumo["precisao_abstencao"], color=COR_AZUL, marker="o", linewidth=2.0)
    for x, y in zip(xs, resumo["precisao_abstencao"], strict=True):
        if np.isnan(y):
            continue
        ax_top.text(x, y + 0.01, f"{y:.1%}", ha="center", va="bottom", fontsize=8)
    ax_top.set_title(_t(lang, "precisao_abstencao"))
    _configurar_percentual(
        ax_top,
        ymax=min(1.0, float(np.nanmax(resumo["precisao_abstencao"]) + 0.05)),
    )
    ax_top.set_ylabel(_t(lang, "precisao"))

    ax_bottom.plot(xs, resumo["n_abstencoes"], color=COR_LARANJA, marker="o", linewidth=2.0)
    min_abst = float(np.nanmin(resumo["n_abstencoes"]))
    max_abst = float(np.nanmax(resumo["n_abstencoes"]))
    amplitude_abst = max(max_abst - min_abst, 1.0)
    offset_abst = max(0.35, amplitude_abst * 0.12)
    for x, n_abst, cobertura, risco in zip(
        xs,
        resumo["n_abstencoes"],
        resumo["cobertura"],
        resumo["risco_seletivo"],
        strict=True,
    ):
        ax_bottom.text(
            x,
            n_abst + offset_abst,
            f"{int(n_abst)}\nĈ={cobertura:.1%}\nR̂_s={risco:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_bottom.set_title(_t(lang, "volume_abstencoes"))
    ax_bottom.set_ylabel(_t(lang, "n_abstencoes"))
    ax_bottom.set_xlabel(r"$C_{min}$")
    ax_bottom.set_xticks(xs)
    ax_bottom.set_ylim(min_abst - offset_abst * 0.7, max_abst + offset_abst * 2.1)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return ResultadoFigura(
        nome="f6_cmin_tradeoff",
        caminhos=salvar_figura(fig, contexto.output_dir, "f6_cmin_tradeoff", formats),
    )


def gerar_figura_diversidade(
    contexto: ContextoRun,
    *,
    formats: Iterable[str],
    lang: str = "pt",
) -> ResultadoFigura:
    """Gera o heatmap de concordância de erros entre agentes."""
    df_ref = contexto.consolidated_by_cmin[contexto.cmin_ref]
    matriz = pd.DataFrame(contexto.summary["diversity_matrix"]).reindex(
        index=contexto.agent_ids,
        columns=contexto.agent_ids,
    )

    display = matriz.copy()
    for aid in contexto.agent_ids:
        taxa_erro = float((df_ref[f"z_agent_{aid}"] == 0).mean())
        display.loc[aid, aid] = taxa_erro

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(
        display.to_numpy(dtype=float),
        cmap="YlOrRd",
        vmin=0.65,
        vmax=0.95,
        aspect="equal",
    )
    for i, aid_i in enumerate(contexto.agent_ids):
        for j, aid_j in enumerate(contexto.agent_ids):
            valor = float(display.loc[aid_i, aid_j])
            texto = f"{valor:.2f}"
            if lang == "pt":
                texto = texto.replace(".", ",")
            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if valor >= 0.83 else "black",
            )
    dependencia = fracao_erro_todos_agentes(
        {
            aid: df_ref[f"z_agent_{aid}"].astype(int).tolist()
            for aid in contexto.agent_ids
        }
    )
    ax.set_xticks(range(len(contexto.agent_ids)))
    ax.set_yticks(range(len(contexto.agent_ids)))
    ax.set_xticklabels(
        [contexto.agent_labels[aid] for aid in contexto.agent_ids],
        rotation=20,
        ha="right",
    )
    ax.set_yticklabels([contexto.agent_labels[aid] for aid in contexto.agent_ids])
    ax.set_xlabel(_t(lang, "agente_j"))
    ax.set_ylabel(_t(lang, "agente_i"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(_t(lang, "cbar_diversidade"))
    rodape = _t(lang, "rodape_diversidade", ratio=dependencia["ratio_observed_to_expected"])
    if lang == "pt":
        valor_ratio = f"{dependencia['ratio_observed_to_expected']:.3f}"
        rodape = rodape.replace(valor_ratio, valor_ratio.replace(".", ","))
    fig.text(
        0.46,
        0.03,
        rodape,
        ha="center",
        va="bottom",
        fontsize=8,
        color=COR_TEXTO,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return ResultadoFigura(
        nome="f7_error_agreement",
        caminhos=salvar_figura(fig, contexto.output_dir, "f7_error_agreement", formats),
    )


def gerar_figuras_artigo(
    *,
    base_dir: Path,
    experiment_id: str,
    comparison_run: str | None = None,
    cmin_ref: float | None = None,
    cmin_eta_principal: float = 0.7,
    cmin_eta_secundario: float | None = 0.9,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    lang: str = "pt",
    main_label: str | None = None,
    comparison_label: str | None = None,
) -> list[ResultadoFigura]:
    """Gera o conjunto principal de figuras do artigo."""
    if lang not in TEXTOS:
        raise ValueError(f"lang deve ser um de {sorted(TEXTOS)}, recebeu {lang!r}")
    aplicar_estilo_artigo()
    contexto = ContextoRun.carregar(
        base_dir,
        experiment_id,
        output_dir=output_dir,
        cmin_ref=cmin_ref,
    )
    comparacao = (
        ContextoRun.carregar(base_dir, comparison_run) if comparison_run is not None else None
    )

    resultados = [
        gerar_figura_risco_cobertura(
            contexto,
            comparacao=comparacao,
            formats=formats,
            lang=lang,
            main_label=main_label,
            comparison_label=comparison_label,
        ),
        gerar_figura_reliability(contexto, formats=formats, lang=lang),
        gerar_figura_arquitetura(contexto, formats=formats, lang=lang),
        gerar_figura_eta(
            contexto,
            cmin_principal=cmin_eta_principal,
            cmin_secundario=cmin_eta_secundario,
            formats=formats,
            lang=lang,
        ),
    ]
    if contexto.suficiencia is not None:
        resultados.append(gerar_figura_suficiencia(contexto, formats=formats, lang=lang))
    else:
        logger.warning(
            "F5 (suficiência) pulada: metrics/suficiencia_contexto.parquet ausente em %s. "
            "Rode scripts/juiz_suficiencia_contexto.py --run %s.",
            contexto.experiment_id,
            contexto.experiment_id,
        )
    resultados += [
        gerar_figura_tradeoff_cmin(contexto, formats=formats, lang=lang),
        gerar_figura_diversidade(contexto, formats=formats, lang=lang),
    ]
    logger.info(
        "Figuras salvas em %s: %s",
        contexto.output_dir,
        ", ".join(resultado.nome for resultado in resultados),
    )
    return resultados
