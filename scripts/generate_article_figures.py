"""Gera as figuras principais a partir dos runs persistidos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ensemble_llm.avaliacao.figuras_artigo import gerar_figuras_artigo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera figuras do artigo a partir de runs locais.")
    parser.add_argument(
        "--run",
        default="phase2_local_full_715",
        help="Experiment ID principal. Default: phase2_local_full_715.",
    )
    parser.add_argument(
        "--comparison-run",
        default="phase1_api_pilot_200_1",
        help=(
            "Experiment ID opcional para a curva risco-cobertura comparativa. "
            "Use string vazia para desabilitar."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default="runs",
        help="Diretório base dos runs. Default: runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de saída. Default: runs/<exp_id>/figures/article_v3.",
    )
    parser.add_argument(
        "--cmin-ref",
        type=float,
        default=None,
        help=(
            "C_min de referência para figuras que usam um consolidado específico. "
            "Default: 0.7 se existir."
        ),
    )
    parser.add_argument(
        "--cmin-eta-principal",
        type=float,
        default=0.7,
        help="C_min do limiar principal em F4. Default: 0.7.",
    )
    parser.add_argument(
        "--cmin-eta-secundario",
        type=float,
        default=0.9,
        help="C_min do limiar secundário em F4. Use valor negativo para desabilitar. Default: 0.9.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Formatos de saída. Default: png pdf.",
    )
    parser.add_argument(
        "--lang",
        default="pt",
        choices=["pt", "en"],
        help="Idioma dos rótulos das figuras. Default: pt.",
    )
    parser.add_argument(
        "--main-label",
        default=None,
        help="Rótulo da curva principal na figura risco-cobertura (f1).",
    )
    parser.add_argument(
        "--comparison-label",
        default=None,
        help="Rótulo da curva de comparação na figura risco-cobertura (f1).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    comparison_run = args.comparison_run or None
    cmin_eta_secundario = None if args.cmin_eta_secundario < 0 else args.cmin_eta_secundario
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    resultados = gerar_figuras_artigo(
        base_dir=Path(args.base_dir),
        experiment_id=args.run,
        comparison_run=comparison_run,
        cmin_ref=args.cmin_ref,
        cmin_eta_principal=args.cmin_eta_principal,
        cmin_eta_secundario=cmin_eta_secundario,
        output_dir=output_dir,
        formats=args.formats,
        lang=args.lang,
        main_label=args.main_label,
        comparison_label=args.comparison_label,
    )

    for resultado in resultados:
        for caminho in resultado.caminhos:
            logger.info("Gerado: %s", caminho)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
