"""
Download do dataset BR-TaxQA-R do Hugging Face.

Salva três arquivos em data/:
  1. questions_QA_2024_v1.1.json          ← questões (usado pelo pipeline)
  2. acordaos_CARF_2023.json              ← corpus de acordãos CARF completo
  3. referred_legal_documents_QA_2024_v1.1.json  ← dispositivos legais referenciados

Uso:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --output-dir data/

Requer conexão com internet. O Hugging Face Hub faz cache local em
~/.cache/huggingface/ nas execuções seguintes (sem custo de rede).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "unicamp-dl/BR-TaxQA-R"
QUESTIONS_FILENAME = "questions_QA_2024_v1.1.json"

# Arquivos adicionais presentes no repositório do dataset (não são splits Arrow)
EXTRA_FILES = [
    "acordaos_CARF_2023.json",
    "referred_legal_documents_QA_2024_v1.1.json",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa BR-TaxQA-R e corpus legal do Hugging Face, salva em data/."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Diretório de saída (padrão: data/)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split a baixar para as questões, ex: 'train'. Padrão: todos concatenados.",
    )
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Baixa apenas as questões; pula acordaos_CARF e referred_legal_documents.",
    )
    return parser.parse_args()


def _baixar_questoes(
    output_path: Path,
) -> int:
    """Baixa o JSON canônico de questões do Hub e retorna o número de itens."""
    from huggingface_hub import hf_hub_download

    print("\n[1] Baixando questões (arquivo JSON canônico do repositório)...")
    try:
        cached = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=QUESTIONS_FILENAME,
            repo_type="dataset",
        )
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar arquivo de questões: {e}") from e

    shutil.copy2(cached, output_path)

    try:
        with output_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Falha ao validar JSON de questões baixado: {e}") from e

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Arquivo de questões baixado está vazio ou em formato inesperado.")

    print(f"    Colunas: {list(rows[0].keys())}")
    mb = output_path.stat().st_size / 1_048_576
    print(f"    OK — {len(rows)} itens, {mb:.1f} MB")
    return len(rows)


def _baixar_arquivo_hub(filename: str, output_path: Path) -> None:
    """Baixa um arquivo individual do repositório do dataset via hf_hub_download."""
    from huggingface_hub import hf_hub_download

    print(f"    Baixando {filename}...")
    cached = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
    )
    shutil.copy2(cached, output_path)
    mb = output_path.stat().st_size / 1_048_576
    print(f"    OK — {mb:.1f} MB → {output_path}")


def _verificar_loader(questions_path: Path) -> None:
    """Verifica compatibilidade com carregar_br_taxqa_r e imprime estatísticas."""
    from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r

    itens = carregar_br_taxqa_r(questions_path)
    n_com = sum(1 for i in itens if i.has_doc_ref)
    print(f"    Itens carregados: {len(itens)}")
    print(f"    Com referência legal (has_doc_ref=True):  {n_com}")
    print(f"    Sem referência legal (has_doc_ref=False): {len(itens) - n_com}")


def main() -> int:
    args = _parse_args()

    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        print(f"Erro: dependência não encontrada — {e}\nExecute: uv sync", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Repositório: {DATASET_REPO}")
    print(f"Destino:     {output_dir.resolve()}")
    print("=" * 60)

    errors: list[str] = []

    # --- Questões ---
    questions_path = output_dir / QUESTIONS_FILENAME
    try:
        if args.split is not None:
            print(
                "    Aviso: --split foi ignorado; o repositório publica o arquivo JSON consolidado.",
                file=sys.stderr,
            )
        _baixar_questoes(questions_path)
    except RuntimeError as e:
        print(f"ERRO nas questões: {e}", file=sys.stderr)
        errors.append(str(e))

    # --- Arquivos extras do corpus ---
    if not args.questions_only:
        print(f"\n[2] Baixando arquivos do corpus legal...")
        for filename in EXTRA_FILES:
            dest = output_dir / filename
            try:
                _baixar_arquivo_hub(filename, dest)
            except Exception as e:
                msg = f"Falha ao baixar {filename}: {e}"
                print(f"    AVISO: {msg}", file=sys.stderr)
                errors.append(msg)

    # --- Verificação do loader ---
    if questions_path.exists():
        print(f"\n[3] Verificando compatibilidade com carregar_br_taxqa_r...")
        try:
            _verificar_loader(questions_path)
        except Exception as e:
            print(f"    Aviso: falha na verificação — {e}", file=sys.stderr)

    # --- Resumo ---
    print("\n" + "=" * 60)
    arquivos_gerados = [p for p in output_dir.iterdir() if p.suffix == ".json"]
    if errors:
        print(f"CONCLUÍDO COM {len(errors)} AVISO(S):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("SUCESSO")
    print(f"\nArquivos em {output_dir.resolve()}:")
    for p in sorted(arquivos_gerados):
        mb = p.stat().st_size / 1_048_576
        print(f"  {p.name:<55s} {mb:6.1f} MB")
    print("=" * 60)
    return 1 if errors and not questions_path.exists() else 0


if __name__ == "__main__":
    sys.exit(main())
