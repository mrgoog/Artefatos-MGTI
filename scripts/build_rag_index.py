"""
Build offline do índice RAG compartilhado.

Uso:
    uv run python scripts/build_rag_index.py --config configs/rag_brtaxq.yaml

O índice é construído uma vez, a partir do dataset completo e dos corpora extras,
e persistido em `retrieval.persisted_index_dir`. As execuções dos experimentos
passam a apenas carregar esse diretório.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ensemble_llm.configuracao import carregar_config_rag_offline
from ensemble_llm.dados.conjunto_dados import (
    carregar_br_taxqa_r,
    carregar_documentos_legais,
    construir_corpus_de_itens,
)
from ensemble_llm.recuperacao.construtor_corpus import (
    construir_recuperador_hibrido,
    salvar_metadados_indice_rag,
)
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrói índice RAG offline compartilhado.")
    parser.add_argument("--config", required=True, help="YAML de configuração do índice RAG.")
    parser.add_argument(
        "--questions-json",
        default=None,
        help=(
            "Override opcional do arquivo principal de questões "
            "definido em dataset.questions_file."
        ),
    )
    parser.add_argument(
        "--corpus-json",
        nargs="*",
        metavar="PATH",
        default=None,
        help=(
            "Caminhos de JSONs adicionais para o corpus RAG. "
            "Se omitido, usa dataset.corpus_files do YAML."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = carregar_config_rag_offline(args.config)

    if config.retrieval.persisted_index_dir is None:
        print(
            "Erro: retrieval.persisted_index_dir não definido no YAML. "
            "Defina um diretório compartilhado para o índice offline.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(config.retrieval.persisted_index_dir)
    questions_json = (
        Path(args.questions_json)
        if args.questions_json is not None
        else config.dataset.resolver_questions_path()
    )
    corpus_paths = (
        [Path(p) for p in args.corpus_json]
        if args.corpus_json is not None
        else config.dataset.resolver_corpus_paths()
    )

    t0 = time.perf_counter()
    logger.info("Carregando dataset completo de %s", questions_json)
    itens = carregar_br_taxqa_r(questions_json)
    logger.info("Itens carregados: %d", len(itens))

    documentos = construir_corpus_de_itens(itens)
    logger.info("Corpus base das questões: %d documento(s).", len(documentos))

    if corpus_paths:
        extra = carregar_documentos_legais(corpus_paths)
        logger.info(
            "Corpus extra: %d documento(s) de %d arquivo(s).",
            len(extra),
            len(corpus_paths),
        )
        documentos.update(extra)

    logger.info(
        "Construindo índice RAG offline em %s com embedder=%s device=%s batch_size=%d",
        output_dir,
        config.retrieval.embedder_model,
        config.retrieval.embedder_device or "auto",
        config.retrieval.embedder_batch_size,
    )
    vetorizador = VetorizadorSentenceTransformer(
        nome_modelo=config.retrieval.embedder_model,
        tamanho_lote=config.retrieval.embedder_batch_size,
        dispositivo=config.retrieval.embedder_device,
    )
    construir_recuperador_hibrido(
        documentos,
        vetorizador=vetorizador,
        top_k_denso=config.retrieval.dense_top_k,
        top_k_bm25=config.retrieval.bm25_top_k,
        diretorio_persistencia=output_dir,
    )
    salvar_metadados_indice_rag(
        output_dir,
        {
            "dataset_id": config.dataset.dataset_id,
            "dataset_format": config.dataset.format,
            "questions_file": str(questions_json),
            "corpus_files": [str(path) for path in corpus_paths],
            "embedder_model": config.retrieval.embedder_model,
            "chunk_max_tokens": config.retrieval.chunk_max_tokens,
            "chunk_overlap_tokens": config.retrieval.chunk_overlap_tokens,
        },
    )
    logger.info(
        "Índice RAG offline concluído em %.2fs e persistido em %s",
        time.perf_counter() - t0,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
