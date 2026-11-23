"""Avaliação offline da qualidade de recuperação do RAG — Nível 1 (match por source_doc).

Uso:
    uv run python scripts/eval_rag.py --config configs/rag_brtaxq.yaml

Executa `recuperar(item.pergunta)` para itens com referência documental,
computa Recall@K, Hit@K e MRR por item, e salva resultados em parquet.
Suporta modos de ID esperado (`item_ids`, `files` ou `auto`).
Não depende de nenhuma run de agentes — usa o índice RAG diretamente.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ensemble_llm.configuracao import carregar_config_rag_offline
from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r
from ensemble_llm.recuperacao.construtor_corpus import carregar_recuperador_hibrido
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _source_docs_esperados_item(item) -> set[str]:
    """Gera os source_doc IDs esperados no esquema disp_/carf_."""
    esperados: set[str] = set()
    for i in range(len(item.dispositivos_legais)):
        esperados.add(f"disp_{item.item_id}_{i:02d}")
    for i in range(len(item.ementas_carf)):
        esperados.add(f"carf_{item.item_id}_{i:02d}")
    return esperados


def _item_id_raw(dados_questao: dict[str, Any]) -> str | None:
    numero_q = dados_questao.get("question_number")
    if numero_q is None:
        return None
    try:
        return f"q_{int(numero_q):04d}"
    except Exception:
        return None


def _coletar_dicts_recursivo(valor: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(valor, dict):
        out.append(valor)
        for v in valor.values():
            out.extend(_coletar_dicts_recursivo(v))
    elif isinstance(valor, list):
        for v in valor:
            out.extend(_coletar_dicts_recursivo(v))
    return out


def _source_docs_esperados_files(dados_questao: dict[str, Any]) -> set[str]:
    """Extrai source_doc esperados por `file`/`filename` do JSON bruto."""
    esperados: set[str] = set()
    for chave in ("all_formatted_references", "formatted_references"):
        if chave not in dados_questao:
            continue
        blocos = _coletar_dicts_recursivo(dados_questao.get(chave))
        for d in blocos:
            for campo in ("file", "filename"):
                valor = d.get(campo)
                if isinstance(valor, str) and valor.strip():
                    esperados.add(valor.strip())
        if esperados:
            break
    return esperados


def _mapa_source_docs_por_arquivo(caminho_json: str | Path) -> dict[str, set[str]]:
    """Mapeia item_id -> source_docs esperados no modo files."""
    caminho = Path(caminho_json)
    with caminho.open("r", encoding="utf-8") as f:
        dados_brutos = json.load(f)

    if isinstance(dados_brutos, list):
        questoes = dados_brutos
    elif isinstance(dados_brutos, dict):
        questoes = dados_brutos.get("questions") or list(dados_brutos.values())
    else:
        raise ValueError(f"Formato inesperado em {caminho}: {type(dados_brutos)}")

    mapa: dict[str, set[str]] = {}
    for q in questoes:
        if not isinstance(q, dict):
            continue
        item_id = _item_id_raw(q)
        if item_id is None:
            continue
        mapa[item_id] = _source_docs_esperados_files(q)
    return mapa


def _detectar_modo_auto(rag_root: Path) -> str:
    """Detecta modo esperado pelos source_docs do índice persistido."""
    chunks_path = rag_root / "denso" / "chunks.parquet"
    if not chunks_path.exists():
        return "item_ids"

    df = pd.read_parquet(chunks_path, columns=["source_doc"])
    if df.empty:
        return "item_ids"

    s = df["source_doc"].astype(str)
    frac_item_ids = float((s.str.startswith("disp_") | s.str.startswith("carf_")).mean())
    modo = "item_ids" if frac_item_ids >= 0.9 else "files"
    logger.info("Auto-detecção: frac_item_ids=%.3f -> modo=%s", frac_item_ids, modo)
    return modo


def _avaliar_item(chunks_retornados: list, source_docs_esperados: set[str]) -> dict[str, float | int]:
    """Computa Recall@K, Hit@K e MRR para um item."""
    if not source_docs_esperados:
        return {
            "recall_at_k": float("nan"),
            "hit_at_k": float("nan"),
            "mrr": float("nan"),
            "n_ref": 0,
            "n_hits": 0,
            "k": len(chunks_retornados),
        }

    source_docs_retornados = [chunk.chunk_id.split("#")[0] for chunk in chunks_retornados]
    source_docs_unicos_retornados = list(dict.fromkeys(source_docs_retornados))

    hits = set(source_docs_unicos_retornados) & source_docs_esperados
    n_hits = len(hits)
    n_ref = len(source_docs_esperados)
    recall = n_hits / n_ref
    hit = 1.0 if n_hits > 0 else 0.0

    mrr = 0.0
    for rank, source_doc in enumerate(source_docs_unicos_retornados, start=1):
        if source_doc in source_docs_esperados:
            mrr = 1.0 / rank
            break

    return {
        "recall_at_k": recall,
        "hit_at_k": hit,
        "mrr": mrr,
        "n_ref": n_ref,
        "n_hits": n_hits,
        "k": len(chunks_retornados),
    }


def _resolver_top_ks(top_k_total: int | None, top_k_denso_cfg: int, top_k_bm25_cfg: int) -> tuple[int, int]:
    """Resolve top-k denso e BM25 a partir do override opcional `--top-k`."""
    if top_k_total is None:
        return top_k_denso_cfg, top_k_bm25_cfg
    if top_k_total < 1:
        raise ValueError("--top-k deve ser >= 1.")

    total_cfg = top_k_denso_cfg + top_k_bm25_cfg
    frac_denso = top_k_denso_cfg / total_cfg
    top_k_denso = max(1, math.ceil(top_k_total * frac_denso))
    top_k_bm25 = max(1, top_k_total - top_k_denso)
    return top_k_denso, top_k_bm25


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliação offline do RAG (Nível 1).")
    parser.add_argument("--config", required=True, help="YAML de configuração.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K total do híbrido. Default: dense_top_k + bm25_top_k do YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Caminho do parquet de saída. "
            "Default: <persisted_index_dir>/metrics/rag_eval.parquet"
        ),
    )
    parser.add_argument(
        "--questions-json",
        default=None,
        help=(
            "Override opcional do arquivo principal de questões "
            "definido em dataset.questions_file."
        ),
    )
    parser.add_argument(
        "--expected-doc-id-mode",
        choices=["auto", "item_ids", "files"],
        default="auto",
        help="auto=detecta pelo índice; item_ids=disp_/carf_; files=file/filename no JSON bruto.",
    )
    args = parser.parse_args()

    config = carregar_config_rag_offline(args.config)

    if config.retrieval.persisted_index_dir is None:
        print("Erro: retrieval.persisted_index_dir não definido no YAML.", file=sys.stderr)
        return 1

    rag_root = Path(config.retrieval.persisted_index_dir)
    if not rag_root.exists():
        print(
            f"Erro: índice RAG não encontrado em {rag_root}. "
            "Rode build_rag_index.py primeiro.",
            file=sys.stderr,
        )
        return 1

    questions_json = (
        Path(args.questions_json)
        if args.questions_json is not None
        else config.dataset.resolver_questions_path()
    )
    logger.info("Carregando dataset de %s", questions_json)
    items = carregar_br_taxqa_r(questions_json)

    modo_ids = args.expected_doc_id_mode
    if modo_ids == "auto":
        modo_ids = _detectar_modo_auto(rag_root)

    esperados_files_map: dict[str, set[str]] = {}
    if modo_ids == "files":
        esperados_files_map = _mapa_source_docs_por_arquivo(questions_json)
        items_com_ref = [it for it in items if esperados_files_map.get(it.item_id)]
    else:
        items_com_ref = [it for it in items if it.has_doc_ref]

    items_sem_ref = [it for it in items if it not in items_com_ref]
    logger.info(
        "Total: %d itens; com ref (modo=%s): %d; sem ref: %d",
        len(items),
        modo_ids,
        len(items_com_ref),
        len(items_sem_ref),
    )

    top_k_denso, top_k_bm25 = _resolver_top_ks(
        args.top_k,
        config.retrieval.dense_top_k,
        config.retrieval.bm25_top_k,
    )
    logger.info("Top-K efetivo: denso=%d, bm25=%d", top_k_denso, top_k_bm25)

    embedder = VetorizadorSentenceTransformer(
        nome_modelo=config.retrieval.embedder_model,
        tamanho_lote=config.retrieval.embedder_batch_size,
        dispositivo=config.retrieval.embedder_device,
    )
    retriever = carregar_recuperador_hibrido(
        rag_root,
        vetorizador=embedder,
        top_k_denso=top_k_denso,
        top_k_bm25=top_k_bm25,
    )
    logger.info("Índice RAG carregado de %s", rag_root)

    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    for i, item in enumerate(items_com_ref, start=1):
        chunks = retriever.recuperar(item.pergunta)
        if modo_ids == "files":
            esperados = esperados_files_map.get(item.item_id, set())
        else:
            esperados = _source_docs_esperados_item(item)

        resultado = _avaliar_item(chunks, esperados)
        resultado["item_id"] = item.item_id
        resultado["has_doc_ref"] = True
        resultado["expected_doc_id_mode"] = modo_ids
        resultado["source_docs_retornados"] = [c.chunk_id.split("#")[0] for c in chunks]
        resultado["source_docs_esperados"] = sorted(esperados)
        rows.append(resultado)

        if i % 50 == 0 or i == len(items_com_ref):
            logger.info("Avaliados %d/%d itens.", i, len(items_com_ref))

    duracao = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    if df.empty:
        print("Erro: nenhum item elegível para avaliação.", file=sys.stderr)
        return 1

    print()
    print("=" * 70)
    print("RESULTADOS — Avaliação do RAG (Nível 1: source_doc prefix match)")
    print("=" * 70)
    print(f"Modo de IDs esperados: {modo_ids}")
    print(f"Itens avaliados: {len(df)}")
    print(f"Tempo total: {duracao:.1f}s ({duracao/len(df)*1000:.0f}ms/item)")
    print()
    print(f"Recall@K médio:   {df['recall_at_k'].mean():.3f}  (mediana: {df['recall_at_k'].median():.3f})")
    print(f"Hit@K:            {df['hit_at_k'].mean():.3f}  ({df['hit_at_k'].sum():.0f}/{len(df)} itens)")
    print(f"MRR:              {df['mrr'].mean():.3f}  (mediana: {df['mrr'].median():.3f})")
    print()

    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
        n = int((df["recall_at_k"] >= threshold).sum())
        print(f"  Recall >= {threshold:.2f}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

    print()
    print(f"n_ref médio:  {df['n_ref'].mean():.1f}  (min={df['n_ref'].min()}, max={df['n_ref'].max()})")
    print(f"n_hits médio: {df['n_hits'].mean():.1f}")
    print()

    output_path = (
        Path(args.output)
        if args.output
        else rag_root / "metrics" / "rag_eval.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_save = df.copy()
    df_save["source_docs_retornados"] = df_save["source_docs_retornados"].apply(lambda x: "|".join(x))
    df_save["source_docs_esperados"] = df_save["source_docs_esperados"].apply(lambda x: "|".join(x))
    df_save.to_parquet(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")

    hit_rate = float(df["hit_at_k"].mean())
    if hit_rate < 0.3:
        print(
            "\nATENCAO: Hit@K < 0.3 — possível problema grave na indexação ou chunking.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
