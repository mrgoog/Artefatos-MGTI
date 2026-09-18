"""Cálculo de retrieval recall — Nível 1 (match por source_doc).

Lógica compartilhada entre `scripts/eval_rag.py` (avaliação offline) e
`orquestracao/execucao.py` (coluna `retrieval_recall` no consolidado).

Suporta dois modos de IDs esperados:
- ``item_ids``: source_docs gerados por `construir_corpus_de_itens`
  (prefixos ``disp_{item_id}_``, ``carf_{item_id}_``).
- ``files``: source_docs extraídos de ``file``/``filename`` no JSON bruto
  do BR-TaxQA-R (ex: ``Lei nº 13.105.txt``).

O modo é auto-detectado a partir do índice RAG persistido: se ≥90% dos
source_docs começam com ``disp_`` ou ``carf_``, assume ``item_ids``;
caso contrário, assume ``files``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ensemble_llm.esquemas import ItemDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modo item_ids (índice construído inline por construir_corpus_de_itens)
# ---------------------------------------------------------------------------

def _source_docs_esperados_item(item: ItemDataset) -> set[str]:
    """Source_doc IDs esperados no esquema ``disp_``/``carf_``."""
    esperados: set[str] = set()
    for i in range(len(item.dispositivos_legais)):
        esperados.add(f"disp_{item.item_id}_{i:02d}")
    for i in range(len(item.ementas_carf)):
        esperados.add(f"carf_{item.item_id}_{i:02d}")
    return esperados


# ---------------------------------------------------------------------------
# Modo files (índice construído por build_rag_index.py com filenames reais)
# ---------------------------------------------------------------------------

def _coletar_dicts_recursivo(valor: Any) -> list[dict[str, Any]]:
    """Percorre estrutura aninhada coletando todos os dicts encontrados."""
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
    """Extrai source_doc esperados por ``file``/``filename`` do JSON bruto."""
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


def mapa_source_docs_por_arquivo(caminho_json: str | Path) -> dict[str, set[str]]:
    """Mapeia ``item_id`` → ``set[source_doc]`` esperados no modo ``files``.

    Lê o JSON bruto do BR-TaxQA-R e extrai filenames de
    ``all_formatted_references`` / ``formatted_references``.
    """
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
        numero_q = q.get("question_number")
        if numero_q is None:
            continue
        try:
            item_id = f"q_{int(numero_q):04d}"
        except Exception:
            continue
        mapa[item_id] = _source_docs_esperados_files(q)
    return mapa


# ---------------------------------------------------------------------------
# Auto-detecção de modo
# ---------------------------------------------------------------------------

def detectar_modo(rag_root: Path) -> str:
    """Detecta modo (``item_ids`` ou ``files``) pelos source_docs do índice persistido.

    Se ≥90% dos source_docs começam com ``disp_`` ou ``carf_``, retorna
    ``item_ids``; caso contrário, ``files``.
    """
    chunks_path = rag_root / "denso" / "chunks.parquet"
    if not chunks_path.exists():
        logger.info("chunks.parquet não encontrado em %s — assumindo item_ids", rag_root)
        return "item_ids"

    df = pd.read_parquet(chunks_path, columns=["source_doc"])
    if df.empty:
        return "item_ids"

    s = df["source_doc"].astype(str)
    frac_item_ids = float((s.str.startswith("disp_") | s.str.startswith("carf_")).mean())
    modo = "item_ids" if frac_item_ids >= 0.9 else "files"
    logger.info("Auto-detecção: frac_item_ids=%.3f → modo=%s", frac_item_ids, modo)
    return modo


# ---------------------------------------------------------------------------
# Função principal: recall por item
# ---------------------------------------------------------------------------

def calcular_retrieval_recall(
    items: list[ItemDataset],
    agent_tables: dict[str, pd.DataFrame],
    *,
    rag_root: Path | None = None,
    questions_json: str | Path | None = None,
) -> dict[str, float]:
    """Computa Recall@K (Nível 1) por item a partir dos chunks já recuperados.

    Usa ``retrieved_chunk_ids`` de qualquer agente (todos compartilham o RAG).
    Para itens sem referência documental, retorna NaN.

    O modo de IDs esperados é auto-detectado:

    - Se ``rag_root`` é fornecido, detecta pelo índice persistido.
    - Se o modo detectado é ``files`` e ``questions_json`` está disponível,
      extrai filenames do JSON bruto.
    - Caso contrário, usa o esquema ``disp_``/``carf_`` (modo ``item_ids``).

    Args:
        items: lista de itens do dataset.
        agent_tables: mapping agent_id → DataFrame (precisa de ``retrieved_chunk_ids``).
        rag_root: diretório do índice RAG persistido (para auto-detecção).
        questions_json: caminho do JSON bruto do BR-TaxQA-R (para modo ``files``).

    Returns:
        Mapping ``item_id`` → ``recall`` (float em [0,1] ou NaN).
    """
    # Determinar modo
    modo = "item_ids"
    if rag_root is not None:
        modo = detectar_modo(rag_root)

    # Construir mapa de source_docs esperados por item
    esperados_map: dict[str, set[str]] = {}

    if modo == "files" and questions_json is not None:
        esperados_map = mapa_source_docs_por_arquivo(questions_json)
        logger.info(
            "retrieval_recall: modo=files, %d itens com referências mapeadas",
            sum(1 for v in esperados_map.values() if v),
        )
    else:
        if modo == "files" and questions_json is None:
            logger.warning(
                "Modo 'files' detectado mas questions_json não fornecido. "
                "Caindo para modo 'item_ids' — recall pode ser 0.0 se o "
                "índice usa filenames. Passe questions_json para corrigir."
            )
            modo = "item_ids"

        for item in items:
            if item.has_doc_ref:
                esperados_map[item.item_id] = _source_docs_esperados_item(item)

    # Pegar chunks do primeiro agente (todos iguais — mesmo RAG, mesma query)
    primeiro_agente = next(iter(agent_tables))
    df_agente = agent_tables[primeiro_agente]
    if "item_id" in df_agente.columns:
        df_agente = df_agente.set_index("item_id")

    item_map = {it.item_id: it for it in items}
    recall_por_item: dict[str, float] = {}

    for item_id, item in item_map.items():
        esperados = esperados_map.get(item_id, set())

        if not item.has_doc_ref or not esperados:
            recall_por_item[item_id] = float("nan")
            continue

        if item_id not in df_agente.index:
            recall_por_item[item_id] = float("nan")
            continue

        chunk_ids = df_agente.loc[item_id, "retrieved_chunk_ids"]
        if chunk_ids is None or (hasattr(chunk_ids, "__len__") and len(chunk_ids) == 0):
            recall_por_item[item_id] = 0.0
            continue

        source_docs_retornados = {cid.split("#")[0] for cid in chunk_ids}
        hits = source_docs_retornados & esperados
        recall_por_item[item_id] = len(hits) / len(esperados)

    n_com_recall = sum(1 for v in recall_por_item.values() if v == v)  # not NaN
    n_nan = sum(1 for v in recall_por_item.values() if v != v)
    logger.info(
        "retrieval_recall: modo=%s, %d itens com recall, %d NaN, média=%.3f",
        modo,
        n_com_recall,
        n_nan,
        (sum(v for v in recall_por_item.values() if v == v) / n_com_recall)
        if n_com_recall > 0
        else float("nan"),
    )

    return recall_por_item

