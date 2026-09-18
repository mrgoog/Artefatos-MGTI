"""Índices deste repositório: construção do índice reparado (sem pseudo-docs), resolução por degrau e comparação de atingibilidade."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ensemble_llm.dados.conjunto_dados import carregar_documentos_legais
from ensemble_llm.recuperacao.construtor_corpus import (
    carregar_metadados_indice_rag,
    construir_chunks_corpus,
    salvar_metadados_indice_rag,
)
from ensemble_llm.recuperacao.indice_bm25 import IndiceBM25
from ensemble_llm.recuperacao.indice_denso import IndiceDenso
from ensemble_llm.recuperacao.segmentacao import ConfigSegmentacao, tokenizador_espacos
from ensemble_llm.recuperacao.vetorizacao import Vetorizador, VetorizadorSentenceTransformer

from rag2.config import ConfigRag2
from rag2.constantes import MODELO_EMBEDDER
from rag2.dados import ItemAvaliado, tipo_fonte
from rag2.indice import IndiceHibrido, carregar_indice
from rag2.metricas import Casador
from rag2.reparo import DIR_REPARADO, NOME_JSON_REPARADO

INDICE_REPARADO = "rag2_reparado"
"""Nome do diretório do índice reparado sob runs/_shared/rag/ e valor de `indice_id` nos metadados."""
BATCH_EMBEDDER = 8
"""Batch do embedder na construção — o da produção (configs/rag_brtaxq.yaml); batch 32 não é mais rápido em fp32 (medido 2026-09-12)."""


class IndiceJaExiste(FileExistsError):
    """O diretório já tem index_metadata.json; registrar uma vez — `--sobrescrever` exige decisão registrada."""


def sha256_arquivo(caminho: Path) -> str:
    """sha256 hex dos bytes do arquivo, lido em blocos de 1 MiB (o CARF tem 173 MB)."""
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_corpus_reparado(arquivo_normas: Path, arquivo_carf: Path) -> dict[str, str]:
    """filename → texto dos acórdãos e das normas reparadas, nesta ordem — a dos chunks não-pseudo do índice de produção.

    A ordem (CARF em 568–61002, normas em 61003–66864, medido 2026-09-12) decide os desempates
    estáveis do BM25 (sort por posição) e do `argpartition` do denso; carregar as normas primeiro
    tornaria a ordem uma terceira diferença entre R0 e R0'.
    Nunca chama `construir_corpus_de_itens` (é o que injeta os pseudo-docs, arch §2); um nome
    `disp_*`/`carf_*` no corpus é ValueError, e arquivo ausente é FileNotFoundError (o loader
    da origem só avisaria e seguiria com menos documentos).
    """
    for p in (arquivo_normas, arquivo_carf):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    docs = carregar_documentos_legais([Path(arquivo_carf), Path(arquivo_normas)])
    pseudo = sorted(d for d in docs if tipo_fonte(d) == "pseudo_doc")
    if pseudo:
        raise ValueError(f"pseudo-documentos no corpus (arch §2): {pseudo[:5]}")
    return docs


def _formatar(cont: Counter) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(cont.items()))


def _nome_gpu(device: str | None) -> str | None:
    """Nome da GPU do device, ou None (CPU, sem CUDA, torch ausente) — procedência, dada a ordem torch ≠ nvidia-smi."""
    if not device or not device.startswith("cuda"):
        return None
    try:
        import torch

        return torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
    except Exception:
        return None


def construir_indice_reparado(
    cfg: ConfigRag2,
    *,
    saida: Path | None = None,
    arquivo_normas: Path | None = None,
    arquivo_carf: Path | None = None,
    vetorizador: Vetorizador | None = None,
    sobrescrever: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Segmenta (512/64, tokenizador_espacos — a produção), embute (bge-m3 fp32, batch 8), indexa BM25, persiste e grava index_metadata.json.

    Composição um nível abaixo de `construir_recuperador_hibrido` da origem (mesmas três chamadas) para
    imprimir a composição do corpus ANTES do embedding: um corpus errado aparece em segundos, não em 45 min.
    Registrar uma vez: com index_metadata.json presente e sem `sobrescrever`, levanta IndiceJaExiste
    sem tocar em nada. Devolve os metadados gravados.
    """
    saida = Path(saida) if saida is not None else cfg.dir_indice_reparado()
    arquivo_normas = Path(arquivo_normas) if arquivo_normas is not None else DIR_REPARADO / NOME_JSON_REPARADO
    arquivo_carf = Path(arquivo_carf) if arquivo_carf is not None else cfg.arquivo_carf()
    if carregar_metadados_indice_rag(saida) is not None and not sobrescrever:
        raise IndiceJaExiste(str(saida))
    t0 = time.perf_counter()
    docs = carregar_corpus_reparado(arquivo_normas, arquivo_carf)
    seg = ConfigSegmentacao()
    chunks = construir_chunks_corpus(docs, tokenizador=tokenizador_espacos, config_segmentacao=seg)
    documentos = Counter(tipo_fonte(d) for d in docs)
    por_tipo = Counter(tipo_fonte(c.source_doc) for c in chunks)
    log(f"documentos: {_formatar(documentos)} | chunks: {_formatar(por_tipo)} | total={len(chunks)}")
    if vetorizador is None:
        vetorizador = VetorizadorSentenceTransformer(
            nome_modelo=MODELO_EMBEDDER, tamanho_lote=BATCH_EMBEDDER, dispositivo=cfg.device
        )
    denso = IndiceDenso(vetorizador=vetorizador)
    denso.construir(chunks)
    bm25 = IndiceBM25()
    bm25.construir(chunks)
    denso.salvar(saida / "denso")
    bm25.salvar(saida / "bm25")
    device = getattr(vetorizador, "dispositivo", None)
    meta = {
        "indice_id": INDICE_REPARADO,
        "corpus_files": [arquivo_carf.name, arquivo_normas.name],   # ordem de carga = ordem dos chunks no índice
        "sha256_corpus": {
            arquivo_carf.name: sha256_arquivo(arquivo_carf),
            arquivo_normas.name: sha256_arquivo(arquivo_normas),
        },
        "sem_pseudo_docs": True,
        "embedder_model": vetorizador.nome_modelo,
        "embedder_batch_size": getattr(vetorizador, "tamanho_lote", None),
        "embedder_device": device,
        "gpu": _nome_gpu(device),
        "embedder_dtype": "float32",   # o vetorizador da origem carrega o SentenceTransformer no dtype padrão (medido)
        "chunk_max_tokens": seg.max_tokens,
        "chunk_overlap_tokens": seg.sobreposicao_tokens,
        "tokenizador": "tokenizador_espacos",
        "documentos": {k: v for k, v in sorted(documentos.items())},
        "chunks": {k: v for k, v in sorted(por_tipo.items())},
        "n_chunks": len(chunks),
        "dim": int(vetorizador.dim),
        "construido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracao_s": round(time.perf_counter() - t0, 1),
    }
    salvar_metadados_indice_rag(saida, meta)
    return meta


SUFIXO_SUBINDICE = {"norma": "normas", "acordao_carf": "carf"}
"""tipo_fonte → sufixo do diretório do subíndice (rag2_reparado_normas, rag2_reparado_carf)."""


def dir_subindice(dir_reparado: Path, tipo: str) -> Path:
    """runs/_shared/rag/rag2_reparado_<sufixo> — irmão do índice reparado; o indice_id (metadados) carrega o mesmo nome, logo os caches não colidem."""
    dir_reparado = Path(dir_reparado)
    return dir_reparado.with_name(f"{dir_reparado.name}_{SUFIXO_SUBINDICE[tipo]}")


def dividir_indice(indice: IndiceHibrido) -> dict[str, IndiceHibrido]:
    """Dois subíndices EM MEMÓRIA a partir do reparado: filtra chunks/vetores por tipo_fonte (preservando a ordem densa) e RECONSTRÓI o BM25 de cada subconjunto (IDF separado, não máscara). Não persiste.

    A ordem preservada e o BM25 refeito são a diferença de R3 para uma máscara sobre o índice único:
    para algumas consultas o top-50 do índice completo não tem nenhum chunk de um tipo (medido). Os
    ids do BM25 ficam iguais aos densos porque o filtro preserva a ordem e o BM25 é construído nessa ordem.
    """
    from ensemble_llm.esquemas import BlocoDocumento
    from ensemble_llm.recuperacao.indice_bm25 import IndiceBM25

    tipos = indice.chunks["source_doc"].map(tipo_fonte)
    subs: dict[str, IndiceHibrido] = {}
    for tipo in ("norma", "acordao_carf"):
        mask = (tipos == tipo).to_numpy()
        chunks = indice.chunks[mask].reset_index(drop=True)
        blocos = [BlocoDocumento.model_validate(r) for r in _registros_chunks(chunks)]
        bm25 = IndiceBM25(remover_stopwords=indice.remover_stopwords)
        bm25.construir(blocos)
        subs[tipo] = IndiceHibrido(
            diretorio=dir_subindice(indice.diretorio, tipo),
            chunks=chunks,
            vetores=indice.vetores[mask],
            bm25=bm25._bm25,
            ids_bm25=chunks["chunk_id"].to_numpy(),
            remover_stopwords=indice.remover_stopwords,
        )
    return subs


def _registros_chunks(chunks) -> list[dict]:
    """Linhas de chunks.parquet como dicts de BlocoDocumento (section_path NaN → None; n_tokens int)."""
    import pandas as pd

    registros = []
    for r in chunks.to_dict("records"):
        registros.append(
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "source_doc": r["source_doc"],
                "section_path": None if r["section_path"] is None or pd.isna(r["section_path"]) else r["section_path"],
                "n_tokens": int(r["n_tokens"]),
            }
        )
    return registros


def construir_subindices(
    cfg: ConfigRag2, *, saida_base: Path | None = None, sobrescrever: bool = False, log: Callable[[str], None] = print
) -> dict[str, dict]:
    """Divide o índice reparado e persiste os dois subíndices no formato padrão (denso/, bm25/, index_metadata.json), registrar uma vez.

    Devolve os metadados por tipo. IndiceJaExiste (por subíndice) se já houver index_metadata.json e não houver `sobrescrever`.
    """
    import pickle

    import numpy as np

    base = Path(saida_base) if saida_base is not None else cfg.dir_indice_reparado()
    subs = dividir_indice(carregar_indice(base))
    metas: dict[str, dict] = {}
    for tipo, sub in subs.items():
        d = dir_subindice(base, tipo)
        if carregar_metadados_indice_rag(d) is not None and not sobrescrever:
            raise IndiceJaExiste(str(d))
        (d / "denso").mkdir(parents=True, exist_ok=True)
        (d / "bm25").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "denso" / "vetores.npz", vetores=sub.vetores)
        sub.chunks.to_parquet(d / "denso" / "chunks.parquet", index=False)
        with (d / "bm25" / "bm25.pkl").open("wb") as f:
            pickle.dump(
                {"bm25": sub.bm25, "chunks": _registros_chunks(sub.chunks), "remover_stopwords": sub.remover_stopwords}, f
            )
        meta = {
            "indice_id": f"{INDICE_REPARADO}_{SUFIXO_SUBINDICE[tipo]}",
            "derivado_de": INDICE_REPARADO,
            "tipo_fonte": tipo,
            "n_chunks": len(sub.chunks),
            "dim": int(sub.vetores.shape[1]),
            "remover_stopwords": sub.remover_stopwords,
        }
        salvar_metadados_indice_rag(d, meta)
        log(f"subíndice {tipo}: {d} | chunks={meta['n_chunks']} dim={meta['dim']} id={meta['indice_id']}")
        metas[tipo] = meta
    return metas


def resolver_indice(cfg: ConfigRag2, degrau: str) -> Path:
    """R0 → índice de produção (congelado, com pseudo-docs); qualquer outro degrau → índice reparado deste repositório."""
    return cfg.dir_indice_producao() if degrau == "R0" else cfg.dir_indice_reparado()


def comparar_atingibilidade(itens: Sequence[ItemAvaliado], antes: Casador, depois: Casador) -> dict[str, int]:
    """Pares (arquivo, artigo) com repetição entre itens — como em teto_oracular — atingíveis em cada índice.

    ganhos = inatingível antes e atingível depois; perdas = o inverso (deve ser 0 para um reparo que só acrescenta texto).
    """
    pares = [(a, b) for it in itens for a, b in it.esperados]
    a = [bool(antes.chunks_do_dispositivo(x, y)) for x, y in pares]
    d = [bool(depois.chunks_do_dispositivo(x, y)) for x, y in pares]
    return {
        "n_pares": len(pares),
        "atingiveis_antes": sum(a),
        "atingiveis_depois": sum(d),
        "ganhos": sum(1 for p, q in zip(a, d, strict=True) if not p and q),
        "perdas": sum(1 for p, q in zip(a, d, strict=True) if p and not q),
    }
