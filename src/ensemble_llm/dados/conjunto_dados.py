"""Loader do dataset BR-TaxQA-R."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ensemble_llm.esquemas import ItemDataset

logger = logging.getLogger(__name__)


def _para_string(valor: Any) -> str:
    """Normaliza str, dict ou list do JSON para uma string limpa e legível."""
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, dict):
        for chave in ("text", "content", "texto", "titulo", "ementa"):
            if chave in valor and isinstance(valor[chave], str):
                return valor[chave].strip()
        return json.dumps(valor, ensure_ascii=False)
    if isinstance(valor, list):
        return "\n".join(_para_string(v) for v in valor if v is not None)
    return str(valor)


def _extrair_referencias(dados_questao: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Classifica referências do item em dispositivos legais e ementas CARF pela presença de termos-chave."""
    candidatos: list[Any] = []
    for chave in (
        "all_formatted_references",
        "formatted_references",
        "embedded_references",
        "references",
    ):
        if chave in dados_questao and dados_questao[chave]:
            valor = dados_questao[chave]
            if isinstance(valor, list):
                candidatos.extend(valor)
            else:
                candidatos.append(valor)
            break

    dispositivos: list[str] = []
    ementas: list[str] = []
    for ref in candidatos:
        texto = _para_string(ref).strip()
        if not texto:
            continue
        if "carf" in texto.lower() or "acórdão" in texto.lower() or "acordao" in texto.lower():
            ementas.append(texto)
        else:
            dispositivos.append(texto)
    return dispositivos, ementas


def _construir_item(dados_questao: dict[str, Any]) -> ItemDataset:
    """Constrói ItemDataset tipado a partir de um dict bruto do JSON do dataset."""
    numero_q = dados_questao.get("question_number")
    item_id = f"q_{int(numero_q):04d}" if numero_q is not None else f"q_unknown_{id(dados_questao)}"

    pergunta = _para_string(dados_questao.get("question_text", ""))
    if not pergunta:
        pergunta = _para_string(dados_questao.get("question_summary", ""))

    ground_truth_bruto = dados_questao.get("answer_cleaned") or dados_questao.get("answer", "")
    ground_truth = _para_string(ground_truth_bruto)

    dispositivos, ementas = _extrair_referencias(dados_questao)
    has_doc_ref = bool(dispositivos or ementas)

    return ItemDataset(
        item_id=item_id,
        pergunta=pergunta,
        ground_truth=ground_truth,
        dispositivos_legais=dispositivos,
        ementas_carf=ementas,
        has_doc_ref=has_doc_ref,
    )


def carregar_br_taxqa_r(
    caminho_json_questoes: str | Path,
    *,
    tamanho_amostra: int | None = None,
    semente_amostra: int = 42,
) -> list[ItemDataset]:
    """Carrega o dataset BR-TaxQA-R de um JSON, com amostragem estratificada opcional."""
    caminho = Path(caminho_json_questoes)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de questões não encontrado: {caminho}")

    with caminho.open("r", encoding="utf-8") as f:
        dados_brutos = json.load(f)

    if isinstance(dados_brutos, list):
        questoes = dados_brutos
    elif isinstance(dados_brutos, dict):
        questoes = dados_brutos.get("questions") or list(dados_brutos.values())
    else:
        raise ValueError(f"Formato inesperado em {caminho}: {type(dados_brutos)}")

    itens: list[ItemDataset] = []
    for q in questoes:
        if not isinstance(q, dict):
            logger.warning("Pulando entrada não-dict: %r", q)
            continue
        try:
            itens.append(_construir_item(q))
        except Exception:
            logger.exception("Erro ao processar questão %s", q.get("question_number"))
            continue

    if tamanho_amostra is not None and tamanho_amostra < len(itens):
        itens = _amostra_estratificada(itens, tamanho_amostra, semente_amostra)

    return itens


def _amostra_estratificada(
    itens: list[ItemDataset],
    tamanho_amostra: int,
    semente: int,
) -> list[ItemDataset]:
    """Amostra de tamanho_amostra itens preservando a proporção com/sem has_doc_ref."""
    import numpy as np

    rng = np.random.default_rng(semente)
    com_ref = [i for i in itens if i.has_doc_ref]
    sem_ref = [i for i in itens if not i.has_doc_ref]

    total = len(itens)
    n_com = round(tamanho_amostra * len(com_ref) / total)
    n_sem = tamanho_amostra - n_com

    idx_com = rng.choice(len(com_ref), size=min(n_com, len(com_ref)), replace=False)
    idx_sem = rng.choice(len(sem_ref), size=min(n_sem, len(sem_ref)), replace=False)

    amostra = [com_ref[i] for i in idx_com] + [sem_ref[i] for i in idx_sem]
    rng.shuffle(amostra)
    return amostra


def construir_corpus_de_itens(itens: list[ItemDataset]) -> dict[str, str]:
    """Monta corpus RAG com os dispositivos legais e ementas CARF extraídos dos itens do dataset."""
    corpus: dict[str, str] = {}
    textos_vistos: set[str] = set()

    for item in itens:
        for i, disp in enumerate(item.dispositivos_legais):
            if disp in textos_vistos:
                continue
            textos_vistos.add(disp)
            corpus[f"disp_{item.item_id}_{i:02d}"] = disp
        for i, ementa in enumerate(item.ementas_carf):
            if ementa in textos_vistos:
                continue
            textos_vistos.add(ementa)
            corpus[f"carf_{item.item_id}_{i:02d}"] = ementa

    return corpus


def _carregar_json_corpus(path: Path) -> dict[str, str]:
    """Parser defensivo para um único arquivo JSON de corpus; tenta múltiplos esquemas comuns."""
    with path.open("r", encoding="utf-8") as f:
        dados = json.load(f)

    corpus: dict[str, str] = {}
    stem = path.stem

    _CAMPOS_ID = ("id", "numero_acordao", "number", "filename", "doc_id", "key", "acórdão")
    _CAMPOS_TEXTO = ("text", "texto", "ementa", "content", "conteudo", "filedata", "body", "decision")

    if isinstance(dados, list):
        for i, entrada in enumerate(dados):
            if not isinstance(entrada, dict):
                continue
            doc_id = next(
                (str(entrada[c]) for c in _CAMPOS_ID if c in entrada),
                f"{stem}_{i:05d}",
            )
            texto = next(
                (_para_string(entrada[c]) for c in _CAMPOS_TEXTO if c in entrada),
                _para_string(entrada),
            )
            if texto.strip():
                corpus[doc_id] = texto

    elif isinstance(dados, dict):
        if "filename" in dados and "filedata" in dados:
            corpus[str(dados["filename"])] = _para_string(dados["filedata"])
        else:
            for k, v in dados.items():
                corpus[k] = _para_string(v)

    if not corpus:
        logger.warning("Nenhum documento extraído de %s — verifique o esquema do arquivo.", path)
    else:
        logger.info("Corpus de %s: %d documentos carregados.", path.name, len(corpus))

    return corpus


def carregar_documentos_legais(caminhos: list[str | Path]) -> dict[str, str]:
    """Carrega e mescla corpus RAG a partir de arquivos JSON específicos (acordãos, dispositivos).

    Recebe uma lista de caminhos de arquivos, não um diretório — para evitar carregar
    acidentalmente o arquivo de questões caso ambos estejam em data/.
    """
    corpus: dict[str, str] = {}
    for caminho in caminhos:
        path = Path(caminho)
        if not path.exists():
            logger.warning("Arquivo de corpus não encontrado, pulando: %s", path)
            continue
        corpus.update(_carregar_json_corpus(path))
    return corpus
