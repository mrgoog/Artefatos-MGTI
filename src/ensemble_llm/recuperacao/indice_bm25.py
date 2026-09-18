"""Índice BM25 lexical via rank-bm25."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ensemble_llm.esquemas import BlocoDocumento

logger = logging.getLogger(__name__)

_STOP_WORDS_PT = frozenset(
    [
        "a", "o", "as", "os", "um", "uma", "uns", "umas",
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "por", "para", "com", "sem", "sob", "sobre", "ante", "perante",
        "e", "ou", "mas", "porém", "se", "que", "como", "quando",
        "é", "são", "foi", "será", "ser", "estar", "ter", "haver",
        "este", "esta", "isto", "esse", "essa", "isso", "aquele",
        "aquela", "aquilo", "seu", "sua", "seus", "suas",
        "ao", "à", "aos", "às",
    ]
)

_PADRAO_PONTUACAO = re.compile(r"[^\w\s]", flags=re.UNICODE)


def tokenizar_pt(texto: str, *, remover_stopwords: bool = True) -> list[str]:
    """Tokeniza texto em português: caixa baixa, remove pontuação e stopwords opcionalmente."""
    limpo = _PADRAO_PONTUACAO.sub(" ", texto.lower())
    tokens = limpo.split()
    if remover_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS_PT and len(t) > 1]
    return tokens


class IndiceBM25:
    """Índice lexical BM25Okapi para busca por correspondência de palavras-chave em PT."""

    def __init__(self, *, remover_stopwords: bool = True) -> None:
        self.remover_stopwords = remover_stopwords
        self._chunks: list[BlocoDocumento] = []
        self._bm25: BM25Okapi | None = None

    def construir(self, chunks: list[BlocoDocumento]) -> None:
        """Tokeniza e indexa os chunks no modelo BM25Okapi."""
        if not chunks:
            raise ValueError("chunks não pode ser vazio")
        corpus_tokenizado = [
            tokenizar_pt(c.text, remover_stopwords=self.remover_stopwords) for c in chunks
        ]
        self._bm25 = BM25Okapi(corpus_tokenizado)
        self._chunks = list(chunks)

    def buscar(self, consulta: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retorna os top-k chunk_ids com maior escore BM25 para a consulta."""
        if self._bm25 is None:
            raise RuntimeError("Índice não construído. Chame .construir() primeiro.")
        if top_k < 1:
            raise ValueError(f"top_k deve ser >= 1, recebeu {top_k}")

        tokens_consulta = tokenizar_pt(consulta, remover_stopwords=self.remover_stopwords)
        if not tokens_consulta:
            return []

        escores = self._bm25.get_scores(tokens_consulta)
        k = min(top_k, len(self._chunks))
        top_idx = sorted(range(len(escores)), key=lambda i: -escores[i])[:k]
        return [(self._chunks[i].chunk_id, float(escores[i])) for i in top_idx]

    def salvar(self, caminho: Path) -> None:
        """Serializa o índice BM25 e os chunks em pickle."""
        if self._bm25 is None:
            raise RuntimeError("Índice não construído")
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        with (caminho / "bm25.pkl").open("wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "chunks": [c.model_dump() for c in self._chunks],
                    "remover_stopwords": self.remover_stopwords,
                },
                f,
            )

    def carregar(self, caminho: Path) -> None:
        """Restaura índice BM25 e chunks a partir do pickle salvo em disco."""
        caminho = Path(caminho)
        with (caminho / "bm25.pkl").open("rb") as f:
            dados = pickle.load(f)
        self._bm25 = dados["bm25"]
        self._chunks = [BlocoDocumento.model_validate(d) for d in dados["chunks"]]
        self.remover_stopwords = dados["remover_stopwords"]

    def __len__(self) -> int:
        return len(self._chunks)
