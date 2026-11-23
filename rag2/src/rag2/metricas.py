"""Métricas de recuperação da dissertação sobre um contexto servido (DESIGN §5.1; recall_passagem.py)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rag2 import canonico
from rag2.constantes import B_BOOTSTRAP, SEMENTE
from rag2.dados import ItemAvaliado, contar_palavras

NOMES_METRICAS = ("recall_passagem", "recall_arquivo", "hit_passagem", "frac_zero")
"""As quatro métricas com IC; recall_passagem é a principal."""


class Casador:
    """Para um índice (chunk_id → texto e source_doc), resolve quais chunks satisfazem (arquivo, artigo)."""

    def __init__(self, texto: Mapping[str, str], doc: Mapping[str, str]) -> None:
        self.texto = texto
        self.doc = doc
        self.chunks_por_doc: dict[str, list[str]] = {}
        for cid, d in doc.items():
            self.chunks_por_doc.setdefault(d, []).append(cid)
        self._cache: dict[tuple[str, str], frozenset[str]] = {}

    def chunks_do_dispositivo(self, arquivo: str, artigo: str) -> frozenset[str]:
        """Chunk_ids do arquivo cujo texto casa o cabeçalho do artigo (regra do canônico); vazio se nenhum."""
        chave = (arquivo, artigo)
        if chave not in self._cache:
            rx = canonico.regex_artigo(artigo)
            self._cache[chave] = frozenset(
                c for c in self.chunks_por_doc.get(arquivo, []) if rx.search(self.texto[c])
            )
        return self._cache[chave]

    def palavras(self, chunk_ids: Iterable[str]) -> int:
        """Palavras (espaço em branco) do contexto servido; chunk desconhecido conta zero."""
        return sum(contar_palavras(self.texto[c]) for c in chunk_ids if c in self.texto)


@dataclass(frozen=True)
class ResultadoItem:
    """Métricas de um item para um contexto servido."""

    item_id: str
    n_esperados: int
    hits_arquivo: int
    hits_passagem: int
    n_chunks: int
    palavras: int

    @property
    def recall_arquivo(self) -> float:
        return self.hits_arquivo / self.n_esperados

    @property
    def recall_passagem(self) -> float:
        return self.hits_passagem / self.n_esperados

    def como_dict(self) -> dict:
        """Campos + os dois recalls, na ordem das colunas de por_item.parquet."""
        d = asdict(self)
        d["recall_arquivo"] = self.recall_arquivo
        d["recall_passagem"] = self.recall_passagem
        return d


def avaliar_item(item: ItemAvaliado, chunk_ids: Sequence[str], casador: Casador) -> ResultadoItem:
    """Definição do canônico: arquivo acerta se algum chunk servido vem dele; passagem se o chunk casa o artigo."""
    servidos = set(chunk_ids)
    docs = {casador.doc[c] for c in servidos if c in casador.doc}
    h_arq = h_pas = 0
    for arquivo, artigo in item.esperados:
        if arquivo in docs:
            h_arq += 1
            if casador.chunks_do_dispositivo(arquivo, artigo) & servidos:
                h_pas += 1
    return ResultadoItem(
        item_id=item.item_id,
        n_esperados=len(item.esperados),
        hits_arquivo=h_arq,
        hits_passagem=h_pas,
        n_chunks=len(chunk_ids),
        palavras=casador.palavras(chunk_ids),
    )


def avaliar_itens(
    itens: Sequence[ItemAvaliado], servidos: Mapping[str, Sequence[str]], casador: Casador
) -> pd.DataFrame:
    """Uma linha por item (ordem de `itens`); item ausente de `servidos` é avaliado com contexto vazio."""
    return pd.DataFrame(
        [avaliar_item(it, list(servidos.get(it.item_id, [])), casador).como_dict() for it in itens]
    )


def teto_oracular(itens: Sequence[ItemAvaliado], casador: Casador) -> float:
    """Fração dos dispositivos esperados (com repetição entre itens) que têm ao menos um chunk no índice."""
    pares = [(arq, art) for it in itens for arq, art in it.esperados]
    return float(np.mean([bool(casador.chunks_do_dispositivo(a, b)) for a, b in pares]))


def estatisticas_pontuais(df: pd.DataFrame) -> dict[str, float]:
    """Médias das métricas sobre o DataFrame de avaliar_itens."""
    return {
        "recall_passagem": float(df["recall_passagem"].mean()),
        "recall_arquivo": float(df["recall_arquivo"].mean()),
        "hit_passagem": float((df["recall_passagem"] > 0).mean()),
        "frac_zero": float((df["recall_passagem"] == 0).mean()),
        "n_chunks_medio": float(df["n_chunks"].mean()),
        "palavras_medias": float(df["palavras"].mean()),
        "n_itens": int(len(df)),
        "n_dispositivos": int(df["n_esperados"].sum()),
    }


_ESTATISTICAS = {
    "recall_passagem": lambda d: float(d["recall_passagem"].mean()),
    "recall_arquivo": lambda d: float(d["recall_arquivo"].mean()),
    "hit_passagem": lambda d: float((d["recall_passagem"] > 0).mean()),
    "frac_zero": lambda d: float((d["recall_passagem"] == 0).mean()),
}


def resumo(df: pd.DataFrame, *, b: int = B_BOOTSTRAP, semente: int = SEMENTE) -> dict:
    """Estatísticas pontuais + IC percentil das quatro métricas (chaves ic_<metrica> = [lo, hi]) + b_bootstrap."""
    from rag2.bootstrap import ic_percentil

    saida: dict = estatisticas_pontuais(df)
    for nome in NOMES_METRICAS:
        ic = ic_percentil(df, _ESTATISTICAS[nome], b=b, semente=semente)
        saida[f"ic_{nome}"] = [float(ic.ci_lo), float(ic.ci_hi)]
    saida["b_bootstrap"] = int(b)
    return saida


def linha_markdown(nome: str, r: Mapping) -> str:
    """Linha da tabela da escada: degrau, recall pass. [IC], recall arq., Hit@K, frac. zero, chunks, palavras."""
    lo, hi = r["ic_recall_passagem"]
    return (
        f"| {nome} | {r['recall_passagem']:.4f} [{lo:.4f}, {hi:.4f}] | {r['recall_arquivo']:.4f} "
        f"| {r['hit_passagem']:.4f} | {r['frac_zero']:.4f} | {r['n_chunks_medio']:.2f} "
        f"| {r['palavras_medias']:.0f} |"
    )


CABECALHO_MARKDOWN = (
    "| degrau | recall passagem [IC95] | recall arquivo | Hit@K | frac. zero | chunks/item | palavras/item |\n"
    "|---|---:|---:|---:|---:|---:|---:|"
)
