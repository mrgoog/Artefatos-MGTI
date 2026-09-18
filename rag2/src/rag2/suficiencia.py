"""Juiz de suficiência de contexto (arch §3): prompt, formatação do contexto e parser importados de scripts/juiz_suficiencia_contexto.py da origem (nunca reescritos); só o laço por item é reimplementado, porque o script lê caminhos fixos da origem."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pandas as pd
from ensemble_llm.agentes.contratos import ClienteLLM
from ensemble_llm.esquemas import ItemDataset

from rag2.config import ConfigRag2
from rag2.geracao import _ler_ou_vazio, _parquet_atomico   # helpers do harness: compartilhados, não copiados

COLUNAS_SUFICIENCIA = (
    "item_id", "has_doc_ref", "n_chunks", "contexto_vazio", "veredicto_suficiencia", "raciocinio_juiz", "cost_usd", "judge_model",
)
"""Schema de suficiencia_contexto.parquet — o mesmo de runs/<run>/metrics/ da origem (M5.2 lê os dois com o mesmo código); veredicto 2/1/0, -1 = parse falhou."""

_NOME = "_juiz_suficiencia_origem"
_MODULO: ModuleType | None = None


def modulo_suficiencia(cfg: ConfigRag2 | None = None) -> ModuleType:
    """Carrega uma única vez scripts/juiz_suficiencia_contexto.py da origem (só definições; main() não roda)."""
    global _MODULO
    if _MODULO is None:
        caminho = (cfg or ConfigRag2()).origem / "scripts" / "juiz_suficiencia_contexto.py"
        spec = importlib.util.spec_from_file_location(_NOME, caminho)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"juiz_suficiencia_contexto.py não encontrado em {caminho}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_NOME] = mod
        spec.loader.exec_module(mod)
        _MODULO = mod
    return _MODULO


def prompt_suficiencia(pergunta: str, contexto: str, ground_truth: str) -> tuple[str, str]:
    """(system, user) do juiz de suficiência com o template da origem; `contexto` é o texto que o agente viu."""
    m = modulo_suficiencia()
    usuario = m.TEMPLATE_USUARIO_SUFICIENCIA.format(pergunta=pergunta.strip(), contexto=contexto, ground_truth=ground_truth.strip())
    return m.PROMPT_SISTEMA_SUFICIENCIA, usuario


def formatar_contexto(chunk_ids: Sequence[str], mapa: Mapping[str, Mapping[str, str]]) -> str:
    """`_contexto` da origem ('[DOCUMENTO k — source_doc]' + texto); verificado idêntico a formatar_chunks_recuperados — o juiz vê o que o agente viu."""
    return modulo_suficiencia()._contexto(list(chunk_ids), mapa)


def parse_suficiencia(bruto: str) -> tuple[str, int]:
    """(raciocinio, veredicto) com o parser tolerante a truncamento da origem; veredicto -1 se irrecuperável."""
    return modulo_suficiencia()._parse(bruto)


def julgar_suficiencia(
    itens: Sequence[ItemDataset],
    contexto: pd.DataFrame,
    cliente: ClienteLLM,
    caminho: Path,
    *,
    log: Callable[[str], None] = print,
    flush_every: int = 10,
) -> pd.DataFrame:
    """Um veredicto por item (o contexto é o mesmo para os três agentes) sobre o contexto congelado; retomável por item.

    Regras do script da origem: contexto vazio → veredicto 0 sem chamada; parse falho → -1, registrado, não aborta.
    Devolve todas as linhas do parquet (as dos itens dados incluídas), na ordem do arquivo.
    """
    atual = _ler_ou_vazio(caminho, COLUNAS_SUFICIENCIA)
    feitos = set(atual["item_id"])
    linhas: list[dict] = atual.to_dict("records")
    mapa = {r.chunk_id: {"source_doc": r.source_doc, "text": r.text} for r in contexto.itertuples(index=False)}
    por_item: dict[str, list[str]] = {}
    for r in contexto.sort_values(["item_id", "combined_rank"]).itertuples(index=False):
        por_item.setdefault(r.item_id, []).append(r.chunk_id)
    faltam = [it for it in itens if it.item_id not in feitos]
    n_vazios, custo = 0, 0.0
    for i, it in enumerate(faltam, 1):
        ids = por_item.get(it.item_id, [])
        if not ids:
            raciocinio, veredicto, c = "(contexto vazio — nenhum documento recuperado)", 0, 0.0
            n_vazios += 1
        else:
            system, user = prompt_suficiencia(it.pergunta, formatar_contexto(ids, mapa), it.ground_truth)
            res = cliente.invoke(user, system_prompt=system)
            raciocinio, veredicto = parse_suficiencia(res.raw_output or "")   # content None da API → caminho -1 (não derruba a corrida)
            c = float(res.cost_usd)
            custo += c
        linhas.append(
            {
                "item_id": it.item_id,
                "has_doc_ref": bool(it.has_doc_ref),
                "n_chunks": len(ids),
                "contexto_vazio": not ids,
                "veredicto_suficiencia": int(veredicto),
                "raciocinio_juiz": raciocinio,
                "cost_usd": c,
                "judge_model": cliente.model,
            }
        )
        if i % flush_every == 0 or i == len(faltam):
            _parquet_atomico(_tipar(pd.DataFrame(linhas, columns=list(COLUNAS_SUFICIENCIA))), caminho)
    df = _tipar(pd.DataFrame(linhas, columns=list(COLUNAS_SUFICIENCIA)))
    if not caminho.exists():
        _parquet_atomico(df, caminho)
    log(
        f"suficiencia: {caminho} | julgados={len(faltam) - n_vazios} vazios={n_vazios} em_cache={len(itens) - len(faltam)} "
        f"custo=US$ {custo:.2f} modelo={cliente.model}"
    )
    return df


def _tipar(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos do parquet da origem (bool/int64/float64), estáveis entre a 1ª gravação e a releitura."""
    return df.astype({"has_doc_ref": bool, "n_chunks": "int64", "contexto_vazio": bool, "veredicto_suficiencia": "int64", "cost_usd": "float64"})
