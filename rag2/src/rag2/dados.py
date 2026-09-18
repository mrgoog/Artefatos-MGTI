"""Itens do BR-TaxQA-R, dispositivos esperados, amostras fixas e classificação de documentos-fonte."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ensemble_llm.dados.conjunto_dados import _amostra_estratificada, carregar_br_taxqa_r
from ensemble_llm.esquemas import ItemDataset

from rag2 import canonico
from rag2.config import ConfigRag2
from rag2.constantes import AGENTE_REFERENCIA, RUN_PILOTO, SEMENTE

PADRAO_ACORDAO = re.compile(r"^\d[\d_]*(\.txt)?$")
"""source_doc de acórdão CARF: só dígitos e sublinhados (ex.: 10865722385201112_6949202.txt)."""


@dataclass(frozen=True)
class ItemAvaliado:
    """Item com dispositivo legal identificável e seus pares (arquivo, artigo) esperados."""

    item: ItemDataset
    esperados: frozenset[tuple[str, str]]

    @property
    def item_id(self) -> str:
        """Atalho para item.item_id."""
        return self.item.item_id

    @property
    def pergunta(self) -> str:
        """Atalho para item.pergunta (a consulta servida aos recuperadores)."""
        return self.item.pergunta


def carregar_itens(cfg: ConfigRag2) -> list[ItemDataset]:
    """Os 715 itens, na ordem do JSON (item_id = q_NNNN), pelo loader da origem."""
    return carregar_br_taxqa_r(cfg.arquivo_questoes())


def itens_com_dispositivo(itens: list[ItemDataset]) -> list[ItemAvaliado]:
    """Os 556 itens com has_doc_ref e ao menos um par (arquivo, artigo) — a população da escada."""
    saida: list[ItemAvaliado] = []
    for it in itens:
        esperados = canonico.dispositivos_esperados(it.dispositivos_legais + it.ementas_carf)
        if it.has_doc_ref and esperados:
            saida.append(ItemAvaliado(item=it, esperados=frozenset(esperados)))
    return saida


def amostra_estratificada(itens: list[ItemDataset], n: int, semente: int = SEMENTE) -> list[ItemDataset]:
    """Amostra estratificada por has_doc_ref, idêntica à da origem (n=200, semente 42 reproduz o piloto)."""
    return _amostra_estratificada(itens, n, semente)


def ids_amostra_200(cfg: ConfigRag2) -> list[str]:
    """Lista ordenada dos 200 item_id do piloto — a amostra fixa da confirmação (DESIGN §2.3)."""
    caminho = cfg.dir_run(RUN_PILOTO) / "agent_responses" / AGENTE_REFERENCIA / "responses.parquet"
    return sorted(pd.read_parquet(caminho, columns=["item_id"])["item_id"].unique().tolist())


def tipo_fonte(source_doc: str) -> str:
    """'pseudo_doc' (disp_/carf_ do dataset), 'acordao_carf' (nome numérico) ou 'norma'."""
    if source_doc.startswith(("disp_", "carf_")):
        return "pseudo_doc"
    if PADRAO_ACORDAO.match(source_doc):
        return "acordao_carf"
    return "norma"


def carregar_normas(caminho: Path) -> list[dict[str, str]]:
    """Lista de {'filename', 'filedata'} do JSON de documentos normativos (478 no original)."""
    with Path(caminho).open(encoding="utf-8") as f:
        dados = json.load(f)
    return [{"filename": d["filename"], "filedata": d["filedata"]} for d in dados]


def contar_palavras(texto: str) -> int:
    """Contagem por espaço em branco — a medida de 'palavras de contexto' deste projeto."""
    return len(texto.split())
