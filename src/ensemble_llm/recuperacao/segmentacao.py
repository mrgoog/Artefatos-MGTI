"""
Segmentação sensível a parágrafos para construção do corpus RAG.

Estratégia:
- Tamanho máximo: 512 tokens (limite do bge-m3)
- Sobreposição: 64 tokens
- Separação primária por `\n\n` (parágrafos)
- Fallback por sentença se um parágrafo exceder 512 tokens
- Metadados preservados: source_doc, section_path

O `tokenizador` é injetado como dependência (callable str -> int), permitindo:
- Em produção: usar o tokenizador do bge-m3
- Em testes: usar split por espaços (proxy rápido)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ensemble_llm.esquemas import BlocoDocumento


Tokenizador = Callable[[str], int]
"""Função que retorna o número de tokens de um texto."""


def tokenizador_espacos(texto: str) -> int:
    """Tokenizador simples por whitespace, para testes e fallback."""
    return len(texto.split())


@dataclass(frozen=True)
class ConfigSegmentacao:
    """Configuração da segmentação."""

    max_tokens: int = 512
    sobreposicao_tokens: int = 64
    separador_paragrafos: str = "\n\n"


_PADRAO_SENTENCA = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _dividir_sentencas(texto: str) -> list[str]:
    """Divide texto em sentenças usando padrão de pontuação seguida de maiúscula."""
    sentencas = _PADRAO_SENTENCA.split(texto)
    return [s.strip() for s in sentencas if s.strip()]


def _dividir_paragrafos(texto: str, separador: str) -> list[str]:
    """Divide texto em parágrafos pelo separador configurado (padrão: dupla quebra de linha)."""
    paragrafos = texto.split(separador)
    return [p.strip() for p in paragrafos if p.strip()]


def _agrupar_em_janelas(
    unidades: list[str],
    tokenizador: Tokenizador,
    max_tokens: int,
    sobreposicao_tokens: int,
) -> list[str]:
    """Agrupa unidades de texto em janelas de até max_tokens com sobreposição configurada."""
    if not unidades:
        return []

    janelas: list[str] = []
    atual: list[str] = []
    tokens_atuais = 0

    for unidade in unidades:
        tokens_unidade = tokenizador(unidade)

        if tokens_unidade > max_tokens:
            if atual:
                janelas.append(" ".join(atual))
                atual = []
                tokens_atuais = 0
            janelas.append(unidade)
            continue

        if tokens_atuais + tokens_unidade > max_tokens and atual:
            janelas.append(" ".join(atual))
            unidades_sobrepostas: list[str] = []
            contagem_sobreposicao = 0
            for unidade_anterior in reversed(atual):
                tokens_anteriores = tokenizador(unidade_anterior)
                if contagem_sobreposicao + tokens_anteriores > sobreposicao_tokens:
                    break
                unidades_sobrepostas.insert(0, unidade_anterior)
                contagem_sobreposicao += tokens_anteriores
            atual = unidades_sobrepostas
            tokens_atuais = contagem_sobreposicao

        atual.append(unidade)
        tokens_atuais += tokens_unidade

    if atual:
        janelas.append(" ".join(atual))

    return janelas


def _subsegmentar_excedente(
    texto: str,
    tokenizador: Tokenizador,
    max_tokens: int,
    sobreposicao_tokens: int,
) -> list[str]:
    """Subsegmenta parágrafo excedente por sentenças; usa split por palavras como fallback."""
    sentencas = _dividir_sentencas(texto)
    if not sentencas:
        palavras = texto.split()
        partes = []
        n_palavras_por_parte = int(max_tokens * 0.75)
        passo = n_palavras_por_parte - int(sobreposicao_tokens * 0.75)
        for i in range(0, len(palavras), max(1, passo)):
            partes.append(" ".join(palavras[i : i + n_palavras_por_parte]))
        return partes
    return _agrupar_em_janelas(sentencas, tokenizador, max_tokens, sobreposicao_tokens)


def segmentar_documento(
    texto: str,
    *,
    source_doc: str,
    section_path: str | None = None,
    tokenizador: Tokenizador = tokenizador_espacos,
    config: ConfigSegmentacao | None = None,
) -> list[BlocoDocumento]:
    """Quebra um documento em `BlocoDocumento`."""
    cfg = config or ConfigSegmentacao()
    if not texto or not texto.strip():
        return []

    paragrafos = _dividir_paragrafos(texto, cfg.separador_paragrafos)
    janelas = _agrupar_em_janelas(
        paragrafos, tokenizador, cfg.max_tokens, cfg.sobreposicao_tokens
    )

    blocos_finais: list[str] = []
    for janela in janelas:
        if tokenizador(janela) > cfg.max_tokens:
            blocos_finais.extend(
                _subsegmentar_excedente(
                    janela,
                    tokenizador,
                    cfg.max_tokens,
                    cfg.sobreposicao_tokens,
                )
            )
        else:
            blocos_finais.append(janela)

    return [
        BlocoDocumento(
            chunk_id=f"{source_doc}#{i:04d}",
            text=texto_bloco,
            source_doc=source_doc,
            section_path=section_path,
            n_tokens=tokenizador(texto_bloco),
        )
        for i, texto_bloco in enumerate(blocos_finais)
    ]
