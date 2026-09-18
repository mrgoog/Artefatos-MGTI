"""R5: reformulação multi-query da pergunta por LLM local (qwen3.5-9b), com cache versionado (registrar uma vez)."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from ensemble_llm.agentes.contratos import ClienteLLM
from ensemble_llm.agentes.estruturado import invocar_estruturado
from pydantic import BaseModel

from rag2.constantes import N_REFORMULACOES
from rag2.pool import hash_consulta

MAX_TOKENS_REFORMULACAO = 512
"""Teto de tokens da reformulação; entra no prompt_hash (procedência). A sondagem (q longas, até 501 chars) coube em 512."""

COLUNAS = ("consulta_hash", "pergunta", "normativa", "decomposta", "literal", "prompt_hash", "valid_json")
"""Esquema do parquet versionado; uma linha por pergunta original. valid_json=False → as três variantes repetem a pergunta (degradação registrada, não erro fatal)."""

PROMPT_SISTEMA_REFORMULACAO = (
    "Você reformula perguntas sobre o Imposto de Renda da Pessoa Física (IRPF) "
    "para melhorar a busca de dispositivos legais e ementas do CARF. "
    "Você NÃO responde à pergunta; apenas a reescreve de três formas. "
    "Não invente fatos, números, artigos ou nomes de normas que não estejam na pergunta."
)

TEMPLATE_USUARIO_REFORMULACAO = (
    "PERGUNTA ORIGINAL:\n{pergunta}\n\n"
    "Gere três reformulações desta mesma pergunta, cada uma otimizada para recuperação:\n"
    "1. normativa: reescreva usando vocabulário técnico-normativo tributário "
    "(termos que apareceriam na legislação e em ementas), preservando o sentido.\n"
    "2. decomposta: decomponha a pergunta em suas subperguntas essenciais, "
    "em uma única string separada por ' | '.\n"
    "3. literal: uma paráfrase literal e concisa da pergunta, sem jargão.\n\n"
    "Responda EXCLUSIVAMENTE em JSON válido, sem texto antes ou depois, sem markdown:\n"
    '{{"normativa": "...", "decomposta": "...", "literal": "..."}}'
)

HINT_REFORMULACAO = '{"normativa": "texto", "decomposta": "texto | texto", "literal": "texto"}'


class ReformulacaoQuery(BaseModel):
    """As três reformulações de uma pergunta (uma chamada estruturada por item)."""

    normativa: str
    decomposta: str
    literal: str


def caminho_reformulacoes(dir_saida: Path) -> Path:
    """runs/escada/R5/reformulacoes.parquet — golden versionado, ao lado do resumo do degrau."""
    return Path(dir_saida) / "reformulacoes.parquet"


class CacheReformulacoes:
    """Reformulações persistidas num parquet (COLUNAS), carregado inteiro; chave = hash_consulta da pergunta original."""

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self._por_hash: dict[str, tuple[str, str, str]] = {}
        if self.caminho.exists():
            df = pd.read_parquet(self.caminho)
            for r in df.itertuples(index=False):
                self._por_hash[r.consulta_hash] = (r.normativa, r.decomposta, r.literal)

    def __len__(self) -> int:
        return len(self._por_hash)

    def __contains__(self, consulta: str) -> bool:
        return hash_consulta(consulta) in self._por_hash

    def faltantes(self, consultas: Sequence[str]) -> list[str]:
        """Perguntas (únicas por hash, na ordem dada) ainda não reformuladas."""
        vistos: set[str] = set()
        saida: list[str] = []
        for c in consultas:
            h = hash_consulta(c)
            if h not in self._por_hash and h not in vistos:
                vistos.add(h)
                saida.append(c)
        return saida

    def variantes(self, consulta: str) -> tuple[str, str, str]:
        """(normativa, decomposta, literal) da pergunta; KeyError se ausente."""
        return self._por_hash[hash_consulta(consulta)]

    def adicionar(self, df: pd.DataFrame) -> int:
        """Acrescenta as perguntas de df ainda ausentes e regrava (tmp + os.replace); devolve quantas entraram."""
        novos = df[~df["consulta_hash"].isin(list(self._por_hash))]
        n = int(novos["consulta_hash"].nunique())
        if n == 0:
            return 0
        for r in novos.itertuples(index=False):
            self._por_hash[r.consulta_hash] = (r.normativa, r.decomposta, r.literal)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        atual = pd.read_parquet(self.caminho) if self.caminho.exists() else None
        tudo = novos if atual is None else pd.concat([atual, novos], ignore_index=True)
        tmp = self.caminho.with_name(self.caminho.name + ".tmp")
        tudo.to_parquet(tmp, index=False)
        os.replace(tmp, self.caminho)
        return n


class Reformulador:
    """Gera as três reformulações por pergunta via um ClienteLLM, com cache versionado (registrar uma vez). Nunca vê ground truth (arch §2)."""

    def __init__(
        self,
        cliente: ClienteLLM,
        cache: CacheReformulacoes,
        *,
        n_reformulacoes: int = N_REFORMULACOES,
        log: Callable[[str], None] = print,
    ) -> None:
        if n_reformulacoes != 3:
            raise ValueError(f"o prompt fixo produz exatamente 3 reformulações; recebeu {n_reformulacoes}")
        self.cliente = cliente
        self.cache = cache
        self.log = log

    def _gerar(self, pergunta: str) -> tuple[tuple[str, str, str], str, bool]:
        """Uma chamada estruturada; em falha de parse, degrada as três variantes para a própria pergunta (registrado, não fatal)."""
        r = invocar_estruturado(
            self.cliente,
            TEMPLATE_USUARIO_REFORMULACAO.format(pergunta=pergunta.strip()),
            ReformulacaoQuery,
            system_prompt=PROMPT_SISTEMA_REFORMULACAO,
            schema_hint=HINT_REFORMULACAO,
            k_max=3,
        )
        ok = bool(r.valid_json) and r.parsed is not None
        v = (r.parsed.normativa, r.parsed.decomposta, r.parsed.literal) if ok else (pergunta, pergunta, pergunta)
        if any(not x.strip() for x in v):
            v = (pergunta, pergunta, pergunta)   # parse inválido OU campo vazio: degrada a single-query, registrado
            ok = False
        return v, r.prompt_hash, ok

    def preaquecer(self, consultas: Sequence[str]) -> int:
        """Gera as reformulações faltantes (uma chamada por pergunta), persistindo a cada bloco de 25; devolve quantas gerou."""
        faltantes = self.cache.faltantes(consultas)
        linhas: list[tuple] = []
        n_falhas = 0
        for i, pergunta in enumerate(faltantes, 1):
            (norm, dec, lit), ph, ok = self._gerar(pergunta)
            n_falhas += 0 if ok else 1
            linhas.append((hash_consulta(pergunta), pergunta, norm, dec, lit, ph, ok))
            if i % 25 == 0 or i == len(faltantes):
                self.cache.adicionar(pd.DataFrame(linhas, columns=list(COLUNAS)))
                linhas = []
        self.log(
            f"reformulacoes: {self.cache.caminho} | geradas={len(faltantes)} "
            f"em_cache={len(consultas) - len(faltantes)} falhas_parse={n_falhas} modelo={self.cliente.model}"
        )
        return len(faltantes)

    def reformular(self, consulta: str) -> tuple[str, str, str]:
        """As três reformulações da pergunta (gera e cacheia se faltar)."""
        if consulta not in self.cache:
            self.preaquecer([consulta])
        return self.cache.variantes(consulta)
