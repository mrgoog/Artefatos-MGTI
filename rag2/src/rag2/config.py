"""Caminhos e parâmetros de ambiente do RAG-2, lidos de `.env` na raiz do repositório (prefixo RAG2_)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag2.constantes import RUN_PILOTO, RUN_PRODUCAO

RAIZ = Path(__file__).resolve().parents[2]
"""Raiz deste repositório (diretório que contém src/, specs/, runs/)."""


class ConfigRag2(BaseSettings):
    """Localização do repositório de origem, dos dados brutos e do índice de produção."""

    model_config = SettingsConfigDict(
        env_file=RAIZ / ".env", env_file_encoding="utf-8", env_prefix="RAG2_", extra="ignore"
    )

    origem: Path = Field(
        default=RAIZ.parent,
        description="Raiz do repositório (código ensemble_llm e runs de referência); rag2/ fica dentro dela",
    )
    dados: Path | None = Field(
        default=None, description="Diretório com os três JSON do BR-TaxQA-R; padrão origem/data/brtaxq"
    )
    indice_producao: Path | None = Field(
        default=None,
        description="Índice br_taxqa_r_bge-m3_cuda1 (bm25/ + denso/); padrão origem/runs/_shared/rag/...",
    )
    device: str = Field(default="cuda:0", description="Device do embedder e do reranker")

    def dir_dados(self) -> Path:
        """Diretório dos JSON brutos."""
        return self.dados if self.dados is not None else self.origem / "data" / "brtaxq"

    def dir_indice_producao(self) -> Path:
        """Diretório do índice híbrido de produção."""
        if self.indice_producao is not None:
            return self.indice_producao
        return self.origem / "runs" / "_shared" / "rag" / "br_taxqa_r_bge-m3_cuda1"

    def dir_indice_reparado(self) -> Path:
        """Índice híbrido reconstruído por M1.2 (corpus reparado, sem pseudo-docs); runs/_shared/ deste repositório (gitignored)."""
        return RAIZ / "runs" / "_shared" / "rag" / "rag2_reparado"

    def arquivo_questoes(self) -> Path:
        """questions_QA_2024_v1.1.json (715 perguntas)."""
        return self.dir_dados() / "questions_QA_2024_v1.1.json"

    def arquivo_normas(self) -> Path:
        """referred_legal_documents_QA_2024_v1.1.json (478 documentos normativos)."""
        return self.dir_dados() / "referred_legal_documents_QA_2024_v1.1.json"

    def arquivo_carf(self) -> Path:
        """acordaos_CARF_2023.json (7.204 acórdãos)."""
        return self.dir_dados() / "acordaos_CARF_2023.json"

    def dir_run(self, run_id: str) -> Path:
        """runs/<run_id>/ do repositório de origem (somente leitura)."""
        return self.origem / "runs" / run_id

    def script_recall_passagem(self) -> Path:
        """scripts/recall_passagem.py da origem — fonte das definições canônicas."""
        return self.origem / "scripts" / "recall_passagem.py"

    def faltantes(self) -> list[Path]:
        """Caminhos obrigatórios ausentes do disco; lista vazia significa ambiente completo."""
        indice = self.dir_indice_producao()
        obrigatorios = [
            self.origem / "src" / "ensemble_llm",
            self.script_recall_passagem(),
            self.arquivo_questoes(),
            self.arquivo_normas(),
            self.arquivo_carf(),
            indice / "bm25" / "bm25.pkl",
            indice / "denso" / "chunks.parquet",
            indice / "denso" / "vetores.npz",
            self.dir_run(RUN_PRODUCAO) / "agent_responses",
            self.dir_run(RUN_PILOTO) / "agent_responses",
        ]
        return [p for p in obrigatorios if not p.exists()]
