"""Confere a resolução de caminhos e a presença dos insumos externos."""

from pathlib import Path

from rag2.config import RAIZ, ConfigRag2


def test_raiz_e_o_repositorio():
    assert (RAIZ / "pyproject.toml").exists()
    assert (RAIZ / "src" / "rag2").is_dir()


def test_ambiente_completo(cfg):
    assert cfg.faltantes() == [], f"ambiente incompleto: {cfg.faltantes()}"


def test_override_por_variavel_de_ambiente(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG2_DADOS", str(tmp_path))
    assert ConfigRag2().dir_dados() == Path(tmp_path)
