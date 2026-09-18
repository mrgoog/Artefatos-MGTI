"""O instrumento reproduz o canônico item a item a partir dos artefatos da run de produção."""

import json

import pandas as pd
import pytest

from rag2.cli import main


@pytest.mark.lento
def test_metricas_run_reproduz_canonico(capsys, tmp_path):
    assert main(["metricas-run", "phase2_local_full_715", "--b", "200", "--saida", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "recall_passagem=0.2355  recall_arquivo=0.4531  hit_passagem=0.3813  frac_zero=0.6187" in out
    assert "max|Δ recall_passagem|=0.0000, max|Δ recall_arquivo|=0.0000" in out
    assert "teto_oracular=0.9145" in out
    assert len(pd.read_parquet(tmp_path / "por_item.parquet")) == 556
    assert json.loads((tmp_path / "resumo.json").read_text())["n_dispositivos"] == 1742
