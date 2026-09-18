"""GPU: R3 real pela CLI em subconjunto — dois subíndices, cota 5+2, dois caches de reranker; exige `rag2 construir-indice reparado --dividir`."""

import pandas as pd
import pytest
import torch

from rag2.config import RAIZ
from rag2.dados import tipo_fonte
from rag2.indices import dir_subindice

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_degrau_r3_limite(cfg, capsys):
    for tipo in ("norma", "acordao_carf"):
        if not (dir_subindice(cfg.dir_indice_reparado(), tipo) / "bm25" / "bm25.pkl").exists():
            pytest.skip("subíndices de R3 não construídos (rag2 construir-indice reparado --dividir)")
    from rag2.cli import main

    assert main(["degrau", "R3", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert out.count("reranker: ") == 2 and "rag2_reparado_normas" in out and "rag2_reparado_carf" in out
    assert "degrau=R3  itens=5" in out and "chunks/item=7.00" in out and "pseudo_doc" not in out
    assert "norma=0.714" in out and "acordao_carf=0.286" in out
    d = pd.read_parquet(RAIZ / "debug-temp" / "escada" / "R3" / "servidos.parquet")
    assert list(d.columns)[-2:] == ["rrf_score", "rerank_score"] and len(d) == 35
    comp = d.assign(t=d.source_doc.map(tipo_fonte)).groupby(["item_id", "t"]).size().unstack(fill_value=0)
    assert (comp["norma"] == 5).all() and (comp["acordao_carf"] == 2).all()
