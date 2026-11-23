"""GPU + LLM: R5 real pela CLI em subconjunto — reformulação por qwen3.5-9b, união multi-query, cota 5+2, caches _mq. Exige subíndices e llama-swap."""

import pandas as pd
import pytest
import torch

from rag2.config import RAIZ
from rag2.dados import tipo_fonte
from rag2.indices import dir_subindice

pytestmark = [pytest.mark.gpu, pytest.mark.llm]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_degrau_r5_limite(cfg, capsys):
    from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap, ErroLlamaSwap

    for tipo in ("norma", "acordao_carf"):
        if not (dir_subindice(cfg.dir_indice_reparado(), tipo) / "bm25" / "bm25.pkl").exists():
            pytest.skip("subíndices de R5 não construídos (rag2 construir-indice reparado --dividir)")
    try:
        ClienteLlamaSwap("qwen3.5-9b").validate_model()
    except ErroLlamaSwap:
        pytest.skip("llama-swap indisponível ou sem qwen3.5-9b")
    from rag2.cli import main

    assert main(["degrau", "R5", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert out.count("reformulacoes: ") == 2 and out.count("reranker: ") == 2
    assert "_mq.parquet" in out and "degrau=R5  itens=5" in out and "chunks/item=7.00" in out and "pseudo_doc" not in out
    assert "norma=0.714" in out and "acordao_carf=0.286" in out
    ref = pd.read_parquet(RAIZ / "runs" / "escada" / "R5" / "reformulacoes.parquet")
    assert set(ref.columns) >= {"consulta_hash", "normativa", "decomposta", "literal", "valid_json"}
    d = pd.read_parquet(RAIZ / "debug-temp" / "escada" / "R5" / "servidos.parquet")
    assert list(d.columns)[-2:] == ["rrf_score", "rerank_score"] and len(d) == 35
    comp = d.assign(t=d.source_doc.map(tipo_fonte)).groupby(["item_id", "t"]).size().unstack(fill_value=0)
    assert (comp["norma"] == 5).all() and (comp["acordao_carf"] == 2).all()
