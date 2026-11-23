"""GPU + LLM + API: variante P0 real pela CLI em --limite 2 sobre R3 — contexto, 3 agentes, juiz (cache pré-carregado), auditor, resumo em debug-temp/."""

import pandas as pd
import pytest
import torch

from rag2.config import RAIZ
from rag2.indices import dir_subindice

pytestmark = [pytest.mark.gpu, pytest.mark.llm]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_variante_p0_limite(cfg, capsys):
    from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap, ErroLlamaSwap
    from ensemble_llm.configuracao import Segredos
    from pydantic import ValidationError

    for tipo in ("norma", "acordao_carf"):
        if not (dir_subindice(cfg.dir_indice_reparado(), tipo) / "bm25" / "bm25.pkl").exists():
            pytest.skip("subíndices de R3 não construídos")
    try:
        Segredos()   # type: ignore[call-arg]
    except ValidationError:
        pytest.skip("OPENROUTER_API_KEY ausente")
    try:
        for m in ("qwen3.5-9b", "granite4.1-8b", "gemma4-e4b"):
            ClienteLlamaSwap(m).validate_model()
    except ErroLlamaSwap:
        pytest.skip("llama-swap indisponível ou sem os três agentes")
    from rag2.cli import main

    assert main(["variante", "P0", "--degrau", "R3", "--limite", "2", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert out.count("agente ") == 3 and "juiz: " in out and "auditor: " in out and "variante=P0  degrau=R3  itens=2  respostas=6" in out
    saida = RAIZ / "debug-temp" / "prompts" / "P0"
    c = pd.read_parquet(saida / "contexto.parquet")
    assert len(c) == 14 and c.groupby("item_id").size().eq(7).all()
    for aid in ("qwen35_9b", "granite41_8b", "gemma4_e4b"):
        r = pd.read_parquet(saida / "agent_responses" / aid / "responses.parquet")
        assert len(r) == 2 and r.agent_id.eq(aid).all() and r.prompt_hash.str.len().eq(64).all()
    assert len(pd.read_parquet(saida / "rotulos.parquet")) == 6
    assert (RAIZ / "runs" / "_shared" / "judge_labels" / "openai_gpt-5.4-mini" / "cache.parquet").exists()
