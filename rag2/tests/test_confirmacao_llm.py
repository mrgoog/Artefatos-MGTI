"""GPU + LLM + API: `rag2 confirmacao executar --limite 2` real sobre R3 — contexto, 3 agentes, juiz, auditor, suficiência, custo.json em debug-temp/."""

import json

import pandas as pd
import pytest
import torch

from rag2.config import RAIZ
from rag2.indices import dir_subindice

pytestmark = [pytest.mark.gpu, pytest.mark.llm]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_confirmacao_executar_limite(cfg, capsys):
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

    assert main(["confirmacao", "executar", "--degrau", "R3", "--limite", "2", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert out.count("agente ") == 3 and "juiz: " in out and "auditor: " in out and "suficiencia: " in out
    assert "confirmacao=novo  variante=P0  degrau=R3  itens=2  respostas=6" in out and "custo desta execução: US$" in out
    saida = RAIZ / "debug-temp" / "confirmacao" / "novo"
    c = pd.read_parquet(saida / "contexto.parquet")
    assert len(c) == 14 and c.groupby("item_id").size().eq(7).all() and c.item_id.tolist()[0] == "q_0004"
    s = pd.read_parquet(saida / "suficiencia_contexto.parquet")
    assert len(s) == 2 and s.veredicto_suficiencia.isin([2, 1, 0, -1]).all() and s.n_chunks.eq(7).all()
    assert len(pd.read_parquet(saida / "rotulos.parquet")) == 6 and "suficiencia" in pd.read_parquet(saida / "por_item.parquet").columns
    custo = json.loads((saida / "custo.json").read_text(encoding="utf-8"))
    # idempotente (--limite retoma parquets já congelados: julgados pode ser 0 no rerun): estrutura, não contagens desta execução; a cobertura dos 2 itens está no parquet acima (len(s) == 2)
    assert custo["n_itens"] == 2 and list(custo["etapas"]) == ["contexto", "agente_qwen35_9b", "agente_granite41_8b", "agente_gemma4_e4b", "juiz", "auditor", "suficiencia"]
