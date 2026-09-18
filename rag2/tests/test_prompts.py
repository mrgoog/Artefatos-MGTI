"""P0 é o prompt da origem byte a byte: mesmo par system/user e, sobre os chunks servidos em produção, o mesmo prompt_hash gravado; o juiz reproduz a chave do cache da produção."""

import pandas as pd
from ensemble_llm.agentes.cliente_llama_swap import _hash_prompt
from ensemble_llm.agentes.cliente_openrouter import _hash_prompt as hash_openrouter
from ensemble_llm.agentes.prompts_agentes import (
    construir_prompt_agente,
    formatar_chunks_recuperados,
)
from ensemble_llm.esquemas import BlocoRecuperado, RespostaAgente
from ensemble_llm.juiz.rubrica import construir_prompt_juiz

from rag2.constantes import AGENTES, MAX_TOKENS_AGENTE, MAX_TOKENS_JUIZ, MODELO_JUIZ, RUN_PRODUCAO
from rag2.dados import carregar_itens
from rag2.prompts import VARIANTES, VarianteP0, VariantePrompt

BLOCOS = [
    BlocoRecuperado(chunk_id="Lei nº 1.txt#0000", text=" Art. 1º Texto. ", source_doc="Lei nº 1.txt", combined_rank=1),
    BlocoRecuperado(chunk_id="10000_1.txt#0000", text="Acórdão.", source_doc="10000_1.txt", combined_rank=2),
]


def test_p0_delega_a_origem():
    v = VarianteP0()
    assert isinstance(v, VariantePrompt) and VARIANTES["P0"] is VarianteP0 and v.nome == "P0"
    assert v.construir("Q?", BLOCOS) == construir_prompt_agente("Q?", BLOCOS)
    assert v.contexto(BLOCOS) == formatar_chunks_recuperados(BLOCOS)
    assert "[DOCUMENTO 1 — Lei nº 1.txt]\nArt. 1º Texto." in v.contexto(BLOCOS)
    assert v.schema is RespostaAgente and {"resposta", "confianca"} <= set(v.schema.model_fields)


def test_p0_contexto_vazio():
    system, user = VarianteP0().construir("Q?", [])
    assert "(Nenhum documento recuperado para esta pergunta.)" in user and system.strip().startswith("Você é um assistente")


def test_p0_reproduz_prompt_hash_da_producao(cfg):
    """Oráculo: com os retrieved_chunk_ids e os textos do índice de produção, P0 gera o prompt_hash gravado na run de produção."""
    r = pd.read_parquet(cfg.dir_run(RUN_PRODUCAO) / "agent_responses" / "qwen35_9b" / "responses.parquet")
    linha = r[r.item_id == "q_0001"].iloc[0]
    chunks = pd.read_parquet(cfg.dir_indice_producao() / "denso" / "chunks.parquet", columns=["chunk_id", "source_doc", "text"]).set_index("chunk_id")
    blocos = [
        BlocoRecuperado(chunk_id=c, text=chunks.loc[c, "text"], source_doc=chunks.loc[c, "source_doc"], combined_rank=k)
        for k, c in enumerate(linha.retrieved_chunk_ids, 1)
    ]
    item = next(it for it in carregar_itens(cfg) if it.item_id == "q_0001")
    system, user = VarianteP0().construir(item.pergunta, blocos)
    assert _hash_prompt(system, user, AGENTES["qwen35_9b"], 0.0, MAX_TOKENS_AGENTE) == linha.prompt_hash


def test_juiz_reproduz_prompt_hash_da_producao(cfg):
    """Oráculo: a resposta de produção de q_0001/qwen35_9b, por construir_prompt_juiz + _hash_prompt do OpenRouter (max_tokens=1024), dá o prompt_hash do cache do juiz (arch §4, mesmo cache)."""
    r = pd.read_parquet(cfg.dir_run(RUN_PRODUCAO) / "agent_responses" / "qwen35_9b" / "responses.parquet")
    linha = r[r.item_id == "q_0001"].iloc[0]
    item = next(it for it in carregar_itens(cfg) if it.item_id == "q_0001")
    system, user = construir_prompt_juiz(
        question=item.pergunta, ground_truth=item.ground_truth, dispositivos_legais=item.dispositivos_legais,
        ementas_carf=item.ementas_carf, candidate_response=str(linha.response_text),
    )
    cache = pd.read_parquet(cfg.dir_run(RUN_PRODUCAO) / "judge_labels" / MODELO_JUIZ.replace("/", "_") / "cache.parquet")
    esperado = cache[(cache.item_id == "q_0001") & (cache.response_source == "qwen35_9b")].prompt_hash.iloc[0]
    assert hash_openrouter(system, user, MODELO_JUIZ, 0.0, MAX_TOKENS_JUIZ) == esperado
