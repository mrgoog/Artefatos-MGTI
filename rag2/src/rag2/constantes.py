"""Hiperparâmetros pré-registrados da escada e da frente B (espelho de docs/pre-registro.md)."""

SEMENTE = 42
"""Semente única: amostras, bootstrap, agentes."""

# --- escada de recuperação (DESIGN §4) ---
TOP_K_PRODUCAO = 5          # R0: k do BM25 = k do denso = 5, união, dedup 1 chunk/doc
N_FINAL = 7                 # R1+: chunks servidos por item (média da produção: 7,05)
POOL_POR_RAMO = 50          # R1+: top-k de cada ramo antes da fusão
RRF_K = 60                  # constante do Reciprocal Rank Fusion
POOL_RERANK = 100           # R2+: chunks pós-RRF pontuados pelo reranker
RERANK_BATCH = 16           # batch do CrossEncoder (fp16)
COTA_NORMAS = 5             # R3: chunks normativos no contexto final
COTA_CARF = 2               # R3: chunks de acórdão no contexto final
BETA_CATEGORIA = 0.10       # R4: bônus por categoria C compatível com a categoria RFB da pergunta
GAMMA_VIGENCIA = 0.05       # R4: bônus por documento vigente
N_REFORMULACOES = 3         # R5: reformulações por pergunta (normativa, decomposta, literal)

# --- geração: frente B e confirmação (arch §4; configs/phase2_local_full.yaml da origem) ---
MAX_TOKENS_AGENTE = 1024    # agentes locais; entra no prompt_hash da resposta (igual à produção)
MAX_TOKENS_JUIZ = 1024      # juiz; entra no prompt_hash = chave do cache de rótulos (igual à produção)
MAX_TOKENS_AUDITOR = 4096   # auditor de recusas; valor de scripts/auditar_recusas.py (1024 truncava o JSON)
MAX_TOKENS_SUFICIENCIA = 4096   # juiz de suficiência de contexto; valor de scripts/juiz_suficiencia_contexto.py (1024 truncava o JSON)

# --- estatística (DESIGN §5) ---
B_BOOTSTRAP = 10_000
NIVEL_IC = 0.95
TOLERANCIA_CANONICO = 0.001  # |recall_passagem(R0) − 0,2351| aceito
CANONICO_R0 = 0.2351         # valor a reproduzir (reference/diagnostico_recall_pool.md §1)

# --- amostras ---
TAMANHO_AMOSTRA_B = 60               # frente B, estratificada por has_doc_ref, semente 42
TAMANHO_AMOSTRA_CONFIRMACAO = 200    # itens do piloto phase1_local_pilot_200_1

# --- modelos e runs (DESIGN §3.1) ---
MODELO_EMBEDDER = "BAAI/bge-m3"
MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
MODELO_REFORMULADOR = "qwen3.5-9b"
MODELO_JUIZ = "openai/gpt-5.4-mini"
AGENTES = {"qwen35_9b": "qwen3.5-9b", "granite41_8b": "granite4.1-8b", "gemma4_e4b": "gemma4-e4b"}
AGENTE_REFERENCIA = "qwen35_9b"      # agente cujo parquet fornece retrieved_chunk_ids e a lista dos 200
RUN_PRODUCAO = "phase2_local_full_715"
RUN_PILOTO = "phase1_local_pilot_200_1"
