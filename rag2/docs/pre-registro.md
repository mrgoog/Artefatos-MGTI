# Pré-registro — protótipo RAG-2 (BR-TaxQA-R)

Data: 2026-09-12. Congelado antes de rodar qualquer degrau acima de R0.
Mudanças posteriores só com entrada datada em `docs/decisions.md` e são
declaradas no relatório como desvios do pré-registro. Fonte: `reference/DESIGN.md`.

## 1. Pergunta e hipóteses

Quanto cada intervenção de recuperação (R1–R5) e de prompt (P1–P3) move as
métricas da dissertação, mantendo fixos dataset, agentes, juiz, auditor e
cabeça de decisão? Hipóteses ordenadas pelo diagnóstico
(`reference/diagnostico_recall_pool.md`): (H1) relaxar a dedup e reranquear um
pool de 100 recupera a maior parte dos 63 % de itens zerados por ordenação;
(H2) a cota por tipo de fonte corrige a dominância dos acórdãos CARF (63–81 %
dos slots, 0 % dos dispositivos esperados); (H3) o reparo do corpus eleva o
teto oracular acima de 0,9145; (H4) metadados taxonômicos como boost dão
ganho pequeno; (H5) multi-query ataca a classe "pool" (36,8 %).

## 2. População e amostras

- Escada: os 556 itens com dispositivo legal identificável (1.742 pares
  arquivo–artigo), derivados de `dispositivos_esperados` do canônico.
- Frente B: 60 itens, `_amostra_estratificada(itens, 60, semente=42)`.
- Confirmação: os 200 `item_id` de `runs/phase1_local_pilot_200_1` (151 com
  dispositivo). Baseline = subconjunto desses 200 na run `phase2_local_full_715`,
  sem reexecução.

## 3. Escada (uma mudança por degrau)

| degrau | mudança única |
|---|---|
| R0 | produção: BM25 k=5 + denso k=5, união, dedup 1 chunk/doc, índice de produção |
| R0' | R0 sobre o índice reconstruído com o corpus normativo reparado, sem pseudo-docs |
| R1 | pool 50+50, RRF k=60, sem dedup por documento, corte em N=7 |
| R2 | R1 + reranker cross-encoder sobre 100 chunks pós-RRF, corte em N=7 |
| R3 | R2 + índices por tipo de fonte, cota 5 normas + 2 CARF |
| ~~R4~~ | ~~R3 + boost taxonômico (β categoria, γ vigência) no score do reranker~~ — NÃO EXECUTADO (desvio 2026-09-12, docs/decisions.md) |
| R5 | **R3** + 3 reformulações por LLM local, união dos pools antes do reranker (desvio 2026-09-12: sai de R3, não de R4; cota 5+2 e corte 7 mantidos) |

## 4. Hiperparâmetros fixados a priori

| constante | valor |
|---|---|
| TOP_K_PRODUCAO | 5 |
| N_FINAL | 7 |
| POOL_POR_RAMO | 50 |
| RRF_K | 60 |
| POOL_RERANK | 100 |
| RERANK_BATCH | 16 |
| COTA_NORMAS | 5 |
| COTA_CARF | 2 |
| BETA_CATEGORIA | 0.1 |
| GAMMA_VIGENCIA | 0.05 |
| N_REFORMULACOES | 3 |
| B_BOOTSTRAP | 10000 |
| NIVEL_IC | 0.95 |
| TOLERANCIA_CANONICO | 0.001 |
| CANONICO_R0 | 0.2351 |
| TAMANHO_AMOSTRA_B | 60 |
| TAMANHO_AMOSTRA_CONFIRMACAO | 200 |
| SEMENTE | 42 |
| MODELO_EMBEDDER | BAAI/bge-m3 |
| MODELO_RERANKER | BAAI/bge-reranker-v2-m3 |
| MODELO_REFORMULADOR | qwen3.5-9b |
| MODELO_JUIZ | openai/gpt-5.4-mini |

Reranker em fp16, score sigmoide em [0, 1]; β e γ somados a esse score.
Nenhum hiperparâmetro é otimizado nos 556 itens; se houver otimização, é em
folds e declarada.

## 5. Métricas

Recuperação (556 itens): recall de passagem (principal), recall de arquivo,
Hit@K de passagem, fração de itens com recall zero, palavras de contexto por
item (contagem por espaço em branco), teto oracular. IC bootstrap percentil,
B = 10.000, semente 42. Geração (60 e 200 itens): adequação z (juiz binário
`openai/gpt-5.4-mini`, mesma rubrica e cache), recusa textual (`PADRAO_RECUSA`),
veredicto do auditor (2/1/0), falha de parse JSON, suficiência de contexto;
bootstrap pareado por item, estratificado por `has_doc_ref`.

## 6. Critérios de decisão

- Melhor degrau de A: maior recall de passagem cujo IC95 não inclua o do
  degrau anterior; empate mantém o degrau anterior (mais simples). Palavras
  por item são reportadas junto e um ganho que multiplique o contexto por
  mais de 2× é discutido, não promovido automaticamente.
- Promoção na frente B: adequação média dos três agentes ≥ P0 e taxa de
  recusa evasiva (auditor = 1) ≤ P0; empate resolve por P0. Sem correção de
  multiplicidade — a frente B é exploratória; a confirmação é a §2.3 do DESIGN.
- Confirmação N = 200: comparação pareada pipeline atual × novo, IC95
  bootstrap estratificado, B = 10.000, nas métricas da §5 mais custo
  (palavras, chamadas de LLM, tempo de GPU).

## 7. Vazamento e vieses declarados

Os 715 pares pergunta–resposta e `linked_questions` nunca entram em índice
nem em prompt de recuperação. Pseudo-documentos `disp_*`/`carf_*` ficam só em
R0. Taxonomias RFB/C são geradas por LLM e não validadas por dupla
codificação: entram como boost, nunca como filtro. A amostra de 60 é pequena
e a de 200 é a do piloto, não uma nova amostra. O reparo do corpus (R0')
completa documentos escolhidos a partir dos dispositivos esperados do próprio
conjunto de teste; é um viés declarado a favor do pipeline novo.
