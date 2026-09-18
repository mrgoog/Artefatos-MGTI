# Números canônicos — RAG-2 (apêndice)

**Regra de uso:** nenhuma cifra entra em `docs/relatorio_apendice.md` sem constar deste arquivo (arch §5). Se um número precisar existir e não estiver aqui, ele é primeiro apurado de um artefato versionado, registrado aqui com procedência, e só então escrito no texto — nunca o caminho inverso.

**Estado:** **gerado** por `uv run rag2 numeros` a partir dos artefatos versionados em `runs/` e `data/`; não edite à mão — regenere. `uv run rag2 numeros --conferir docs/relatorio_apendice.md` confere que toda cifra com 4 casas do relatório está aqui e que as cinco tabelas de §11 aparecem nele verbatim. As cifras do braço de API (§10) são exploratórias: conferíveis, mas fora do pré-registro.

**Legenda de estado:**

| símbolo | significado |
|---|---|
| ✅ | apurado de artefato versionado neste repositório; reproduzível por `rag2 numeros` |
| ⚠️ | apurado, mas diverge do que o relatório publica — exige edição |
| ❌ | citado e sem lastro — não usar |

**Duas armadilhas que este arquivo existe para evitar:** (1) há **duas definições de "palavras"** — ver §8; (2) a comparação atual × novo (§6) **não é evidência causal** — deriva de runtime dos agentes, decisão B3 de 2026-09-13 (`docs/decisions.md`); o apêndice a publica com esse caveat.

---

## 1. Desenho experimental

| grandeza | valor | estado | procedência |
|---|---|---|---|
| itens do BR-TaxQA-R | 715 | ✅ | `questions_QA_2024_v1.1.json` (origem) |
| itens com dispositivo identificável (escada) | 556 | ✅ | `runs/escada/R0/resumo.json` |
| dispositivos esperados (pares arquivo–artigo) | 1.742 | ✅ | `runs/escada/R0/resumo.json` |
| amostra da confirmação (piloto) | 200 (168 has_doc_ref; 151 com dispositivo) | ✅ | `runs/confirmacao/comparacao.json`; `phase1_local_pilot_200_1` (origem) |
| frente B (60 itens) | 60 — **não executada** (decisão 2026-09-13) | ✅ | `docs/decisions.md` |
| agentes | qwen35_9b (qwen3.5-9b), granite41_8b (granite4.1-8b), gemma4_e4b (gemma4-e4b) | ✅ | `src/rag2/constantes.py`; `manifest.yaml` da produção |
| juiz, auditor e juiz de suficiência | `openai/gpt-5.4-mini`, temperatura 0 | ✅ | `src/rag2/constantes.py` |
| embedder / reranker / reformulador | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` / `qwen3.5-9b` | ✅ | `src/rag2/constantes.py` |
| hiperparâmetros pré-registrados | pool 50+50; RRF k = 60; rerank 100; cota 5 normas + 2 CARF; N_FINAL = 7; reformulações 3 | ✅ | `src/rag2/constantes.py` ≡ `docs/pre-registro.md` §4 |
| bootstrap | percentil, B = 10.000, semente 42, estratificado por has_doc_ref na geração | ✅ | `src/rag2/constantes.py` |
| índice reparado | 67.127 chunks (6.692 normas + 60.435 CARF), sem pseudo-docs | ✅ | `runs/escada/R0'/indice_metadata.json` |

## 2. Canônico e R0

| grandeza | valor | estado | procedência |
|---|---|---|---|
| recall de passagem canônico (diagnóstico) | 0,2351 | ✅ | `src/rag2/constantes.py` (`CANONICO_R0`); `reference/diagnostico_recall_pool.md` §1 |
| R0 pela cadeia nova (mesmo índice de produção) | 0,2355 [0,2072; 0,2658] | ✅ | `runs/escada/R0/resumo.json` |
| Δ R0 − canônico | +0,0004 (tolerância 0,001) | ✅ | `runs/escada/R0/resumo.json` |
| produção recomputada dos artefatos (`metricas-run`) | 0,2355 | ✅ | `runs/escada/producao_phase2_local_full_715/resumo.json` |

## 3. Escada de recuperação (556 itens)

| grandeza | valor | estado | procedência |
|---|---|---|---|
| R0 — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,2355 [0,2072; 0,2658]; 0,4531; 0,3813; 0,6187; 7,05; 3.200; 0,9145 | ✅ | `runs/escada/R0/resumo.json` |
| R0' — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,2366 [0,2078; 0,2669]; 0,4563; 0,3813; 0,6187; 7,04; 3.196; 0,9615 | ✅ | `runs/escada/R0'/resumo.json` |
| R1 — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,2397 [0,2107; 0,2703]; 0,3912; 0,3903; 0,6097; 7,00; 3.205; 0,9615 | ✅ | `runs/escada/R1/resumo.json` |
| R2 — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,2521 [0,2219; 0,2836]; 0,3849; 0,3957; 0,6043; 7,00; 3.372; 0,9615 | ✅ | `runs/escada/R2/resumo.json` |
| R3 — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,3426 [0,3106; 0,3754]; 0,5708; 0,5468; 0,4532; 7,00; 3.487; 0,9615 | ✅ | `runs/escada/R3/resumo.json` |
| R5 — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto | 0,3381 [0,3064; 0,3711]; 0,5664; 0,5396; 0,4604; 7,00; 3.481; 0,9615 | ✅ | `runs/escada/R5/resumo.json` |
| R4 | não executado (decisão 2026-09-12); β = 0,10 e γ = 0,05 sem uso | ✅ | `docs/decisions.md`; `docs/pre-registro.md` §3 |
| pares esperados que passam a casar com o reparo | 82 | ✅ | `data/reparado/relatorio_reparo.md` |
| melhor degrau de A (critério §6) | R3 | ✅ | `docs/decisions.md` 2026-09-13; §4 |

## 4. Hipóteses (critério pré-registrado §6: o IC95 do degrau não inclui o ponto do anterior)

| hipótese | o que testa | resultado | veredito |
|---|---|---|---|
| H1 | relaxar a dedup e reranquear (R1, R2) recupera os itens zerados por ordenação | R1 0,2397 [0,2107; 0,2703]: IC inclui R0 (0,2355); R2 0,2521 [0,2219; 0,2836]: IC inclui R1 (0,2397) | não confirmada |
| H2 | a cota por tipo de fonte (R3) corrige a dominância dos acórdãos | R3 0,3426 [0,3106; 0,3754]: IC exclui R2 (0,2521); itens sem dispositivo 0,6043 → 0,4532 | confirmada |
| H3 | o reparo do corpus eleva o teto oracular acima de 0,9145 | teto oracular 0,9145 → 0,9615 (82 pares passam a casar); R0' 0,2366 [0,2078; 0,2669] | confirmada |
| H4 | metadados taxonômicos como boost (R4) dão ganho pequeno | R4 não executado (decisão 2026-09-12; M2.1–M2.3 abandonados) | não testada |
| H5 | multi-query (R5) ataca a classe "pool" | R5 0,3381 [0,3064; 0,3711]: IC inclui R3 (0,3426) | não confirmada |


## 5. Geração no pipeline novo — P0 sobre R3, N = 200

| grandeza | valor | estado | procedência |
|---|---|---|---|
| adequação média (3 agentes) [IC95] | 0,2300 [0,1883; 0,2733] | ✅ | `runs/confirmacao/novo/resumo.json` |
| adequação qwen35_9b | 0,2300 (46/200) | ✅ | `runs/confirmacao/novo/resumo.json` |
| adequação granite41_8b | 0,2300 (46/200) | ✅ | `runs/confirmacao/novo/resumo.json` |
| adequação gemma4_e4b | 0,2300 (46/200) | ✅ | `runs/confirmacao/novo/resumo.json` |
| recusa textual [IC95] | 0,0667 [0,0450; 0,0917] | ✅ | `runs/confirmacao/novo/resumo.json` |
| recusa evasiva (auditor = 1) [IC95] | 0,0183 [0,0083; 0,0300] | ✅ | `runs/confirmacao/novo/resumo.json` |
| falha de parse | 0,0100 | ✅ | `runs/confirmacao/novo/resumo.json` |
| suficiência de contexto: suficiente [IC95]; parcial; insuficiente; parse falho | 69 (0,3450 [0,2800; 0,4100]); 115; 16; 0 | ✅ | `runs/confirmacao/novo/resumo.json` |
| palavras/item; chunks/item | 3.584; 7,00 | ✅ | `runs/confirmacao/novo/resumo.json` |
| smoke test de M4.1 (20 itens) — **mecanismo, não resultado** | 0,2333 [0,1167; 0,3667] | ✅ | `runs/prompts/P0/resumo.json`; `docs/decisions.md` 2026-09-13 |
| P1–P3 | não construídos (condicionais, a jusante do N = 200) | ✅ | `docs/decisions.md` 2026-09-12 |

## 6. Comparação pareada atual × novo (N = 200; recall nos 151 com dispositivo)

| grandeza | valor | estado | procedência |
|---|---|---|---|
| recall de passagem | atual 0,2137; novo 0,3308; Δ +0,1171 [+0,0712; +0,1661]; n = 151 | ✅ | `runs/confirmacao/comparacao.json` |
| recall de arquivo | atual 0,4140; novo 0,5652; Δ +0,1512 [+0,0946; +0,2095]; n = 151 | ✅ | `runs/confirmacao/comparacao.json` |
| adequação média (3 agentes) | atual 0,2267; novo 0,2300; Δ +0,0033 [−0,0350; +0,0417]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| adequação qwen35_9b | atual 0,2400; novo 0,2300; Δ −0,0100 [−0,0700; +0,0500]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| adequação granite41_8b | atual 0,2050; novo 0,2300; Δ +0,0250 [−0,0400; +0,0900]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| adequação gemma4_e4b | atual 0,2350; novo 0,2300; Δ −0,0050 [−0,0651; +0,0600]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| adequação B0 (melhor agente fixo) | atual 0,2400; novo 0,2300; Δ −0,0100 [−0,0700; +0,0500]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| recusa textual | atual 0,0617; novo 0,0667; Δ +0,0050 [−0,0150; +0,0267]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| recusa evasiva (auditor = 1) | atual 0,0133; novo 0,0183; Δ +0,0050 [−0,0067; +0,0183]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| falha de parse | atual 0,0083; novo 0,0100; Δ +0,0017 [−0,0083; +0,0117]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| contexto suficiente (= 2) | atual 0,2500; novo 0,3450; Δ +0,0950 [+0,0450; +0,1450]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| contexto insuficiente (= 0) | atual 0,1050; novo 0,0800; Δ −0,0250 [−0,0700; +0,0150]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| palavras/item | atual 3.264,6800; novo 3.583,8700; Δ +319,1900 [+149,9739; +492,6401]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| chunks/item | atual 7,1200; novo 7,0000; Δ −0,1200 [−0,4100; +0,1650]; n = 200 | ✅ | `runs/confirmacao/comparacao.json` |
| B0 (melhor agente fixo, escolhido na produção) | qwen35_9b | ✅ | `runs/confirmacao/comparacao.json` |
| condição de revisão B3 (IC de Δ adequação média exclui 0 por cima?) | não | ✅ | `runs/confirmacao/comparacao.json`; `docs/decisions.md` 2026-09-13 |

## 7. Custo nos 200

| grandeza | valor | estado | procedência |
|---|---|---|---|
| atual: agentes — chamadas; US$ | 600; 0,0000 | ✅ | `runs/confirmacao/comparacao.json` |
| atual: juiz — chamadas; US$ | 595; 0,8854 | ✅ | `runs/confirmacao/comparacao.json` |
| atual: auditor — chamadas; US$ | 37; 0,1771 | ✅ | `runs/confirmacao/comparacao.json` |
| atual: suficiencia — chamadas; US$ | 200; 1,0442 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: agentes — chamadas; US$ | 600; 0,0000 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: juiz — chamadas; US$ | 582; 0,8503 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: auditor — chamadas; US$ | 40; 0,2037 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: suficiencia — chamadas; US$ | 200; 1,0950 | ✅ | `runs/confirmacao/comparacao.json` |
| atual: total US$ | 2,1067 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: total US$ | 2,1490 | ✅ | `runs/confirmacao/comparacao.json` |
| novo: segundos de parede por etapa | contexto 136; agente_qwen35_9b 869; agente_granite41_8b 1.402; agente_gemma4_e4b 678; juiz 1.240; auditor 104; suficiencia 535 | ✅ | `runs/confirmacao/comparacao.json` (de `runs/confirmacao/novo/custo.json`) |
| atual: segundos de parede | não registrados na produção | ✅ | — |
| palavras/item; chunks/item | atual 3.264,7; 7,12 · novo 3.583,9; 7,00 | ✅ | `runs/confirmacao/comparacao.json` |

## 8. Duas definições de "palavras"

| grandeza | valor | estado | procedência |
|---|---|---|---|
| palavras de contexto por item — este projeto (R0 = produção) | 3.200 = `contar_palavras` (split por espaço em branco do texto dos chunks servidos) | ✅ | `runs/escada/R0/resumo.json` |
| palavras de contexto por item — canônico antigo | 5.009 = coluna `tokens_aprox` de `varredura_recuperacao.parquet` | ✅ | `../runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/metrics/varredura_recuperacao.parquet` |
| regra | definições diferentes para o mesmo contexto: **nunca comparar uma com a outra**; o apêndice usa só a primeira | ✅ | `docs/tooling-and-debugging.md` |

## 9. Desvios do pré-registro e vieses declarados

| grandeza | valor | estado | procedência |
|---|---|---|---|
| R4 não executado; M2.1–M2.3 abandonados; H4 não testada | 2026-09-12 | ✅ | `docs/decisions.md`; `specs/progress.md` |
| R5 construído sobre R3 (não R4), cota 5+2 e corte 7 mantidos | 2026-09-12 | ✅ | `docs/decisions.md` |
| parâmetros do reformulador fixados (temp 0, seed 42, max_tokens 512) | 2026-09-12 | ✅ | `docs/decisions.md` |
| frente B (60) não executada; M4.1 = smoke de 20 itens; P1–P3 não construídos | 2026-09-13 | ✅ | `docs/decisions.md` |
| MAX_TOKENS_AUDITOR = 4096 e MAX_TOKENS_SUFICIENCIA = 4096, seed 42 (valores dos scripts da origem) | 2026-09-13 / 2026-09-14 | ✅ | `docs/decisions.md` |
| comparabilidade: deriva de runtime dos agentes (1/9 respostas idênticas com o mesmo prompt); sem braço-controle P0×R0; comparação fora da dissertação (B3 waived) | 2026-09-13; condição de revisão não disparada em M5.2 | ✅ | `docs/decisions.md`; §6 |
| config do llama-swap mudou de sha entre M4.1 e M5.1; flags dos três agentes idênticas | 2026-09-14 | ✅ | `docs/decisions.md` |
| reparo do corpus escolhe documentos a partir dos dispositivos esperados do conjunto de teste — viés a favor do novo | pré-registro §7 | ✅ | `docs/pre-registro.md`; `data/reparado/relatorio_reparo.md` |
| taxonomias RFB/C não validadas por dupla codificação | sem uso: R4 não executado | ✅ | `docs/pre-registro.md` §7 |
| amostra de 200 é a do piloto, não uma amostra nova; 49 itens sem dispositivo (sem recall) | pré-registro §2/§7 | ✅ | `runs/confirmacao/comparacao.json` |

## 10. Braço de agentes de API sobre R3 — **exploratório, fora do pré-registro** (N = 200)

Etiqueta: cifras apuradas de artefato e conferíveis, mas de um experimento fora do pré-registro (decisão 2026-09-14) — peso inferencial de exploração, não de confirmação. Contexto e suficiência reusados byte a byte de `novo`; agentes de API com `max_tokens` 4096 (piloto da origem) contra 1024 dos locais: comparação de configurações de geração sob contexto fixo.

| grandeza | valor | estado | procedência |
|---|---|---|---|
| agentes de API | claude-haiku-4.5, gemini-3.1-flash-lite, gpt-5.4-nano | ✅ | `runs/confirmacao/novo_api/run_api.py` |
| adequação média: produção; locais@R3; API@R3 | 0,2267; 0,2300; 0,2750 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| piloto de API sobre a recuperação antiga (média dos três agentes) | 0,245 (0,260; 0,240; 0,235) | ✅ | `../runs/phase1_api_pilot_200_1/judge_labels/openai_gpt-5.4-mini/cache.parquet` (média de z por agente) |
| adequação claude-haiku-4.5 | 0,2800 (56/200) | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação gemini-3.1-flash-lite | 0,2500 (50/200) | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação gpt-5.4-nano | 0,2950 (59/200) | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| Δ API@R3 − locais@R3 (mesmo contexto; agente e orçamento de saída variam) [IC95]; n | +0,0450 [+0,0050; +0,0850]; 200 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| Δ locais@R3 − produção (≡ §6, adequação média) [IC95]; n | +0,0033 [−0,0350; +0,0417]; 200 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| Δ API@R3 − produção (as duas mudanças) [IC95]; n | +0,0483 [+0,0067; +0,0917]; 200 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| Δ API@R3 − locais@R3 dentro do estrato suficiente (reamostragem simples) [IC95]; n | +0,0193 [−0,0628; +0,0966]; 69 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação com contexto suficiente: API; local; n; contribuição ao Δ global | 0,4783; 0,4589; 69; +0,0067 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação com contexto parcial: API; local; n; contribuição ao Δ global | 0,1884; 0,1217; 115; +0,0383 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação com contexto insuficiente: API; local; n; contribuição ao Δ global | 0,0208; 0,0208; 16; +0,0000 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação por agente no estrato suficiente (API) | claude-haiku-4.5 0,5072; gemini-3.1-flash-lite 0,4493; gpt-5.4-nano 0,4783 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| adequação por agente no estrato suficiente (locais) | qwen35_9b 0,4783; granite41_8b 0,3913; gemma4_e4b 0,5072 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| recusas textuais: API; locais; auditor da API (justificada; evasiva; não recusa) | 70; 40; 43; 24; 3 | ✅ | `runs/confirmacao/novo_api/contrastes.json` |
| custo US$: geração por agente | claude-haiku-4.5 1,9859; gemini-3.1-flash-lite 0,4100; gpt-5.4-nano 0,3212 | ✅ | `runs/confirmacao/novo_api/contrastes.json`; `responses.parquet` de cada agente |
| custo US$: geração; avaliação (juiz; auditor; suficiência); total | 2,7171; 1,1576 (0,8611; 0,2965; 0,0000); 3,8747 | ✅ | `runs/confirmacao/novo_api/contrastes.json`; `runs/confirmacao/novo_api/custo.json` |

## 11. Tabelas do apêndice (geradas — coladas verbatim em `docs/relatorio_apendice.md`)


### escada

| degrau | recall de passagem [IC95] | recall de arquivo | Hit@K | itens sem dispositivo | chunks/item | palavras/item | teto oracular |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0,2355 [0,2072; 0,2658] | 0,4531 | 0,3813 | 0,6187 | 7,05 | 3.200 | 0,9145 |
| R0' | 0,2366 [0,2078; 0,2669] | 0,4563 | 0,3813 | 0,6187 | 7,04 | 3.196 | 0,9615 |
| R1 | 0,2397 [0,2107; 0,2703] | 0,3912 | 0,3903 | 0,6097 | 7,00 | 3.205 | 0,9615 |
| R2 | 0,2521 [0,2219; 0,2836] | 0,3849 | 0,3957 | 0,6043 | 7,00 | 3.372 | 0,9615 |
| R3 | 0,3426 [0,3106; 0,3754] | 0,5708 | 0,5468 | 0,4532 | 7,00 | 3.487 | 0,9615 |
| R5 | 0,3381 [0,3064; 0,3711] | 0,5664 | 0,5396 | 0,4604 | 7,00 | 3.481 | 0,9615 |


### hipoteses

| hipótese | o que testa | resultado | veredito |
|---|---|---|---|
| H1 | relaxar a dedup e reranquear (R1, R2) recupera os itens zerados por ordenação | R1 0,2397 [0,2107; 0,2703]: IC inclui R0 (0,2355); R2 0,2521 [0,2219; 0,2836]: IC inclui R1 (0,2397) | não confirmada |
| H2 | a cota por tipo de fonte (R3) corrige a dominância dos acórdãos | R3 0,3426 [0,3106; 0,3754]: IC exclui R2 (0,2521); itens sem dispositivo 0,6043 → 0,4532 | confirmada |
| H3 | o reparo do corpus eleva o teto oracular acima de 0,9145 | teto oracular 0,9145 → 0,9615 (82 pares passam a casar); R0' 0,2366 [0,2078; 0,2669] | confirmada |
| H4 | metadados taxonômicos como boost (R4) dão ganho pequeno | R4 não executado (decisão 2026-09-12; M2.1–M2.3 abandonados) | não testada |
| H5 | multi-query (R5) ataca a classe "pool" | R5 0,3381 [0,3064; 0,3711]: IC inclui R3 (0,3426) | não confirmada |


### geracao

| variante | degrau | N | adequação média [IC95] | qwen35_9b | granite41_8b | gemma4_e4b | recusa [IC95] | evasiva [IC95] | falha de parse | suficiente [IC95] | parcial | insuficiente | palavras/item |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | R3 | 200 | 0,2300 [0,1883; 0,2733] | 0,2300 | 0,2300 | 0,2300 | 0,0667 [0,0450; 0,0917] | 0,0183 [0,0083; 0,0300] | 0,0100 | 0,3450 [0,2800; 0,4100] | 115 | 16 | 3.584 |


### comparacao

| métrica | n | atual | novo | Δ novo − atual [IC95] |
|---|---:|---:|---:|---:|
| recall de passagem | 151 | 0,2137 | 0,3308 | +0,1171 [+0,0712; +0,1661] |
| recall de arquivo | 151 | 0,4140 | 0,5652 | +0,1512 [+0,0946; +0,2095] |
| adequação média (3 agentes) | 200 | 0,2267 | 0,2300 | +0,0033 [−0,0350; +0,0417] |
| adequação qwen35_9b | 200 | 0,2400 | 0,2300 | −0,0100 [−0,0700; +0,0500] |
| adequação granite41_8b | 200 | 0,2050 | 0,2300 | +0,0250 [−0,0400; +0,0900] |
| adequação gemma4_e4b | 200 | 0,2350 | 0,2300 | −0,0050 [−0,0651; +0,0600] |
| adequação B0 (melhor agente fixo) | 200 | 0,2400 | 0,2300 | −0,0100 [−0,0700; +0,0500] |
| recusa textual | 200 | 0,0617 | 0,0667 | +0,0050 [−0,0150; +0,0267] |
| recusa evasiva (auditor = 1) | 200 | 0,0133 | 0,0183 | +0,0050 [−0,0067; +0,0183] |
| falha de parse | 200 | 0,0083 | 0,0100 | +0,0017 [−0,0083; +0,0117] |
| contexto suficiente (= 2) | 200 | 0,2500 | 0,3450 | +0,0950 [+0,0450; +0,1450] |
| contexto insuficiente (= 0) | 200 | 0,1050 | 0,0800 | −0,0250 [−0,0700; +0,0150] |
| palavras/item | 200 | 3.264,6800 | 3.583,8700 | +319,1900 [+149,9739; +492,6401] |
| chunks/item | 200 | 7,1200 | 7,0000 | −0,1200 [−0,4100; +0,1650] |


### custo

| etapa | atual: chamadas | atual: US$ | novo: chamadas | novo: US$ | novo: segundos |
|---|---:|---:|---:|---:|---:|
| agentes | 600 | 0,0000 | 600 | 0,0000 | 2.949 |
| juiz | 595 | 0,8854 | 582 | 0,8503 | 1.240 |
| auditor | 37 | 0,1771 | 40 | 0,2037 | 104 |
| suficiência | 200 | 1,0442 | 200 | 1,0950 | 535 |
| contexto (recuperação) | — | — | — | — | 136 |
| total | | 2,1067 | | 2,1490 | 4.964 |


## 12. Como manter este arquivo

1. Nunca edite à mão: `uv run rag2 numeros` regenera tudo dos artefatos.
2. Antes de commitar o apêndice: `uv run rag2 numeros --conferir docs/relatorio_apendice.md` (exit 0).
3. Um número novo no apêndice nasce de um artefato versionado, entra aqui por `numeros.py`, e só então no texto.

Comandos que produziram os artefatos citados:

```bash
uv run rag2 degrau R0   # … R0', R1, R2, R3, R5 (goldens em runs/escada/)
uv run rag2 metricas-run phase2_local_full_715
uv run rag2 variante P0 --degrau R3           # smoke (M4.1)
RAG2_DEVICE=cuda:0 uv run rag2 confirmacao executar   # N=200 (M5.1)
uv run rag2 confirmacao comparar              # comparação e custo (M5.2)
uv run rag2 confirmacao braco-api             # contrastes do braço de API (exploratório, §10)
uv run rag2 numeros                           # este arquivo
```
