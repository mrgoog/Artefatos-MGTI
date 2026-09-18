# Braço de agentes de API sobre R3 — exploratório, fora do pré-registro (N=200)

Contexto e suficiência de R3 reusados byte a byte do braço local; agentes de API com `max_tokens` 4096 (piloto da origem) contra 1024 dos locais — comparação de configurações de geração sob contexto fixo, não só de família de agentes. IC95 percentil pareado por item, B = 10000, semente 42, estratificado por has_doc_ref (sem estratificação dentro do estrato suficiente).

| contraste | n | a | b | Δ a − b [IC95] |
|---|---:|---:|---:|---:|
| API@R3 − locais@R3 | 200 | 0.2750 | 0.2300 | +0.0450 [+0.0050; +0.0850] |
| locais@R3 − produção | 200 | 0.2300 | 0.2267 | +0.0033 [-0.0350; +0.0417] |
| API@R3 − produção | 200 | 0.2750 | 0.2267 | +0.0483 [+0.0067; +0.0917] |
| API@R3 − locais@R3, só contexto suficiente | 69 | 0.4783 | 0.4589 | +0.0193 [-0.0628; +0.0966] |

| suficiência | n | API | local | contribuição ao Δ global |
|---|---:|---:|---:|---:|
| suficiente | 69 | 0.4783 | 0.4589 | +0.0067 |
| parcial | 115 | 0.1884 | 0.1217 | +0.0383 |
| insuficiente | 16 | 0.0208 | 0.0208 | +0.0000 |

| agente | adequação (200) | adequação no estrato suficiente |
|---|---:|---:|
| claude-haiku-4.5 | 0.2800 | 0.5072 |
| gemini-3.1-flash-lite | 0.2500 | 0.4493 |
| gpt-5.4-nano | 0.2950 | 0.4783 |
| qwen35_9b (local) | — | 0.4783 |
| granite41_8b (local) | — | 0.3913 |
| gemma4_e4b (local) | — | 0.5072 |

Recusas: API 70 (auditor: {'justificada': 43, 'evasiva': 24, 'nao_recusa': 3}); locais 40.

| custo US$ | valor |
|---|---:|
| geração claude-haiku-4.5 | 1.9859 |
| geração gemini-3.1-flash-lite | 0.4100 |
| geração gpt-5.4-nano | 0.3212 |
| geração total | 2.7171 |
| avaliação juiz | 0.8611 |
| avaliação auditor | 0.2965 |
| avaliação suficiencia | 0.0000 |
| avaliação total | 1.1576 |
| total | 3.8747 |
