# Comparação pareada N=200: atual (phase2_local_full_715) × novo (P0/R3)

Itens: 200 (has_doc_ref 168; com dispositivo 151 — recall só nestes). B0 = qwen35_9b (melhor agente fixo, escolhido na produção). IC95 percentil pareado do Δ, estratificado por has_doc_ref, B = 10000, semente 42.

| métrica | n | atual | novo | Δ novo − atual [IC95] |
|---|---:|---:|---:|---:|
| recall_passagem | 151 | 0.2137 | 0.3308 | +0.1171 [+0.0712, +0.1661] |
| recall_arquivo | 151 | 0.4140 | 0.5652 | +0.1512 [+0.0946, +0.2095] |
| z_media | 200 | 0.2267 | 0.2300 | +0.0033 [-0.0350, +0.0417] |
| z_qwen35_9b | 200 | 0.2400 | 0.2300 | -0.0100 [-0.0700, +0.0500] |
| z_granite41_8b | 200 | 0.2050 | 0.2300 | +0.0250 [-0.0400, +0.0900] |
| z_gemma4_e4b | 200 | 0.2350 | 0.2300 | -0.0050 [-0.0651, +0.0600] |
| z_b0 | 200 | 0.2400 | 0.2300 | -0.0100 [-0.0700, +0.0500] |
| recusa | 200 | 0.0617 | 0.0667 | +0.0050 [-0.0150, +0.0267] |
| evasiva | 200 | 0.0133 | 0.0183 | +0.0050 [-0.0067, +0.0183] |
| falha_parse | 200 | 0.0083 | 0.0100 | +0.0017 [-0.0083, +0.0117] |
| suficiente | 200 | 0.2500 | 0.3450 | +0.0950 [+0.0450, +0.1450] |
| insuficiente | 200 | 0.1050 | 0.0800 | -0.0250 [-0.0700, +0.0150] |
| palavras | 200 | 3264.6800 | 3583.8700 | +319.1900 [+149.9739, +492.6401] |
| n_chunks | 200 | 7.1200 | 7.0000 | -0.1200 [-0.4100, +0.1650] |

Condição de revisão (B3 waived, decisions 2026-09-13): Δ adequação média = +0.0033 [-0.0350, +0.0417] — IC inclui 0: **não** significativamente melhor; a decisão B3 permanece.
