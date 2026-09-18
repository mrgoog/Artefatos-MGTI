# Reparo do corpus normativo

Fontes: `FONTES` em `src/rag2/reparo.py`; textos extraídos em `data/reparo/`; data de acesso e não reparados em `docs/decisions.md`. Artigos casáveis medidos no texto integral (limite superior do casamento por chunk).

| status | arquivo | palavras antes → depois | trecho comum | artigos casáveis antes → depois | pares que passam a casar | fonte / nota |
|---|---|---:|:-:|---:|---:|---|
| REPARADO | Medida Provisória nº 252.txt | 27 → 9758 | sim | 0/1 → 1/1 | 1 | https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2005/mpv/252.htm |
| REPARADO | Lei nº 10.406.txt | 62 → 102764 | sim | 0/49 → 49/49 | 57 | https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm |
| REPARADO | Decreto nº 3.000.txt | 90 → 151757 | sim | 0/2 → 2/2 | 2 | https://www.planalto.gov.br/ccivil_03/decreto/d3000.htm |
| SEM_FONTE | Instrução Normativa RFB nº 118.txt | 109 → 109 | — | 0/12 → 0/12 | 0 |  SIJUT/RFB sem HTML estático nesta data (SPA desde 2024-09; ver docs/decisions.md) |
| SEM_GANHO | Decreto nº 93.153.txt | 171 → 171 | sim | 0/1 → 0/1 | 0 | https://www.planalto.gov.br/ccivil_03/decreto/1980-1989/1985-1987/d93153.htm página idêntica ao dataset; o dispositivo esperado está em anexo PDF digitalizado (sem camada de texto) |
| SEM_GANHO | Decreto nº 85.801.txt | 183 → 183 | sim | 0/2 → 0/2 | 0 | https://www.planalto.gov.br/ccivil_03/atos/decretos/1981/d85801.html página idêntica ao dataset; o dispositivo esperado está em anexo PDF digitalizado (sem camada de texto) |
| SEM_GANHO | Decreto nº 27.784.txt | 184 → 184 | sim | 0/1 → 0/1 | 0 | https://www.planalto.gov.br/ccivil_03/decreto/antigos/d27784.htm página idêntica ao dataset; o dispositivo esperado está em anexo PDF digitalizado (sem camada de texto) |
| SEM_FONTE | Instrução Normativa SRF nº 4, de 13 de janeiro de 1999.txt | 232 → 232 | — | 1/1 → 1/1 | 0 |  SIJUT/RFB sem HTML estático nesta data (SPA desde 2024-09; ver docs/decisions.md) |
| SEM_GANHO | Decreto nº 361.txt | 245 → 245 | sim | 0/1 → 0/1 | 0 | https://www.planalto.gov.br/ccivil_03/decreto/1990-1994/d0361.htm página idêntica ao dataset; o dispositivo esperado está em anexo PDF digitalizado (sem camada de texto) |
| SEM_GANHO | Lei nº 8.971.txt | 285 → 285 | sim | 1/1 → 1/1 | 0 | https://www.planalto.gov.br/ccivil_03/leis/l8971.htm lei curta (5 artigos); a fonte tem as mesmas 285 palavras |
| REPARADO | Constituição Federal de 1988.txt | 8106 → 108480 | sim | 1/11 → 11/11 | 22 | https://www.planalto.gov.br/ccivil_03/constituicao/constituicaocompilado.htm |

**Resumo:** reparados=4 sem_ganho=5 sem_fonte=2 falhas=0 · pares que passam a casar: 82 · registros: 478

sha256 do JSON reparado: `aa55ee56a4b42cbe4cae936b5e48ae5cc22d7a1acd3164a789db0c0ed27a7acd`
