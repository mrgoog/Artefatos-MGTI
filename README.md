# Artefatos-MGTI

## UNIVERSIDADE CATÓLICA DE BRASILIA

## MESTRADO EM GESTÃO, TECNOLOGIA DA INFORMAÇÃO E INOVAÇÃO

### Racionalidade limitada e governança seletiva em sistemas baseados em LLMs:

### Projeto e auditoria de um ensemble sob RAG compartilhado

Autor:              André Luiz Castilhos Magoga
Orientador:     Dr. Rosalvo Ermes Streit
Coorientador: Dr. Eduardo Amadeu Dutra Moresi


Este repositório acompanha a dissertação como um todo, agregando os artefatos computacionais da dissertação e avaliação empírica.

| Diretório | Parte da dissertação |
|---|---|
| raiz (`src/ensemble_llm`, `scripts/`, `configs/`, `runs/`, `tabelas/`, `data/`, `bibliografia/`) | Corpo do texto (Seções 2 a 6) e Apêndice A: artefatos A0 e A5, auditoria da cabeça de decisão, verificação humana e estudo bibliométrico |
| `rag2/` | **Somente o Apêndice B**: protótipo RAG-2 (escada de recuperação R0–R5, geração em N = 200, comparação pareada e custo). Ver [`rag2/README.md`](rag2/README.md) |

O repositório reúne apenas o necessário para inspecionar e reproduzir os números publicados: código-fonte,
configurações, os scripts que geraram os artefatos citados e as execuções descritas no texto. 

## Conteúdo

```
src/ensemble_llm/        pacote do pipeline (recuperação, agentes, juiz, calibração, decisão, avaliação)
configs/                 configurações declarativas das execuções e do índice RAG
scripts/                 pontos de entrada do pipeline e scripts de análise que geraram tabelas e figuras
runs/
  phase2_local_full_715/          A0: execução principal (715 itens, 3 agentes locais)
  phase2_local_full_715_meanraw/  A5: mesmas respostas, cabeça de decisão redesenhada; figures/dissertacao = Figuras 1–5
  phase1_local_pilot_200_1/       piloto local, N = 200 (Tabela 10; amostra de 200 do Apêndice B)
  phase1_api_pilot_200_1/         piloto com agentes de API, N = 200 (Tabela 10; Apêndice B4.2)
  _shared/rag/br_taxqa_r_bge-m3_cuda1/metrics/  métricas do índice de produção (Tabela 8; distribuição dos trechos)
tabelas/                 tabelas derivadas (Markdown + Parquet) usadas no Capítulo 6
data/validacao_humana/   amostra sorteada e estado final das anotações (Seção 6.2.2, Tabela 3)
bibliografia/            corpus e saídas do mapeamento bibliométrico (Seção 2, Apêndice A4); inventário em bibliografia/README.md
tests/                   testes das mensurações de decisão
rag2/                    protótipo RAG-2 (somente Apêndice B)
```

Cada execução em `runs/` guarda respostas dos agentes (`agent_responses/`), rótulos do juiz (`judge_labels/`), partições e objetos de decisão por fold
(`folds/`), consolidados de teste, métricas (`metrics/`) e o `manifest.yaml` com configuração, sementes e commit de origem.

## Mapa artefato → script

| Artefato na dissertação | Script (`scripts/`) | Saída |
|---|---|---|
| Tabela 3 (verificação humana) | `sortear_amostra_validacao.py`, `anotar_validacao.py` (Streamlit), `analisar_validacao_humana.py` | `data/validacao_humana/` |
| Tabela 4 e Figura 2 (suficiência do contexto) | `juiz_suficiencia_contexto.py`, `analise_suficiencia_decisao.py` | `runs/phase2_local_full_715/metrics/suficiencia_contexto.parquet` |
| Tabela 5 e Seção 6.2.5 (pontos de operação e contrastes de decisão) | `mensuracoes_decisao_artigo2.py` | `tabelas/pontos_operacao_*`, `contraste_pontos_operacao_*`, `contrastes_decisao_*`, `metricas_decisao_cobertura_*`, `auditoria_decisao_itens_*` |
| Tabela 6 (ECE) | pipeline (`ensemble_llm.orquestracao.execucao`) | `runs/phase2_local_full_715/metrics/summary.json` |
| Tabela 7, interação 2×2 e matriz pareada (ablação da cabeça de decisão) | `ablacao_arquitetura.py`, `gerar_tabelas_ablacao.py` | `tabelas/ablacao_*`, `interacao_2x2_*`, `matriz_pareada_*` |
| Tabela 8 (varredura de recuperação) | `varredura_recuperacao.py` | `runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/metrics/varredura_recuperacao.parquet` |
| Tabela 10 (estabilidade dos agentes) | pipeline, execuções `phase1_*` e `phase2_local_full_715` | `runs/*/judge_labels/`, `runs/*/metrics/` |
| Seção 6.2.1 (auditoria de recusas) | `auditar_recusas.py` | `runs/phase2_local_full_715/metrics/auditoria_recusas.parquet` |
| Seção 6.3.3 (seleção condicionada) | `analisar_selecao_condicionada.py` | `tabelas/selecao_condicionada_*` |
| Seções 6.3.4, 6.4.2 e 6.4.3 (recomputações, distribuição dos trechos) | `recomputacao_fase0.py` | saída padrão; `distribuicao_chunks.json` |
| Seção 6.4.3 (recall de passagem) | `recall_passagem.py` | `runs/phase2_local_full_715/metrics/recall_passagem.parquet` |
| Figuras 1–5 (1 `f3_architecture`, 2 `f5_suficiencia`, 3 `f2_reliability`, 4 `f1_risk_coverage`, 5 `f7_error_agreement`) | `generate_article_figures.py` | `runs/phase2_local_full_715_meanraw/figures/dissertacao/` |

## Como reproduzir

Requer [uv](https://docs.astral.sh/uv/) e Python 3.11+. Copie `.env.example` para `.env` e preencha
`OPENROUTER_API_KEY` (juiz e auditores). Os agentes locais foram servidos por llama-swap/llama-server,
com os modelos e temperaturas registrados nas configurações.

Análises sobre os artefatos versionados (sem GPU e sem chamadas de modelo):

```bash
uv sync
uv run python scripts/gerar_tabelas_ablacao.py
uv run python scripts/mensuracoes_decisao_artigo2.py
uv run python scripts/analisar_selecao_condicionada.py
uv run python scripts/analise_suficiencia_decisao.py --run phase2_local_full_715_meanraw --cmin 0.7
uv run python scripts/analisar_validacao_humana.py
uv run python scripts/generate_article_figures.py --run phase2_local_full_715_meanraw \
  --comparison-run phase2_local_full_715 \
  --main-label "Cabeça minimalista (A5: média das confianças brutas)" \
  --comparison-label "Arquitetura proposta (A0: máximo da confiança ponderada e calibrada)" \
  --output-dir runs/phase2_local_full_715_meanraw/figures/dissertacao
uv run pytest
```

Pipeline completo, a partir dos dados públicos:

```bash
uv run python scripts/download_dataset.py --output-dir data/brtaxq
uv run python scripts/build_rag_index.py --config configs/rag_brtaxq.yaml
uv run python scripts/eval_rag.py --config configs/rag_brtaxq.yaml
uv run python scripts/varredura_recuperacao.py --config configs/rag_brtaxq.yaml
uv run python scripts/validate_config.py configs/phase2_local_full.yaml
uv run python -m ensemble_llm.orquestracao.execucao --config configs/phase2_local_full.yaml
uv run python -m ensemble_llm.orquestracao.execucao --config configs/phase2_local_full_meanraw.yaml
uv run python scripts/recall_passagem.py --run phase2_local_full_715
uv run python scripts/auditar_recusas.py --run phase2_local_full_715
uv run python scripts/juiz_suficiencia_contexto.py --run phase2_local_full_715
uv run python scripts/recomputacao_fase0.py --run phase2_local_full_715 --c-min 0.7
```

Os scripts que dependem do índice RAG ou dos dados brutos (`recall_passagem.py`, `juiz_suficiencia_contexto.py`,
`recomputacao_fase0.py`, `varredura_recuperacao.py`, `sortear_amostra_validacao.py`) só rodam depois dos dois
primeiros comandos acima.

As respostas de modelos locais não se reproduzem byte a byte entre execuções. A reprodução exata dos números
parte das respostas e dos rótulos persistidos em `runs/`, não da reexecução dos modelos.

## O que não está versionado

- **Dados brutos do BR-TaxQA-R** (`data/brtaxq/`): baixados por `scripts/download_dataset.py` de
  <https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R> (v1.1, licença CC-BY-4.0).
- **Corpo dos índices RAG** (`runs/_shared/rag/*/bm25/` e `*/denso/`, cerca de 700 MB): excedem o limite de
  arquivo do GitHub e são reconstruídos por `scripts/build_rag_index.py`. As métricas dos índices ficam versionadas.

## Licença

Código sob GPL-3.0 ([`LICENSE`](LICENSE)). O arquivo em `rag2/data/reparado/` redistribui o corpus normativo do
BR-TaxQA-R (CC-BY-4.0, UNICAMP-DL), com quatro documentos completados a partir de fontes oficiais.
