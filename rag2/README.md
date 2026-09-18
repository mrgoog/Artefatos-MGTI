# RAG-2 (Apêndice B)

Este diretório refere-se **apenas ao Apêndice B** da dissertação: o protótipo RAG-2 de recuperação e
geração para o BR-TaxQA-R. O restante da dissertação é coberto pela raiz do repositório
([`../README.md`](../README.md)), cujo pacote `ensemble_llm` o RAG-2 importa como dependência.

## Conteúdo

```
src/rag2/            pacote e CLI (`rag2`)
tests/               testes (os marcados gpu/llm exigem GPU, llama-swap ou OpenRouter)
data/reparo/         textos oficiais extraídos (Planalto) usados no reparo do corpus
data/reparado/       corpus normativo reparado e relatório do reparo (82 pares ganhos, teto 0,9145 → 0,9615)
runs/escada/         degraus R0, R0', R1, R2, R3, R5 e a produção recomputada (Tabela 11)
runs/prompts/P0/     teste de fumaça do harness de geração (20 itens; mecanismo, não resultado)
runs/confirmacao/    geração em N = 200 sobre R3 (novo/), comparação pareada com a produção (Tabelas 12–13, 16)
                     e grupo com agentes de API (novo_api/, Tabelas 14–15; runner em novo_api/run_api.py)
docs/pre-registro.md             hipóteses H1–H5 e critério de decisão, fixados antes dos degraus acima de R0
docs/numeros_canonicos_rag2.md   procedência de cada cifra do apêndice, gerado por `rag2 numeros`
docs/decisions.md                registro datado das decisões de implementação e dos desvios
```

`docs/decisions.md` é o registro de trabalho do projeto e cita arquivos internos (specs, revisões,
rascunhos) que não foram publicados.

## Como reproduzir

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. Copie `.env.example` para `.env`. Por padrão, o RAG-2
lê da raiz do repositório as execuções de referência (`../runs/phase2_local_full_715`,
`../runs/phase1_local_pilot_200_1`), os scripts canônicos (`recall_passagem.py`, `auditar_recusas.py`,
`juiz_suficiencia_contexto.py`), os dados brutos (`../data/brtaxq`) e o índice de produção
(`../runs/_shared/rag/br_taxqa_r_bge-m3_cuda1`). Os dois últimos são gerados pelos comandos
`download_dataset.py` e `build_rag_index.py` do README da raiz.

Sem GPU e sem chamadas de modelo, a partir dos artefatos versionados:

```bash
uv sync
uv run rag2 numeros --saida /tmp/ledger.md && diff docs/numeros_canonicos_rag2.md /tmp/ledger.md
uv run pytest -m "not gpu and not llm"
```

Cadeia completa (GPU para embedder e reranker; llama-swap para os agentes locais e o reformulador de R5;
OpenRouter para juiz, auditor e juiz de suficiência):

```bash
uv run rag2 reparar-corpus --offline             # data/reparado/ a partir de data/reparo/
uv run rag2 construir-indice reparado            # runs/_shared/rag/rag2_reparado (não versionado)
uv run rag2 construir-indice reparado --dividir  # subíndices de normas e de acórdãos (R3, R5)
uv run rag2 comparar-indices                     # teto oracular 0,9145 → 0,9615
uv run rag2 degrau R0                            # e R0', R1, R2, R3, R5 → runs/escada/
uv run rag2 metricas-run phase2_local_full_715   # produção recomputada
uv run rag2 variante P0 --degrau R3              # teste de fumaça
uv run rag2 confirmacao executar                 # N = 200 sobre R3 → runs/confirmacao/novo/
uv run rag2 confirmacao comparar                 # comparação pareada e custo
# grupo com agentes de API (fora do pré-registro): copie contexto.parquet e suficiencia_contexto.parquet
# de runs/confirmacao/novo/ para a pasta de saída e rode
uv run python runs/confirmacao/novo_api/run_api.py runs/confirmacao/novo_api
uv run rag2 confirmacao braco-api                # contrastes do grupo de API
uv run rag2 numeros                              # docs/numeros_canonicos_rag2.md
```

As reformulações de R5, as respostas, os contextos e os rótulos estão congelados em `runs/`. Reexecutar um
degrau ou a confirmação com eles presentes reproduz os resultados sem chamar modelos. Caches de pools, do
reranker e do juiz ficam em `runs/_shared/` (não versionado) e são refeitos quando ausentes.
