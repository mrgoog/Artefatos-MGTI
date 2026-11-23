# Verificação humana exploratória do juiz LLM

Verificação exploratória de plausibilidade, por anotador único (o autor), dos rótulos
produzidos pelos juízes automáticos da dissertação (Seção 6.2.2, Tabela 3). Não substitui
validação por especialistas.
Nos pares substantivos, o cegamento não foi assegurado pela interface.

Uma anotação (`q_0065 · gemma4_e4b`) é preservada no arquivo bruto, mas excluída das
métricas: nela, o anotador aplicou justificabilidade contextual a um caso substantivo,
em desacordo com o protocolo de comparação com a referência. A exclusão foi decidida
durante a revisão metodológica posterior à anotação; ela é declarada em
`scripts/analisar_validacao_humana.py` e reproduzida em `resultados.md`.

## Arquivos

- `amostra.parquet` — todos os 859 pares item×agente rotulados, em três estratos
  (`adequada` 250, `inadequada` 463, `recusa` 146), cada um com `ordem_sorteio` fixa
  (seed 42). Gerado por `scripts/sortear_amostra_validacao.py`; não editar à mão.
- `anotacoes.parquet` — uma linha por par anotado, regravada a cada salvamento pela UI.
- `resultados.md` — métricas da Tabela 3, geradas por `scripts/analisar_validacao_humana.py`
  a partir dos dois parquet; não editar à mão.

## Uso

```bash
uv run python scripts/sortear_amostra_validacao.py   # uma vez (já executado)
uv run streamlit run scripts/anotar_validacao.py
```

Na barra lateral escolha quantos itens de cada estrato entram na sequência (padrão:
20 adequadas, 15 inadequadas, todas as 146 recusas). Aumentar N depois não altera os itens
já sorteados: a sequência é sempre o prefixo da ordem de sorteio.

## Protocolo

1. O rótulo explícito e o raciocínio do juiz ficam ocultos até a anotação ser salva. A
   interface, contudo, mostra o estrato `adequada` ou `inadequada`, derivado do rótulo
   automático; portanto, essa ocultação não assegura cegamento nos pares substantivos.
2. Respostas substantivas: julgar contra a referência e os dispositivos citados, como na
   rubrica do juiz binário (`src/ensemble_llm/juiz/rubrica.py`). Opções: adequada,
   inadequada, não sei julgar; categorias de problema opcionais.
3. Recusas: julgar contra o contexto servido, como no auditor
   (`scripts/auditar_recusas.py`). Opções: justificada (2), evasiva (1), não é recusa (0),
   não sei julgar.
4. Após salvar, "Revelar" mostra o juiz; o comentário pós-revelação é gravado em campo
   separado. A interface permite alterar uma anotação salva e o arquivo conserva apenas
   seu estado mais recente, sem trilha imutável de versões.

## Vocabulário dos veredictos de recusa

O script do auditor define 2 = justificada, 1 = evasiva, 0 = não é recusa.
