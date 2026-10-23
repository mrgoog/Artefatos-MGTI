# Verificação humana exploratória do juiz LLM — resultados

Anotador único (autor). 65 pares anotados de 859 registros no quadro amostral; 64 pares incluídos na análise. Nos pares substantivos, o cegamento não foi assegurado pela interface. Gerado por `scripts/analisar_validacao_humana.py`.

## Exclusão documentada

- `q_0065 · gemma4_e4b`: O anotador aplicou justificabilidade contextual a um par do estrato substantivo, em desacordo com o protocolo de comparação com a referência. A exclusão foi decidida durante a revisão metodológica posterior à anotação. O registro bruto foi preservado, mas não participa das métricas.

## Juiz binário de adequação (estratos `adequada` e `inadequada`)

n incluídos na análise = 34; 'não sei julgar' = 0.

| juiz       |   adequada |   inadequada |
|:-----------|-----------:|-------------:|
| adequada   |         20 |            0 |
| inadequada |          2 |           12 |

| estrato    |   n |   concordam | acordo   | IC 95% exato    | interpretação                          |
|:-----------|----:|------------:|:---------|:----------------|:---------------------------------------|
| adequada   |  20 |          20 | 100.0%   | [83.2%; 100.0%] | P[humano=adequada | juiz=adequada]     |
| inadequada |  14 |          12 | 85.7%    | [57.2%; 98.2%]  | P[humano=inadequada | juiz=inadequada] |

Os 250 registros `adequada` e 463 `inadequada` resultam da seleção de no máximo um par por item e da resolução de sobreposições. Eles não constituem a população dos 2.129 julgamentos; por isso, não se reconstrói uma acurácia global.

Acordo bruto na amostra = 94.1%; kappa de Cohen = **0.876** [IC bootstrap 95%: 0.691; 1.000] (n = 34; amostra estratificada, kappa é descritivo).

Desacordos: juiz=adequada/humano=inadequada = 0; juiz=inadequada/humano=adequada = 2. O viés observado é unidirecional: o juiz é mais severo que o anotador.

Categorias de problema apontadas pelo anotador (estratos substantivos):

| categorias_erro                               |   n |
|:----------------------------------------------|----:|
| Afirma algo não sustentado pelo material      |   4 |
| Contradiz a referência em ponto material      |   4 |
| Incompleta: não esgota o que a pergunta exige |   1 |

## Auditor de recusas (estrato `recusa`)

n anotados = 30 de 146; 'não sei julgar' = 0.

| auditor      |   justificada |   não é recusa |
|:-------------|--------------:|---------------:|
| evasiva      |             4 |              0 |
| justificada  |            24 |              0 |
| não é recusa |             0 |              2 |

Acordo bruto = 26/30 = **86.7%** [IC exato 95%: 69.3%; 96.2%]; kappa de Cohen = 0.464 [0.000; 0.872].

Fração de recusas julgadas **justificadas pelo humano** = 28/30 = 93.3% [77.9%; 99.2%]; o auditor automático marcou justificadas 24/30 nesta amostra e 112/146 no total.

Nos 6 casos em que o auditor NÃO marcou 'justificada', o humano marcou: justificada ×4, não é recusa ×2.

## Desacordos e comentários

### q_0490 · granite41_8b · adequada #4 — concorda

- pergunta: A receita correspondente ao aluguel de pastagens, máquinas e equipamentos é tributada na atividade rural?
- juiz/auditor: **adequada** · humano: **adequada** · confiança do agente: 0.98
- pós-revelação: Resposta correta. O juiz não teve acesso aos dados que confirmar o que definiu como imprecisão.

### q_0479 · gemma4_e4b · adequada #5 — concorda

- pergunta: Todas as pessoas que exploram imóvel rural estão obrigadas a inscrever-se no Cadastro de Imóveis Rurais do ITR da Secretaria Especial da Receita Federal do Brasil (RFB)?
- juiz/auditor: **adequada** · humano: **adequada** · confiança do agente: 0.50
- pós-revelação: faltou o contexto ao juiz. correto.

### q_0213 · gemma4_e4b · adequada #6 — concorda

- pergunta: Os rendimentos oriundos de contrato de arrendamento de imóvel rural são tributáveis?
- juiz/auditor: **adequada** · humano: **adequada** · confiança do agente: 1.00
- categorias: Afirma algo não sustentado pelo material
- pós-revelação: o modelo forneceu uma resposta correta, mas alucinou informações. o juiz detectou corretamente ambas.

### q_0647 · qwen35_9b · adequada #7 — concorda

- pergunta: Com relação à isenção para alienação de bens ou direitos por valor igual ou inferior a R$ 35.000,00, como proceder na alienação de bens ou direitos possuídos em condomínio?
- juiz/auditor: **adequada** · humano: **adequada** · confiança do agente: 0.95
- categorias: Afirma algo não sustentado pelo material
- pós-revelação: modelo usou seu conhecimento para responder. juiz detectou corretamente, mas a resposta é adequada.

### q_0430 · granite41_8b · adequada #14 — concorda

- pergunta: Os pagamentos efetuados por profissional autônomo a terceiros correspondentes a serviços prestados em anos anteriores podem ser deduzidos?
- juiz/auditor: **adequada** · humano: **adequada** · confiança do agente: 0.92
- categorias: Afirma algo não sustentado pelo material
- comentário: A resposta está correta, mas não houve recuperação do RAG (ou ocorreu erro) e o modelo forneceu informações tecnicamente imprecisas.

### q_0244 · granite41_8b · inadequada #2 — concorda

- pergunta: Qual é o tratamento tributário dos rendimentos recebidos pela pessoa física em decorrência da locação de unidade imobiliária em sistema denominado de pool hoteleiro?
- juiz/auditor: **inadequada** · humano: **inadequada** · confiança do agente: 0.92
- pós-revelação: Resposta incorreta. O Juiz recusou pela apresentação de adendos que ele não viu, mas o erro real é determina que os rendimentos seriam tributados como "aluguel" e sujeitos à retenção de IR na fonte pela tabela progressiva de aluguéis, quando na verdade seguem as regras de distribuição de lucros aos sócios da SCP

### q_0146 · granite41_8b · inadequada #11 — DESACORDO

- pergunta: Religiosos, missionários, pesquisadores e professores não residentes, trabalhando no Brasil, e recebendo proventos diretamente de entidades e empresas sediadas no exterior, são contribuintes do Impost
- juiz/auditor: **inadequada** · humano: **adequada** · confiança do agente: 0.95
- pós-revelação: Resposta adequada. Juiz por não ter acesso aos dados recuperados, criticou informações adicionais da resposta.

### q_0575 · gemma4_e4b · inadequada #12 — concorda

- pergunta: Contribuinte que possui apenas usufruto sobre um imóvel e propriedade de outro, tem direito a isenção de ganho de capital se alienar o imóvel do qual tem propriedade?
- juiz/auditor: **inadequada** · humano: **inadequada** · confiança do agente: 0.30
- categorias: Incompleta: não esgota o que a pergunta exige
- comentário: Houve recuperação incompleta para a correta resposta, mas o agente divagou e não declarou claramente este fato.

### q_0663 · granite41_8b · inadequada #14 — DESACORDO

- pergunta: Quem é o contribuinte do imposto sobre o ganho de capital na transferência de ativos em decorrência de dissolução da sociedade conjugal ou da união estável?
- juiz/auditor: **inadequada** · humano: **adequada** · confiança do agente: 0.98
- pós-revelação: o juiz não teve acesso aos dados recuperados do RAG e desqualificou a resposta por isto.

### q_0658 · gemma4_e4b · recusa #3 — DESACORDO

- pergunta: O ganho de capital apurado por contribuinte que alienou o único imóvel em 2023, por valor inferior a R$ 440.000,00, adquirido por meio de dação em pagamento no ano anterior, faz jus à isenção do único
- juiz/auditor: **evasiva** · humano: **justificada** · confiança do agente: 0.50

### q_0019 · gemma4_e4b · recusa #6 — DESACORDO

- pergunta: O contribuinte que optar pelo desconto simplificado pode excluir os honorários advocatícios pagos referentes a rendimentos recebidos acumuladamente, por decisão judicial?
- juiz/auditor: **evasiva** · humano: **justificada** · confiança do agente: 0.30

### q_0523 · gemma4_e4b · recusa #15 — DESACORDO

- pergunta: Qual é o tratamento tributário do valor recebido em moeda corrente do fornecedor, decorrente da devolução de bens e insumos rurais anteriormente adquiridos e pagos?
- juiz/auditor: **evasiva** · humano: **justificada** · confiança do agente: 0.10

### q_0181 · qwen35_9b · recusa #20 — DESACORDO

- pergunta: Qual é o tratamento aplicável às importâncias recebidas a título de residência médica ou por estágio remunerado em hospitais, laboratórios, centro de pesquisa e universidades, para complementação de e
- juiz/auditor: **evasiva** · humano: **justificada** · confiança do agente: 0.10
