# Inventário dos artefatos bibliométricos

Este diretório reúne os dados e produtos de trabalho do mapeamento bibliométrico descrito no Capítulo 2 da dissertação. Esta nota identifica os arquivos que sustentam os números apresentados no manuscrito. Materiais intermediários e históricos do estudo não foram publicados.

## Fontes canônicas

| Arquivo | Função | Verificação local |
|---|---|---|
| `scopus-andre.csv` | Exportação do corpus final (versão reduzida, ver abaixo) | 2.924 registros, sem contar o cabeçalho |
| `Bibliometrix-Export-File-2025-06-23.xlsx` | Corpus preparado para o Bibliometrix (versão reduzida, ver abaixo) | 2.924 registros; metadado de criação em 23/06/2025 |
| `informações corpus - bibliometrix.xlsx` | Indicadores descritivos produzidos no Bibliometrix | 1.097 fontes, 9.272 autores e demais indicadores da Tabela 1 |
| `andre-5.gml` | Rede final de coocorrência de palavras-chave | 178 nós, 1.724 arestas e sete agrupamentos do VOSviewer |
| `tesauro.txt` | Normalização terminológica usada na rede | Arquivo de tesauro do VOSviewer |
| `andre.gephi` | Projeto de análise da rede | Métricas calculadas após a remoção dos dois termos dominantes |
| `centralidade autovetor.xlsx` | Medidas de centralidade apresentadas no texto | Saída tabular da análise de rede |
| `TrendTopics.xlsx` | Ocorrências e mediana temporal dos termos | Saída tabular da análise de tendências |
| `10-most-cited.csv` | Lista qualitativa final: mais citados | 10 registros |
| `10-most-relevant.csv` | Lista qualitativa final: mais relevantes | 10 registros |
| `10-most-recent.csv` | Lista qualitativa final: mais recentes | 10 registros |

## Versão reduzida da exportação

`scopus-andre.csv` e `Bibliometrix-Export-File-2025-06-23.xlsx` são publicados em versão reduzida. Foram retirados os resumos, as referências citadas, os endereços de correspondência (que contêm e-mails de autores) e os campos derivados deles (`AB`, `AB_raw`, `CR`, `RP`, `C1`, `C1raw`, `AU1_UN` na planilha), além de campos sem uso na análise. Permanecem identificação (EID, DOI, título), ano, fonte, tipo documental, autores e seus identificadores, afiliações, palavras-chave e citações recebidas: 27 das 44 colunas do CSV e 35 das 55 da planilha. Os registros são os mesmos 2.924.

A versão reduzida reproduz diretamente a distribuição anual, as 1.097 fontes, os 9.272 autores, os 6.133 termos de palavras-chave de autores, os 164 documentos de autoria única, a média de 5,684 citações por documento e a composição por tipo documental. Os indicadores que dependem dos campos retirados, como as 173.007 referências citadas e a coautoria internacional, constam de `informações corpus - bibliometrix.xlsx`, produzido no Bibliometrix a partir da exportação completa.

As três listas qualitativas somam 30 posições e 27 trabalhos distintos. Um trabalho aparece nas três listas, e outro aparece nas listas de mais relevantes e mais recentes.

## Decisões de reconciliação

O CSV bruto registra a seguinte distribuição anual: 5 documentos em 2022, 163 em 2023, 1.644 em 2024 e 1.112 em 2025. Esses valores somam 2.924 e constituem a fonte de verdade adotada na dissertação.

O arquivo `evolucao anual.xlsx`, produzido em uma etapa derivada, registra 1.645 documentos em 2024 e 1.111 em 2025. Como há uma diferença de um registro entre esses dois anos, os valores dessa planilha não foram usados na versão revisada do texto.

A rede completa preservada em `andre-5.gml` tem 178 nós e 1.724 arestas. As medidas do Gephi foram calculadas depois da remoção dos nós “large language model” e “artificial intelligence”. A rede reduzida tem 176 nós e 1.397 arestas, o que corresponde ao grau médio 15,875 registrado no projeto. O projeto do Gephi informa ainda diâmetro 4, modularidade 0,216, quatro comunidades e coeficiente médio de agrupamento 0,489.
