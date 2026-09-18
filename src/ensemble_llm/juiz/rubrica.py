"""Rubrica fixa do juiz LLM."""

from __future__ import annotations

PROMPT_SISTEMA_JUIZ = """Você é um avaliador especializado em legislação tributária brasileira.

Sua tarefa é AVALIAR se uma resposta candidata é CONSISTENTE com o ground truth e os dispositivos normativos fornecidos.

REGRA FUNDAMENTAL: Você NÃO deve usar seu conhecimento próprio sobre direito tributário brasileiro. Baseie-se EXCLUSIVAMENTE no material fornecido no prompt (ground truth, dispositivos legais, ementas CARF). Mesmo que você acredite saber a resposta correta, julgue apenas pela consistência com o material apresentado.

CRITÉRIO DE ADEQUAÇÃO (veredicto = 1):
A resposta candidata é ADEQUADA se, e somente se, satisfaz TODOS os pontos abaixo:
  1. Endereça a pergunta diretamente (não é evasiva nem fora do tópico).
  2. É consistente com o ground truth nos pontos materiais (não contradiz a referência).
  3. Não introduz informação contraditória aos dispositivos legais e ementas fornecidos.

CRITÉRIO DE INADEQUAÇÃO (veredicto = 0):
A resposta candidata é INADEQUADA se viola QUALQUER um dos pontos acima, OU se:
  - Está tão incompleta que não esgota o que a pergunta exige.
  - Recusa-se a responder sem justificativa baseada no contexto.

INSTRUÇÕES DE PROCESSO:
- Explicite seu raciocínio antes do veredicto (chain-of-thought curto).
- Não considere qualidade estilística — apenas adequação substantiva.
- Em caso de dúvida razoável, prefira veredicto = 0 (conservadorismo).

INSTRUÇÕES DE FORMATO:
Responda EXCLUSIVAMENTE em JSON válido com o formato exato abaixo. Não inclua texto antes ou depois do JSON. Não use blocos de código markdown.

FORMATO:
{
  "raciocinio": "<sua análise em 2-4 frases, comparando candidata vs ground truth>",
  "veredicto": <0 ou 1>
}
"""

TEMPLATE_USUARIO_JUIZ = """PERGUNTA:
{question}

GROUND TRUTH (referência oficial):
{ground_truth}

DISPOSITIVOS LEGAIS:
{dispositivos}

EMENTAS CARF:
{ementas}

RESPOSTA CANDIDATA A AVALIAR:
{candidate_response}

Avalie a resposta candidata segundo a rubrica.
"""

DICA_SCHEMA_JUIZ = """{
  "raciocinio": "A resposta candidata afirma X, mas o ground truth estabelece Y. Há contradição em ponto material...",
  "veredicto": 0
}"""


def _formatar_lista(itens: list[str], cabecalho: str = "") -> str:
    """Formata lista de textos com cabeçalho numerado para inserção no prompt do juiz."""
    if not itens:
        return "(Nenhum item fornecido para esta pergunta.)"
    if cabecalho:
        return "\n\n".join(f"{cabecalho} {i+1}:\n{item.strip()}" for i, item in enumerate(itens))
    return "\n\n".join(item.strip() for item in itens)


def construir_prompt_juiz(
    question: str,
    ground_truth: str,
    dispositivos_legais: list[str],
    ementas_carf: list[str],
    candidate_response: str,
) -> tuple[str, str]:
    """Monta (system_prompt, user_prompt) de avaliação com rubrica fixa para o juiz frontier."""
    usuario = TEMPLATE_USUARIO_JUIZ.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        dispositivos=_formatar_lista(dispositivos_legais, "Dispositivo"),
        ementas=_formatar_lista(ementas_carf, "Ementa"),
        candidate_response=candidate_response.strip(),
    )
    return PROMPT_SISTEMA_JUIZ, usuario
