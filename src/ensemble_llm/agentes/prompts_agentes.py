"""Templates de prompt dos agentes."""

from __future__ import annotations

from ensemble_llm.esquemas import BlocoRecuperado

PROMPT_SISTEMA_AGENTE = """
Você é um assistente especializado em legislação tributária brasileira (IRPF).

Sua tarefa é responder à pergunta do usuário com base EXCLUSIVAMENTE nos 
dispositivos legais e ementas do CARF fornecidos como contexto. 
Siga rigorosamente estas regras:

1. FUNDAMENTE cada afirmação no contexto fornecido. Cite os dispositivos 
ou ementas que sustentam sua resposta.
2. NÃO cite, mencione ou invoque leis, artigos, decretos, instruções 
normativas, súmulas ou qualquer normativo que NÃO esteja presente no 
contexto abaixo — mesmo que você acredite que sejam corretos.
3. Se o contexto for insuficiente para responder completamente, responda 
apenas o que o contexto permite e indique uma confiança baixa.
4. Se o contexto não contiver nenhuma informação relevante para a pergunta, 
responda que não é possível responder com base no material fornecido e 
atribua confiança próxima de 0.

Responda em português, de forma técnica mas clara.
"""

TEMPLATE_USUARIO_AGENTE = """\
CONTEXTO RECUPERADO (use APENAS este material para fundamentar sua resposta):
{retrieved_context}

PERGUNTA:
{question}

INSTRUÇÕES DE FORMATO:
Responda EXCLUSIVAMENTE em JSON válido com o formato exato abaixo. 
Não inclua texto antes ou depois do JSON. Não use blocos de código markdown.

FORMATO:
{{
  "resposta": "<sua resposta substantiva, fundamentada apenas no contexto acima>",
  "confianca": <número entre 0.0 e 1.0 indicando sua confiança na resposta>
}}
"""

DICA_SCHEMA_AGENTE = """{
  "resposta": "<texto>",
  "confianca": 0.85
}"""


def formatar_chunks_recuperados(chunks: list[BlocoRecuperado]) -> str:
    """Formata lista de chunks recuperados como texto numerado para inserção no prompt do agente."""
    if not chunks:
        return "(Nenhum documento recuperado para esta pergunta.)"
    partes = []
    for chunk in chunks:
        cabecalho = f"[DOCUMENTO {chunk.combined_rank} — {chunk.source_doc}]"
        partes.append(f"{cabecalho}\n{chunk.text.strip()}")
    return "\n\n".join(partes)


def construir_prompt_agente(question: str, chunks: list[BlocoRecuperado]) -> tuple[str, str]:
    """Monta (system_prompt, user_prompt) para invocação do agente com contexto RAG."""
    usuario = TEMPLATE_USUARIO_AGENTE.format(
        retrieved_context=formatar_chunks_recuperados(chunks),
        question=question.strip(),
    )
    return PROMPT_SISTEMA_AGENTE, usuario
