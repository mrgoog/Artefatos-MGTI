"""Regex, prompt e parser do auditor vêm da origem via importlib (sem chamar modelo)."""

import re

from rag2.auditoria import eh_recusa, modulo_auditor, padrao_recusa, parse_auditor, prompt_auditor


def test_modulo_da_origem_carrega_uma_vez(cfg):
    m = modulo_auditor(cfg)
    assert m is modulo_auditor() and isinstance(padrao_recusa(), re.Pattern)
    assert hasattr(m, "PROMPT_SISTEMA_AUDITOR") and hasattr(m, "TEMPLATE_USUARIO_AUDITOR")


def test_eh_recusa():
    assert eh_recusa("Não é possível responder com base no material fornecido.")
    assert eh_recusa("O contexto fornecido é insuficiente para responder.")
    assert not eh_recusa("Está obrigada a apresentar a declaração.")
    assert not eh_recusa(None) and not eh_recusa("") and not eh_recusa(float("nan"))   # pandas 3: response_text nulo relido como NaN


def test_prompt_e_parse_do_auditor():
    system, user = prompt_auditor(" Q? ", "[DOCUMENTO 1 — X]\ntexto", " GT ", " recuso ")
    assert system.startswith("Você é um auditor") and "VEREDICTOS" in system
    assert "PERGUNTA:\nQ?\n" in user and "[DOCUMENTO 1 — X]\ntexto" in user and "RESPOSTA DE REFERÊNCIA" in user and user.rstrip().endswith("recuperado?")
    assert parse_auditor('{"raciocinio": "faltava", "veredicto": 2}') == ("faltava", 2)
    assert parse_auditor('```json\n{"raciocinio": "ok", "veredicto": 1}\n```') == ("ok", 1)
    assert parse_auditor('{"raciocinio": "cortado no meio') == ("cortado no meio", -1)
    assert parse_auditor('{"raciocinio": "x", "veredicto": 7}')[1] == -1
