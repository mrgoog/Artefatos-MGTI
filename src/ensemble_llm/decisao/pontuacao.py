"""Pontuação multiplicativa, seleção direta e cálculo de incerteza."""

from __future__ import annotations


def calcular_pontuacoes_agentes(
    pesos: dict[str, float],
    confiancas_calibradas: dict[str, float],
) -> dict[str, float]:
    """s_i = w_i · ĉ_i para cada agente i."""
    if set(pesos.keys()) != set(confiancas_calibradas.keys()):
        raise ValueError(
            f"Conjuntos de agentes divergem: pesos={set(pesos)} "
            f"vs confiancas_calibradas={set(confiancas_calibradas)}"
        )
    return {a: pesos[a] * confiancas_calibradas[a] for a in pesos}


def selecionar_vencedor(
    pontuacoes: dict[str, float],
    pesos: dict[str, float],
    desempate: str = "maior_peso",
) -> str:
    """Seleciona o agente vencedor com regra de desempate."""
    if not pontuacoes:
        raise ValueError("pontuacoes não pode ser vazio")

    pontuacao_max = max(pontuacoes.values())
    candidatos = [a for a, p in pontuacoes.items() if p == pontuacao_max]

    if len(candidatos) == 1:
        return candidatos[0]

    if desempate in {"maior_peso", "highest_weight"}:
        return max(candidatos, key=lambda a: (pesos[a], a))
    if desempate in {"lexicografico", "lexicographic"}:
        return min(candidatos)

    raise ValueError(f"Desempate desconhecido: {desempate!r}")


def calcular_eta(pontuacoes: dict[str, float]) -> float:
    """η(q) = 1 - max_i s_i(q)."""
    if not pontuacoes:
        raise ValueError("pontuacoes não pode ser vazio")
    return 1.0 - max(pontuacoes.values())


def decidir(
    confiancas_brutas: dict[str, float],
    confiancas_calibradas: dict[str, float],
    pesos: dict[str, float],
    *,
    metodo: str = "weighted_calibrated",
    desempate: str = "highest_weight",
) -> tuple[str, float]:
    """Cabeça de decisão: devolve (vencedor, η) segundo o método escolhido.

    - ``weighted_calibrated``: s_i = w_i·ĉ_i; vencedor = argmax_i s_i; η = 1 − max_i s_i.
    - ``mean_raw`` (pipeline minimalista): vencedor = argmax_i c_bruta_i;
      η = 1 − média_i(c_bruta_i). Ignora pesos e calibração — a média das confianças
      cruas remove o winner's curse do ``max``.

    A chave ``decision.method`` no YAML alterna entre os dois métodos.
    """
    if metodo == "weighted_calibrated":
        pontuacoes = calcular_pontuacoes_agentes(pesos, confiancas_calibradas)
        return selecionar_vencedor(pontuacoes, pesos, desempate), calcular_eta(pontuacoes)
    if metodo == "mean_raw":
        if not confiancas_brutas:
            raise ValueError("confiancas_brutas não pode ser vazio")
        c_max = max(confiancas_brutas.values())
        vencedor = min(a for a, c in confiancas_brutas.items() if c == c_max)
        eta = 1.0 - sum(confiancas_brutas.values()) / len(confiancas_brutas)
        return vencedor, eta
    raise ValueError(f"método de decisão desconhecido: {metodo!r}")
