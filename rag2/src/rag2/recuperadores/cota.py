"""R3: um RecuperadorReranker por subíndice (normas × CARF, IDF separado), cota fixa por tipo, contexto final reordenado por rerank_score."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ensemble_llm.esquemas import BlocoRecuperado

from rag2.constantes import N_FINAL
from rag2.esquemas import BlocoReranqueado
from rag2.recuperadores.base import validar_saida
from rag2.recuperadores.reranqueado import RecuperadorReranker


class RecuperadorCota:
    """por_tipo: {tipo → RecuperadorReranker sobre o subíndice}; cada um serve até a sua cota; o contexto final são os ≤ N_FINAL reordenados por rerank_score desc."""

    def __init__(
        self,
        por_tipo: Mapping[str, RecuperadorReranker],
        cotas: Mapping[str, int],
        *,
        n_final: int = N_FINAL,
        degrau: str = "R3",
        log: Callable[[str], None] = print,
    ) -> None:
        if set(por_tipo) != set(cotas):
            raise ValueError(f"tipos de por_tipo {sorted(por_tipo)} ≠ tipos de cotas {sorted(cotas)}")
        for tipo, rec in por_tipo.items():
            if rec.n_final != cotas[tipo]:
                raise ValueError(f"n_final do recuperador de {tipo} ({rec.n_final}) ≠ cota ({cotas[tipo]})")
        self.por_tipo = por_tipo
        self.cotas = cotas
        self.n_final = n_final
        self.degrau = degrau
        self.log = log

    def preaquecer(self, consultas: Sequence[str]) -> dict[str, int]:
        """Pré-aquece pools e scores de cada subíndice (na ordem de por_tipo); devolve {tipo: consultas pontuadas}."""
        return {tipo: rec.preaquecer(consultas) for tipo, rec in self.por_tipo.items()}

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]:
        """Concatena os ≤ cota de cada tipo (ordem de por_tipo), reordena por rerank_score desc (estável) e reindexa combined_rank 1..n.

        Desempate: `sorted` estável — entre tipos, a ordem de inserção de por_tipo (normas antes de CARF);
        dentro de um tipo, a ordem do sub-recuperador (rerank_score desc). Uma cota não atingida (subíndice
        com menos chunks que a cota) deixa o slot vazio: não é preenchida pelo outro tipo (isso mudaria a variável).
        """
        candidatos: list[BlocoReranqueado] = [
            bl for rec in self.por_tipo.values() for bl in rec.recuperar(consulta)  # type: ignore[misc]
        ]
        candidatos.sort(key=lambda bl: -bl.rerank_score)
        escolhidos = candidatos[: self.n_final]
        finais = [bl.model_copy(update={"combined_rank": i}) for i, bl in enumerate(escolhidos, 1)]
        return validar_saida(finais, n_max=self.n_final)
