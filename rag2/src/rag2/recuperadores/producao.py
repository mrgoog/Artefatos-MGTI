"""R0: o recuperador híbrido de produção (BM25 k=5 + denso k=5, união, dedup 1 chunk/doc) sobre um índice persistido."""

from __future__ import annotations

from pathlib import Path

from ensemble_llm.esquemas import BlocoRecuperado
from ensemble_llm.recuperacao.construtor_corpus import carregar_recuperador_hibrido
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

from rag2.constantes import MODELO_EMBEDDER, TOP_K_PRODUCAO
from rag2.recuperadores.base import validar_saida


class RecuperadorProducao:
    """Compõe RecuperadorHibrido da origem sem alterá-lo; R0 devolve até 2·top_k chunks (união com dedup)."""

    def __init__(self, indice_dir: Path, device: str, *, top_k: int = TOP_K_PRODUCAO, degrau: str = "R0") -> None:
        self.degrau = degrau
        self.top_k = top_k
        vetorizador = VetorizadorSentenceTransformer(
            nome_modelo=MODELO_EMBEDDER, tamanho_lote=32, dispositivo=device
        )
        self._hibrido = carregar_recuperador_hibrido(
            Path(indice_dir), vetorizador=vetorizador, top_k_denso=top_k, top_k_bm25=top_k
        )

    def recuperar(self, consulta: str) -> list[BlocoRecuperado]:
        """Delega à produção e valida a forma da saída."""
        return validar_saida(self._hibrido.recuperar(consulta), n_max=2 * self.top_k)
