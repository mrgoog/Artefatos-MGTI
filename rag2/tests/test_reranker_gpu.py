"""GPU: o cross-encoder real em pares do índice reparado (q_0371) — escala, discriminação, textos idênticos, determinismo; R2 pela CLI em subconjunto."""

import numpy as np
import pandas as pd
import pytest
import torch

from rag2.config import RAIZ
from rag2.constantes import MODELO_RERANKER
from rag2.dados import carregar_itens
from rag2.indice import carregar_indice
from rag2.reranker import Pontuador, Reranker

pytestmark = pytest.mark.gpu

ACORDAO_LENTE = "10730010769201025_6982101.txt#0004"            # acórdão sobre lente intraocular; 1º de q_0371 pós-reranker (medido 0,9999)
NORMA_ART94_OUTRO_ASSUNTO = "Instrução Normativa RFB nº 1.500.txt#0034"   # casa o regex do art. 94, trata de serviços de registro (medido 0,0134)
FRASE_ALUGUEL = "Art. 7º Rendimentos de aluguel entre pai e filho."       # irrelevante (medido 0,0000)
TEXTO_IDENTICO = (                                               # dois chunk_id, um texto (1.839 casos no índice)
    "Parecer SEI Nº 110 2018 CRJPGACETPGFN-MF, aprovado pelo Despacho nº 3482020PGFN-ME, de 26 de agosto de 2020.txt#0000",
    "Parecer SEI nº 110 2018 CRJPGACETPGFN-MF.txt#0000",
)


def _indice_reparado_ou_skip(cfg):
    if not (cfg.dir_indice_reparado() / "bm25" / "bm25.pkl").exists():
        pytest.skip("índice reparado não construído (rag2 construir-indice reparado)")
    return carregar_indice(cfg.dir_indice_reparado())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_reranker_real_discrimina(cfg):
    idx = _indice_reparado_ou_skip(cfg)
    q = next(it.pergunta for it in carregar_itens(cfg) if it.item_id == "q_0371")
    assert "lente intraocular" in q
    rr = Reranker(cfg.device)
    assert isinstance(rr, Pontuador) and rr.nome == MODELO_RERANKER and rr._modelo is None   # lazy até a primeira chamada
    assert rr.pontuar(q, []).shape == (0,) and rr._modelo is None
    textos = [idx.texto[ACORDAO_LENTE], idx.texto[NORMA_ART94_OUTRO_ASSUNTO], FRASE_ALUGUEL, idx.texto[TEXTO_IDENTICO[0]], idx.texto[TEXTO_IDENTICO[1]]]
    assert textos[3] == textos[4]
    s = rr.pontuar(q, textos)
    assert rr.max_seq_length == 8192 and next(rr._modelo.model.parameters()).dtype == torch.float16
    assert s.dtype == np.float32 and s.shape == (5,) and bool(((s >= 0) & (s <= 1)).all())
    assert s[0] > 0.9 and s[1] < 0.1 and s[2] < 0.01                      # relevante ≫ cabeçalho de outro assunto ≫ irrelevante
    assert s[3] == s[4]                                                   # textos idênticos → score idêntico
    assert np.array_equal(rr.pontuar(q, textos), s)                       # mesma lista, mesma chamada → Δ = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_degrau_r2_limite(cfg, capsys):
    _indice_reparado_ou_skip(cfg)
    from rag2.cli import main

    assert main(["degrau", "R2", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert "pools: " in out and "reranker: " in out and "modelo=BAAI/bge-reranker-v2-m3" in out
    assert "degrau=R2  itens=5" in out and "chunks/item=7.00" in out and "pseudo_doc" not in out
    assert "índice: " + str(cfg.dir_indice_reparado()) in out and "debug-temp" in out
    d = pd.read_parquet(RAIZ / "debug-temp" / "escada" / "R2" / "servidos.parquet")
    assert list(d.columns)[-2:] == ["rrf_score", "rerank_score"] and len(d) == 35 and d.rerank_score.between(0, 1).all()
