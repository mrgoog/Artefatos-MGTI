"""GPU: a busca em lote reproduz a união 50+50 do diagnóstico no índice de produção; R1 roda pela CLI em subconjunto."""

import pytest
import torch
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

from rag2.constantes import MODELO_EMBEDDER, POOL_POR_RAMO
from rag2.dados import carregar_itens, itens_com_dispositivo
from rag2.indice import carregar_indice
from rag2.metricas import Casador, avaliar_itens
from rag2.pool import CachePools, caminho_cache
from rag2.recuperadores.fusao import RecuperadorRRF

pytestmark = pytest.mark.gpu


@pytest.mark.lento
@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_uniao_50_50_reproduz_diagnostico(cfg):
    """reference/diagnostico_recall_pool.md §2: união k=50 → recall de passagem 0,5109 e 87,3 chunks/item (índice de produção)."""
    dir_indice = cfg.dir_indice_producao()
    idx = carregar_indice(dir_indice)
    itens = itens_com_dispositivo(carregar_itens(cfg))
    cache = CachePools(caminho_cache(dir_indice, POOL_POR_RAMO))
    assert cache.caminho.name == f"{dir_indice.name}_k50.parquet"
    vet = VetorizadorSentenceTransformer(nome_modelo=MODELO_EMBEDDER, tamanho_lote=32, dispositivo=cfg.device)
    RecuperadorRRF(idx, cache, vet, log=lambda s: None).preaquecer([it.pergunta for it in itens])
    servidos = {}
    for it in itens:
        p = cache.pools(it.pergunta)
        servidos[it.item_id] = list(dict.fromkeys([c for c, _ in p["denso"]] + [c for c, _ in p["bm25"]]))
    df = avaliar_itens(itens, servidos, Casador(idx.texto, idx.doc))
    assert df["recall_passagem"].mean() == pytest.approx(0.5109, abs=0.005)
    assert df["n_chunks"].mean() == pytest.approx(87.3, abs=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_degrau_r1_limite(cfg, capsys):
    if not (cfg.dir_indice_reparado() / "bm25" / "bm25.pkl").exists():
        pytest.skip("índice reparado não construído (rag2 construir-indice reparado)")
    from rag2.cli import main

    assert main(["degrau", "R1", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert "pools: " in out and "degrau=R1  itens=5" in out and "chunks/item=7.00" in out and "pseudo_doc" not in out
    assert "índice: " + str(cfg.dir_indice_reparado()) in out and "debug-temp" in out
