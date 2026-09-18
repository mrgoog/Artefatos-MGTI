"""Índice reparado sobre corpus sintético (VetorizadorHash): composição sem pseudo-docs, metadados, registrar uma vez, resolução por degrau."""

import hashlib
import json

import pytest
from ensemble_llm.esquemas import ItemDataset
from ensemble_llm.recuperacao.vetorizacao import VetorizadorHash

from rag2.config import RAIZ, ConfigRag2
from rag2.dados import ItemAvaliado, tipo_fonte
from rag2.escada import FABRICAS, ORDEM_DEGRAUS
from rag2.indice import carregar_indice
from rag2.indices import (
    IndiceJaExiste,
    comparar_atingibilidade,
    construir_indice_reparado,
    construir_subindices,
    dir_subindice,
    dividir_indice,
    resolver_indice,
    sha256_arquivo,
)
from rag2.metricas import Casador
from rag2.pool import indice_id


def _paragrafo(cabecalho: str, n: int = 300) -> str:
    return cabecalho + " " + " ".join(f"palavra{i}" for i in range(n))


NORMAS = [
    {"filename": "Lei nº 1.txt", "filedata": "\n\n".join(_paragrafo(f"Art. {a}º Texto.") for a in (1, 2, 3)),
     "reparado": True, "palavras_antes": 5, "palavras_depois": 909, "fonte": "https://exemplo.gov.br/l1.htm"},
    {"filename": "Lei nº 2.txt", "filedata": _paragrafo("Art. 7º Texto."),
     "reparado": False, "palavras_antes": 301, "palavras_depois": 301, "fonte": ""},
    {"filename": "Decreto nº 3.txt", "filedata": _paragrafo("Art. 12 Texto."),
     "reparado": False, "palavras_antes": 301, "palavras_depois": 301, "fonte": ""},
]
CARF = [
    {"filename": "10000_1.txt", "filedata": _paragrafo("Acórdão um.")},
    {"filename": "10000_2.txt", "filedata": "\n\n".join(_paragrafo(f"Acórdão dois parte {p}.") for p in (1, 2))},
]


def _escrever(tmp_path, normas=NORMAS, carf=CARF):
    n, c = tmp_path / "normas.json", tmp_path / "carf.json"
    n.write_text(json.dumps(normas, ensure_ascii=False), encoding="utf-8")
    c.write_text(json.dumps(carf, ensure_ascii=False), encoding="utf-8")
    return n, c


def _construir(tmp_path, **kw):
    n, c = _escrever(tmp_path)
    return construir_indice_reparado(
        ConfigRag2(), saida=tmp_path / "idx", arquivo_normas=n, arquivo_carf=c,
        vetorizador=VetorizadorHash(dim=16), log=kw.pop("log", lambda s: None), **kw,
    )


def test_construir_indice_sintetico(tmp_path):
    linhas = []
    meta = _construir(tmp_path, log=linhas.append)
    assert linhas == ["documentos: acordao_carf=2 norma=3 | chunks: acordao_carf=3 norma=5 | total=8"]
    assert meta["documentos"] == {"acordao_carf": 2, "norma": 3} and meta["chunks"] == {"acordao_carf": 3, "norma": 5}
    assert (meta["n_chunks"], meta["dim"], meta["sem_pseudo_docs"], meta["indice_id"]) == (8, 16, True, "rag2_reparado")
    assert (meta["chunk_max_tokens"], meta["chunk_overlap_tokens"], meta["tokenizador"]) == (512, 64, "tokenizador_espacos")
    assert meta["sha256_corpus"] == {
        "normas.json": hashlib.sha256((tmp_path / "normas.json").read_bytes()).hexdigest(),
        "carf.json": sha256_arquivo(tmp_path / "carf.json"),
    }
    assert meta["embedder_model"].startswith("vetorizador-hash") and meta["gpu"] is None
    idx_dir = tmp_path / "idx"
    assert json.loads((idx_dir / "index_metadata.json").read_text(encoding="utf-8")) == meta
    idx = carregar_indice(idx_dir)
    assert len(idx) == 8 and idx.vetores.shape == (8, 16) and idx.remover_stopwords is True
    assert "pseudo_doc" not in idx.composicao()["documentos"]
    assert idx.composicao()["chunks"] == {"acordao_carf": 3, "norma": 5}
    assert [c for c in idx.ids_denso if c.startswith("Lei nº 1.txt")] == [f"Lei nº 1.txt#{i:04d}" for i in range(3)]
    assert list(idx.ids_denso[:3]) == ["10000_1.txt#0000", "10000_2.txt#0000", "10000_2.txt#0001"]  # CARF antes das normas, como na produção
    assert "Art. 2º" in idx.texto["Lei nº 1.txt#0001"] and idx.doc["10000_2.txt#0001"] == "10000_2.txt"


def test_registrar_uma_vez(tmp_path):
    meta1 = _construir(tmp_path)
    mtime = (tmp_path / "idx" / "denso" / "vetores.npz").stat().st_mtime_ns
    with pytest.raises(IndiceJaExiste):
        _construir(tmp_path)
    assert (tmp_path / "idx" / "denso" / "vetores.npz").stat().st_mtime_ns == mtime
    meta2 = _construir(tmp_path, sobrescrever=True)
    assert meta2["n_chunks"] == meta1["n_chunks"] and meta2["sha256_corpus"] == meta1["sha256_corpus"]


def test_rejeita_pseudo_docs_e_arquivo_ausente(tmp_path):
    n, c = _escrever(tmp_path, normas=NORMAS + [{"filename": "disp_q_0001_00", "filedata": _paragrafo("Art. 1º")}])
    with pytest.raises(ValueError, match="pseudo"):
        construir_indice_reparado(ConfigRag2(), saida=tmp_path / "idx", arquivo_normas=n, arquivo_carf=c,
                                  vetorizador=VetorizadorHash(dim=16), log=lambda s: None)
    with pytest.raises(FileNotFoundError):
        construir_indice_reparado(ConfigRag2(), saida=tmp_path / "idx2", arquivo_normas=n,
                                  arquivo_carf=tmp_path / "nao_existe.json", vetorizador=VetorizadorHash(dim=16))
    assert not (tmp_path / "idx").exists() and not (tmp_path / "idx2").exists()


def test_resolver_indice_e_diretorio(cfg):
    assert cfg.dir_indice_reparado() == RAIZ / "runs" / "_shared" / "rag" / "rag2_reparado"
    assert resolver_indice(cfg, "R0") == cfg.dir_indice_producao()
    assert resolver_indice(cfg, "R0'") == cfg.dir_indice_reparado()
    assert resolver_indice(cfg, "R1") == cfg.dir_indice_reparado()
    assert cfg.dir_indice_reparado() not in cfg.faltantes()


def test_fabricas_registra_r0_linha():
    assert sorted(FABRICAS) == ["R0", "R0'", "R1", "R2", "R3", "R5"] and "R0'" in ORDEM_DEGRAUS


def test_dividir_indice(indice_sintetico):
    subs = dividir_indice(indice_sintetico)
    assert set(subs) == {"norma", "acordao_carf"}
    assert len(subs["norma"]) == 3 and len(subs["acordao_carf"]) == 2
    assert len(subs["norma"]) + len(subs["acordao_carf"]) == len(indice_sintetico)
    for tipo, sub in subs.items():
        assert {tipo_fonte(d) for d in sub.doc.values()} == {tipo}                      # cada subíndice só tem seu tipo
        assert list(sub.ids_bm25) == list(sub.ids_denso)                                # ordem densa preservada
        assert sub.vetores.shape == (len(sub), indice_sintetico.vetores.shape[1])
        assert sub.bm25.corpus_size == len(sub) and sub.remover_stopwords is True       # BM25 refeito só sobre o subconjunto
    assert list(subs["acordao_carf"].ids_denso) == ["10000_1.txt#0000", "10000_2.txt#0000"]
    assert subs["norma"].diretorio == dir_subindice(indice_sintetico.diretorio, "norma")
    assert subs["norma"].bm25.idf != indice_sintetico.bm25.idf                          # IDF separado, não máscara


def test_construir_subindices_registrar_uma_vez(indice_sintetico, tmp_path):
    # indice_sintetico já construiu o reparado (3 normas + 2 CARF) em tmp_path/idx
    metas = construir_subindices(ConfigRag2(), saida_base=tmp_path / "idx")
    assert metas["norma"]["indice_id"] == "rag2_reparado_normas" and metas["acordao_carf"]["n_chunks"] == 2
    for tipo, sufixo, n in (("norma", "normas", 3), ("acordao_carf", "carf", 2)):
        d = tmp_path / f"idx_{sufixo}"
        idx = carregar_indice(d)
        assert len(idx) == n and {tipo_fonte(x) for x in idx.doc.values()} == {tipo}
        assert indice_id(d) == f"rag2_reparado_{sufixo}"
    mtime = (tmp_path / "idx_normas" / "denso" / "vetores.npz").stat().st_mtime_ns
    with pytest.raises(IndiceJaExiste):
        construir_subindices(ConfigRag2(), saida_base=tmp_path / "idx")                 # registrar uma vez
    assert (tmp_path / "idx_normas" / "denso" / "vetores.npz").stat().st_mtime_ns == mtime


def _item(iid, esperados):
    return ItemAvaliado(
        item=ItemDataset(item_id=iid, pergunta=f"pergunta {iid}", ground_truth="g", has_doc_ref=True),
        esperados=frozenset(esperados),
    )


def test_comparar_atingibilidade():
    antes = Casador({"Lei A.txt#0000": "Art. 1º Preâmbulo."}, {"Lei A.txt#0000": "Lei A.txt"})
    depois = Casador(
        {"Lei A.txt#0000": "Art. 1º Preâmbulo.", "Lei A.txt#0001": "Art. 25 Texto.", "Lei B.txt#0000": "Art. 3º X."},
        {"Lei A.txt#0000": "Lei A.txt", "Lei A.txt#0001": "Lei A.txt", "Lei B.txt#0000": "Lei B.txt"},
    )
    itens = [_item("q_0001", {("Lei A.txt", "1"), ("Lei A.txt", "25")}), _item("q_0002", {("Lei A.txt", "25"), ("Lei C.txt", "9")})]
    assert comparar_atingibilidade(itens, antes, depois) == {
        "n_pares": 4, "atingiveis_antes": 1, "atingiveis_depois": 3, "ganhos": 2, "perdas": 0,
    }
    assert comparar_atingibilidade(itens, depois, antes)["perdas"] == 2


@pytest.mark.gpu
def test_degrau_r0_linha_limite(cfg, capsys):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sem CUDA")
    if not (cfg.dir_indice_reparado() / "bm25" / "bm25.pkl").exists():
        pytest.skip("índice reparado não construído (rag2 construir-indice reparado)")
    from rag2.cli import main

    assert main(["degrau", "R0'", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert "degrau=R0'  itens=5" in out and "pseudo_doc" not in out
    assert "índice: " + str(cfg.dir_indice_reparado()) in out and "debug-temp" in out
