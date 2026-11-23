"""Varredura offline de configurações de recuperação (custo zero em dólares).

Mede **recall de passagem** — arquivo E artigo corretos — para várias configurações
de query-time, sobre o índice já persistido. Nenhum agente, nenhum juiz, nenhum
rebuild de índice. Serve para decidir se vale pagar o re-run antes de pagá-lo.

## O que a varredura testa, e por quê

**(1) `max_por_doc` — o eixo principal.**

`recuperacao/hibrido.py:71-84` deduplica por `source_doc`: devolve NO MÁXIMO UM chunk
por documento. O docstring diz que é para "maximizar a cobertura documental".

Mas o RIR (Decreto 9.580) tem 483 chunks. Se o ranqueamento põe cinco deles no topo,
quatro são descartados. Se o artigo necessário não for o único sobrevivente, é
ESTRUTURALMENTE IMPOSSÍVEL recuperá-lo.

Isso é Goodhart: o instrumento `retrieval_recall` premiava cobertura de DOCUMENTO
(ele casa nomes de arquivo, `retrieval_recall.py:221`), e o retriever foi desenhado
para maximizar cobertura de DOCUMENTO. Métrica e mecanismo erraram junto.

**(2) `top_k` — o orçamento de recuperação.**

Hoje: ~7 chunks por pergunta, para achar ~3 dispositivos espalhados por vários
arquivos, num corpus de 7.682 documentos.

**(3) denso vs BM25 vs híbrido.**

Decomposição da falha atual (por dispositivo esperado, n=1.742):
    acerto (arquivo E artigo)                : 13.5%
    falha de ARQUIVO (nem chegou perto)      : 61.0%
    falha de ARTIGO (achou o doc, errou a região) : 25.4%

Uso:
    uv run python scripts/varredura_recuperacao.py --config configs/rag_brtaxq.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from recall_passagem import _regex_artigo, dispositivos_esperados

from ensemble_llm.configuracao import carregar_config_rag_offline
from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r
from ensemble_llm.recuperacao.construtor_corpus import carregar_recuperador_hibrido
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

logging.basicConfig(level=logging.WARNING)


@dataclass(frozen=True)
class Config:
    """Uma configuração de recuperação de query-time."""

    nome: str
    top_k_denso: int
    top_k_bm25: int
    max_por_doc: int  # 1 = comportamento atual (dedup total por documento)


def recuperar(
    retriever: object, consulta: str, *, max_por_doc: int, limite: int
) -> list[tuple[str, str]]:
    """Reimplementa `RecuperadorHibrido.recuperar` com `max_por_doc` configurável.

    Deliberadamente NÃO modifica o pacote: a execução normal do pipeline fica
    intacta e rastreável. A ordenação é idêntica à do original; só a regra de
    deduplicação por `source_doc` é relaxada.

    Devolve [(source_doc, text)] na ordem do ranking.
    """
    denso = dict(retriever.indice_denso.buscar(consulta, top_k=retriever.top_k_denso))  # type: ignore[attr-defined]
    bm25 = dict(retriever.indice_bm25.buscar(consulta, top_k=retriever.top_k_bm25))  # type: ignore[attr-defined]
    if retriever._lookup_chunks is None:  # type: ignore[attr-defined]
        retriever._lookup_chunks = retriever._construir_lookup()  # type: ignore[attr-defined]
    lookup = retriever._lookup_chunks  # type: ignore[attr-defined]

    def chave(cid: str) -> tuple[int, float, str]:
        em_ambos = cid in denso and cid in bm25
        return (0 if em_ambos else 1, -denso.get(cid, -float("inf")), cid)

    vistos: dict[str, int] = {}
    saida: list[tuple[str, str]] = []
    for cid in sorted(set(denso) | set(bm25), key=chave):
        chunk = lookup.get(cid)
        if chunk is None:
            continue
        if vistos.get(chunk.source_doc, 0) >= max_por_doc:
            continue
        vistos[chunk.source_doc] = vistos.get(chunk.source_doc, 0) + 1
        saida.append((chunk.source_doc, chunk.text))
        if len(saida) >= limite:
            break
    return saida


def avaliar(
    recuperados: list[tuple[str, str]], esperados: set[tuple[str, str]]
) -> dict[str, float]:
    """Recall de ARQUIVO (instrumento atual) e de PASSAGEM (arquivo E artigo)."""
    arquivos = {sd for sd, _ in recuperados}
    hits_arq = hits_pas = 0
    for arquivo, artigo in esperados:
        if arquivo not in arquivos:
            continue
        hits_arq += 1
        rx = _regex_artigo(artigo)
        if any(sd == arquivo and rx.search(txt) for sd, txt in recuperados):
            hits_pas += 1
    n = len(esperados)
    return {
        "recall_arquivo": hits_arq / n,
        "recall_passagem": hits_pas / n,
        "hit_passagem": float(hits_pas > 0),
        "n_chunks": float(len(recuperados)),
        "n_chars": float(sum(len(t) for _, t in recuperados)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rag_brtaxq.yaml")
    args = p.parse_args()

    cfg = carregar_config_rag_offline(args.config)
    rag_root = Path(cfg.retrieval.persisted_index_dir)
    questoes = Path(cfg.dataset.root_dir) / cfg.dataset.questions_file

    itens = [i for i in carregar_br_taxqa_r(questoes) if i.has_doc_ref]
    alvo = [
        (i, dispositivos_esperados(i.dispositivos_legais + i.ementas_carf)) for i in itens
    ]
    alvo = [(i, e) for i, e in alvo if e]
    print(f"itens com dispositivos anotados: {len(alvo)}\n")

    embedder = VetorizadorSentenceTransformer(
        nome_modelo=cfg.retrieval.embedder_model,
        tamanho_lote=cfg.retrieval.embedder_batch_size,
        dispositivo=cfg.retrieval.embedder_device,
    )
    # Carrega os índices UMA vez com o maior top_k; as variantes só reordenam/filtram.
    retriever = carregar_recuperador_hibrido(
        rag_root, vetorizador=embedder, top_k_denso=50, top_k_bm25=50
    )
    print(f"índice carregado de {rag_root}\n")

    configs = [
        # --- baseline: exatamente o que o pipeline faz hoje ---
        Config("ATUAL (k=5+5, 1 chunk/doc)", 5, 5, 1),
        # --- eixo 1: relaxar a deduplicação por documento (mesmo orçamento) ---
        Config("dedup relaxada: 3 chunks/doc", 5, 5, 3),
        Config("dedup relaxada: sem limite  ", 5, 5, 99),
        # --- eixo 2: aumentar o orçamento de recuperação ---
        Config("k=15+15, 1 chunk/doc       ", 15, 15, 1),
        Config("k=15+15, 3 chunks/doc      ", 15, 15, 3),
        Config("k=30+30, 1 chunk/doc       ", 30, 30, 1),
        Config("k=30+30, 3 chunks/doc      ", 30, 30, 3),
        Config("k=50+50, sem limite        ", 50, 50, 99),
        # --- eixo 3: denso puro vs BM25 puro (com dedup relaxada) ---
        Config("só DENSO k=30, 3 chunks/doc", 30, 1, 3),
        Config("só BM25  k=30, 3 chunks/doc", 1, 30, 3),
    ]

    print(f"{'configuração':30s} {'Recall@K':>9s} {'Hit@K':>7s} {'R.arquivo':>10s} "
          f"{'chunks':>7s} {'~tokens':>8s}")
    print("=" * 80)
    linhas = []
    for c in configs:
        retriever.top_k_denso = c.top_k_denso
        retriever.top_k_bm25 = c.top_k_bm25
        # limite de chunks servidos ao agente: mantém o orçamento comparável ao atual
        limite = c.top_k_denso + c.top_k_bm25
        res = [
            avaliar(
                recuperar(retriever, i.pergunta, max_por_doc=c.max_por_doc, limite=limite),
                e,
            )
            for i, e in alvo
        ]
        d = pd.DataFrame(res)
        linha = {
            "config": c.nome.strip(),
            "recall_passagem": d.recall_passagem.mean(),
            "hit_passagem": d.hit_passagem.mean(),
            "recall_arquivo": d.recall_arquivo.mean(),
            "n_chunks": d.n_chunks.mean(),
            "tokens_aprox": d.n_chars.mean() / 4,
        }
        linhas.append(linha)
        print(f"{c.nome:30s} {linha['recall_passagem']:9.3f} {linha['hit_passagem']:7.3f} "
              f"{linha['recall_arquivo']:10.3f} {linha['n_chunks']:7.1f} "
              f"{linha['tokens_aprox']:8.0f}")

    saida = rag_root / "metrics" / "varredura_recuperacao.parquet"
    saida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_parquet(saida)
    print("\n" + "=" * 80)
    base = linhas[0]["recall_passagem"]
    melhor = max(linhas, key=lambda x: x["recall_passagem"])
    print(f"baseline (pipeline atual): Recall@K de passagem = {base:.3f}")
    print(f"melhor configuração      : {melhor['config']} → {melhor['recall_passagem']:.3f} "
          f"({melhor['recall_passagem'] / max(base, 1e-9):.2f}x)")
    print(f"custo em contexto        : {melhor['tokens_aprox']:.0f} tokens "
          f"(atual: {base and linhas[0]['tokens_aprox']:.0f})")
    print(f"\nsalvo em {saida}")


if __name__ == "__main__":
    main()
