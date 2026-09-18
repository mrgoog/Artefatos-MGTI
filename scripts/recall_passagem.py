"""Recall de PASSAGEM — o recall honesto da recuperação (custo zero).

Um dispositivo (file, artigo) é HIT apenas se algum chunk recuperado vem do arquivo
certo E contém o artigo certo — recall em granularidade de passagem, não de arquivo.
Compara os dois instrumentos (passagem e arquivo) lado a lado. Nada é modificado —
leitura pura.

Uso:
    uv run python scripts/recall_passagem.py --run phase2_local_full_715
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r

INDICE_RAG = Path("runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/denso/chunks.parquet")
QUESTOES = Path("data/brtaxq/questions_QA_2024_v1.1.json")


def _normalizar_artigo(artigo: str) -> str:
    """'1º' -> '1'; '12-A' -> '12-A'; '  25 ' -> '25'."""
    a = unicodedata.normalize("NFKD", str(artigo)).strip()
    a = re.sub(r"[ºo°\.]+$", "", a).strip()
    return a.upper()


def _regex_artigo(artigo: str) -> re.Pattern[str]:
    """Casa o cabeçalho do artigo no texto do chunk ('Art. 25', 'Artigo 25', 'Art. 12-A').

    O lookahead `(?!\\d)` impede que 'Art. 25' case dentro de 'Art. 250', mas — ao
    contrário da versão anterior `(?![\\dº°o])` — NÃO exclui o marcador ordinal, de
    modo que 'Art. 5º' e 'Art. 1º' casam para os artigos '5' e '1'. Como os artigos
    1º–9º são grafados com ordinal em toda a legislação brasileira, a régua antiga
    perdia sistematicamente os dispositivos de dígito único (subestimava o recall).
    """
    num = re.escape(_normalizar_artigo(artigo))
    return re.compile(
        rf"art(?:igo)?\.?\s*0*{num}(?!\d)",
        re.IGNORECASE,
    )


def dispositivos_esperados(bruto: list[str]) -> set[tuple[str, str]]:
    """Extrai pares (arquivo, artigo) do JSON de anotação do curador."""
    esperados: set[tuple[str, str]] = set()
    for texto in bruto:
        try:
            dados = json.loads(texto)
        except (json.JSONDecodeError, TypeError):
            continue
        for blocos in dados.values():
            if not isinstance(blocos, list):
                continue
            for bloco in blocos:
                arquivo = bloco.get("file")
                if not isinstance(arquivo, str) or not arquivo.strip():
                    continue
                for art in bloco.get("artigos") or []:
                    numero = art.get("artigo") if isinstance(art, dict) else None
                    if numero:
                        esperados.add((arquivo.strip(), _normalizar_artigo(numero)))
    return esperados


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    chunks = pd.read_parquet(INDICE_RAG)
    texto_por_chunk = chunks.set_index("chunk_id")[["source_doc", "text"]].to_dict("index")
    itens = {i.item_id: i for i in carregar_br_taxqa_r(QUESTOES)}

    agentes = sorted(p.name for p in (run_dir / "agent_responses").iterdir())
    resp = pd.read_parquet(run_dir / "agent_responses" / agentes[0] / "responses.parquet")

    linhas = []
    for _, row in resp.iterrows():
        item = itens[row.item_id]
        esperados = dispositivos_esperados(item.dispositivos_legais + item.ementas_carf)
        if not item.has_doc_ref or not esperados:
            continue

        recuperados = [texto_por_chunk[c] for c in row.retrieved_chunk_ids]
        arquivos_recuperados = {c["source_doc"] for c in recuperados}

        hits_arquivo = 0
        hits_passagem = 0
        for arquivo, artigo in esperados:
            achou_arquivo = arquivo in arquivos_recuperados
            hits_arquivo += achou_arquivo
            if achou_arquivo:
                rx = _regex_artigo(artigo)
                hits_passagem += any(
                    c["source_doc"] == arquivo and rx.search(c["text"]) for c in recuperados
                )

        linhas.append(
            {
                "item_id": row.item_id,
                "n_esperados": len(esperados),
                "recall_arquivo": hits_arquivo / len(esperados),
                "recall_passagem": hits_passagem / len(esperados),
            }
        )

    d = pd.DataFrame(linhas)
    print(f"run={args.run}   itens com dispositivos anotados: {len(d)}\n")
    print("=" * 74)
    print("RECALL DE ARQUIVO (instrumento atual)  vs  RECALL DE PASSAGEM (honesto)")
    print("=" * 74)
    instrumentos = [
        ("recall de ARQUIVO ", "recall_arquivo"),
        ("recall de PASSAGEM", "recall_passagem"),
    ]
    for nome, col in instrumentos:
        v = d[col]
        print(f"  {nome}:  Recall@K = {v.mean():.3f}   Hit@K = {(v > 0).mean():.3f}   "
              f"perfeito(=1) = {(v == 1).mean():.3f}")
    infl = d.recall_arquivo.mean() / max(d.recall_passagem.mean(), 1e-9)
    print(f"\n  INFLAÇÃO: o instrumento atual reporta {infl:.2f}x o recall real.")
    print(f"  Itens contados como acerto pelo arquivo mas SEM a passagem certa: "
          f"{((d.recall_arquivo > 0) & (d.recall_passagem == 0)).sum()} "
          f"({((d.recall_arquivo > 0) & (d.recall_passagem == 0)).mean():.1%})")

    print("\n" + "=" * 74)
    print("O RECALL HONESTO PREDIZ MELHOR A ADEQUAÇÃO?")
    print("=" * 74)
    cons = pd.read_parquet(run_dir / "consolidated_test_full_cmin=0.8.parquet")
    m = d.merge(cons, on="item_id")
    from sklearn.metrics import roc_auc_score

    for ag in [c.removeprefix("z_agent_") for c in cons.columns if c.startswith("z_agent_")]:
        z = m[f"z_agent_{ag}"]
        print(f"  {ag:14s} AUROC(arquivo)={roc_auc_score(z, m.recall_arquivo):.3f}   "
              f"AUROC(passagem)={roc_auc_score(z, m.recall_passagem):.3f}")
    print(f"  {'z_winner':14s} AUROC(arquivo)={roc_auc_score(m.z_winner, m.recall_arquivo):.3f}   "
          f"AUROC(passagem)={roc_auc_score(m.z_winner, m.recall_passagem):.3f}")

    print("\n  α do vencedor por faixa de recall:")
    for col in ["recall_arquivo", "recall_passagem"]:
        g = m.groupby(np.where(m[col] == 0, "recall=0", np.where(m[col] < 1, "0<r<1", "recall=1")))
        t = g.agg(n=("item_id", "size"), alpha=("z_winner", "mean")).round(3)
        print(f"\n    por {col}:")
        print("      " + t.to_string().replace("\n", "\n      "))

    saida = run_dir / "metrics" / "recall_passagem.parquet"
    saida.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(saida)
    print(f"\nsalvo em {saida}")


if __name__ == "__main__":
    main()
