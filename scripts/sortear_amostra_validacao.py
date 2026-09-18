"""Sorteia a amostra para validação humana do juiz LLM.

Gera `data/validacao_humana/amostra.parquet` com TODOS os pares item×agente rotulados,
classificados em três estratos e com uma ordem de sorteio fixa por estrato:

  adequada     — resposta substantiva com z = 1 pelo juiz binário
  inadequada   — resposta substantiva com z = 0 pelo juiz binário
  recusa       — recusa textual auditada em `auditoria_recusas.parquet` (146 pares)

A ordem é uma permutação com semente fixa; a UI (`anotar_validacao.py`) exibe os N
primeiros de cada estrato, de modo que aumentar N depois preserva os itens já anotados.
Nos estratos substantivos, é sorteado no máximo um par por item, para evitar que a mesma
pergunta apareça duas vezes.

O parquet contém o contexto servido reconstruído e o rótulo/raciocínio do juiz, que a UI
mantém ocultos até a anotação ser salva.

Uso:
    uv run python scripts/sortear_amostra_validacao.py
    uv run python scripts/sortear_amostra_validacao.py --run phase2_local_full_715 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r

INDICE_RAG = Path("runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/denso/chunks.parquet")
QUESTOES = Path("data/brtaxq/questions_QA_2024_v1.1.json")
SAIDA = Path("data/validacao_humana/amostra.parquet")

PADRAO_RECUSA = re.compile(
    r"não é possível responder|nao e possivel responder|não posso responder"
    r"|contexto (?:fornecido )?(?:é|e) insuficiente|não há informa|nao ha informa"
    r"|material fornecido não|não contém informa|não permite responder",
    re.IGNORECASE,
)


def _carregar_respostas(run: Path) -> pd.DataFrame:
    partes = []
    for pasta in sorted((run / "agent_responses").iterdir()):
        df = pd.read_parquet(pasta / "responses.parquet")
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def _formatar_contexto(chunk_ids: list[str], chunks: pd.DataFrame) -> str:
    if len(chunk_ids) == 0:
        return "(Nenhum documento recuperado para esta pergunta.)"
    partes = []
    for k, cid in enumerate(chunk_ids, start=1):
        if cid in chunks.index:
            linha = chunks.loc[cid]
            partes.append(f"[DOCUMENTO {k} — {linha['source_doc']}]\n{str(linha['text']).strip()}")
        else:
            partes.append(f"[DOCUMENTO {k} — {cid}]\n(chunk não encontrado no índice)")
    return "\n\n".join(partes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="phase2_local_full_715", help="run com judge_labels e auditoria_recusas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()

    run = Path("runs") / args.run
    respostas = _carregar_respostas(run)
    juiz = pd.read_parquet(next((run / "judge_labels").glob("*/cache.parquet")))
    juiz = juiz.rename(columns={"response_source": "agent_id", "raciocinio": "raciocinio_juiz"})
    auditoria = pd.read_parquet(run / "metrics" / "auditoria_recusas.parquet")
    chunks = pd.read_parquet(INDICE_RAG).set_index("chunk_id")

    itens = {it.item_id: it for it in carregar_br_taxqa_r(QUESTOES)}

    df = respostas.merge(juiz[["item_id", "agent_id", "z", "raciocinio_juiz"]], on=["item_id", "agent_id"], how="inner")
    df = df.merge(auditoria[["item_id", "agent_id", "veredicto_recusa", "raciocinio_auditor"]], on=["item_id", "agent_id"], how="left")

    df["recusa_textual"] = df["response_text"].fillna("").str.contains(PADRAO_RECUSA)
    df["estrato"] = np.where(
        df["veredicto_recusa"].notna(), "recusa", np.where(df["z"] == 1, "adequada", "inadequada")
    )

    df["pergunta"] = df["item_id"].map(lambda i: itens[i].pergunta)
    df["ground_truth"] = df["item_id"].map(lambda i: itens[i].ground_truth)
    df["dispositivos_legais"] = df["item_id"].map(lambda i: json.dumps(itens[i].dispositivos_legais, ensure_ascii=False))
    df["ementas_carf"] = df["item_id"].map(lambda i: json.dumps(itens[i].ementas_carf, ensure_ascii=False))
    df["contexto_servido"] = df["retrieved_chunk_ids"].map(lambda ids: _formatar_contexto(list(ids), chunks))

    rng = np.random.default_rng(args.seed)
    partes = []
    for estrato, grupo in df.groupby("estrato"):
        grupo = grupo.iloc[rng.permutation(len(grupo))].copy()
        if estrato != "recusa":
            # no máximo um par por item nos estratos substantivos
            grupo = grupo.drop_duplicates("item_id", keep="first")
        grupo["ordem_sorteio"] = np.arange(1, len(grupo) + 1)
        partes.append(grupo)
    amostra = pd.concat(partes, ignore_index=True)

    # evita que um item sorteado cedo em "adequada" também apareça cedo em "inadequada":
    # itens repetidos entre estratos substantivos ficam apenas no estrato em que têm menor ordem
    sub = amostra[amostra["estrato"] != "recusa"].sort_values("ordem_sorteio")
    duplicados = sub[sub.duplicated("item_id", keep="first")].index
    amostra = amostra.drop(duplicados)
    for estrato in ("adequada", "inadequada"):
        mask = amostra["estrato"] == estrato
        amostra.loc[mask, "ordem_sorteio"] = np.arange(1, mask.sum() + 1)

    colunas = [
        "estrato", "ordem_sorteio", "item_id", "agent_id", "pergunta", "ground_truth",
        "dispositivos_legais", "ementas_carf", "response_text", "confidence_raw",
        "contexto_servido", "retrieved_chunk_ids", "z", "raciocinio_juiz",
        "veredicto_recusa", "raciocinio_auditor", "recusa_textual", "prompt_hash",
    ]
    amostra = amostra[colunas].sort_values(["estrato", "ordem_sorteio"]).reset_index(drop=True)
    amostra["retrieved_chunk_ids"] = amostra["retrieved_chunk_ids"].map(list)
    amostra["seed"] = args.seed
    amostra["run"] = args.run

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_parquet(args.saida, index=False)
    print(f"gravado {args.saida} ({len(amostra)} pares)")
    print(amostra["estrato"].value_counts().to_string())
    print("\nrecusas por veredicto do auditor (2=justificada, 1=evasiva, 0=não é recusa):")
    print(amostra.loc[amostra.estrato == "recusa", "veredicto_recusa"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
