#!/usr/bin/env python3
"""Mede a seleção entre agentes nos itens em que seus rótulos divergem."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ablacao", RAIZ / "scripts" / "ablacao_arquitetura.py"
)
assert _SPEC and _SPEC.loader
ablacao = importlib.util.module_from_spec(_SPEC)
sys.modules["ablacao"] = ablacao
_SPEC.loader.exec_module(ablacao)


def _rotulo(nome: str) -> str:
    return nome.split()[0].rstrip(":")


def _markdown(df: pd.DataFrame) -> str:
    colunas = list(df.columns)
    linhas = [
        "| " + " | ".join(colunas) + " |",
        "|" + "|".join("---" for _ in colunas) + "|",
    ]
    linhas.extend(
        "| " + " | ".join(str(valor) for valor in linha) + " |"
        for linha in df.itertuples(index=False)
    )
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="phase2_local_full_715")
    parser.add_argument("--c-min", type=float, default=0.7)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dados, agentes = ablacao.carregar(Path("runs") / args.run, args.c_min)
    variantes = {_rotulo(v.nome): v for v in ablacao.montar_variantes(dados, agentes)}
    a5, b0 = variantes["A5"], variantes["B0"]

    matriz_z = dados[[f"z_agent_{agente}" for agente in agentes]].to_numpy()
    desacordo = matriz_z.min(axis=1) != matriz_z.max(axis=1)
    indices = np.flatnonzero(desacordo)
    if len(indices) == 0:
        raise RuntimeError("Não há itens com desacordo de adequação entre agentes.")

    z_a5 = a5.z_vencedor[indices].astype(float)
    z_b0 = b0.z_vencedor[indices].astype(float)
    delta = float((z_a5 - z_b0).mean())

    rng = np.random.default_rng(args.seed)
    estratos = dados["has_doc_ref"].to_numpy()[indices]
    por_estrato = [np.flatnonzero(estratos == valor) for valor in np.unique(estratos)]
    amostras = np.empty(args.bootstrap, dtype=float)
    diferencas = z_a5 - z_b0
    for i in range(args.bootstrap):
        reamostra = np.concatenate(
            [grupo[rng.integers(0, len(grupo), len(grupo))] for grupo in por_estrato]
        )
        amostras[i] = diferencas[reamostra].mean()
    ic_lo, ic_hi = (float(v) for v in np.percentile(amostras, [2.5, 97.5]))

    resumo = pd.DataFrame(
        [
            {
                "run": args.run,
                "c_min": args.c_min,
                "n_total": len(dados),
                "n_desacordo": len(indices),
                "a5_adequadas": int(z_a5.sum()),
                "a5_taxa": round(float(z_a5.mean()), 6),
                "b0_adequadas": int(z_b0.sum()),
                "b0_taxa": round(float(z_b0.mean()), 6),
                "delta_a5_b0": round(delta, 6),
                "delta_ic95_lo": round(ic_lo, 6),
                "delta_ic95_hi": round(ic_hi, 6),
                "bootstrap": args.bootstrap,
                "seed": args.seed,
            }
        ]
    )

    destino = RAIZ / "tabelas" / f"selecao_condicionada_{args.run}_cmin{args.c_min}"
    resumo.to_parquet(f"{destino}.parquet", index=False)
    Path(f"{destino}.md").write_text(
        "<!-- gerado por scripts/analisar_selecao_condicionada.py; não editar à mão -->\n"
        "# Seleção condicionada ao desacordo de adequação entre agentes\n\n"
        f"{_markdown(resumo)}\n\n"
        "A condição inclui itens com pelo menos um agente adequado e um inadequado. "
        "O intervalo é percentil, pareado por item e estratificado por `has_doc_ref`.\n",
        encoding="utf-8",
    )
    print(resumo.to_string(index=False))
    print(f"\nGerados: {destino}.parquet e {destino}.md")


if __name__ == "__main__":
    main()
