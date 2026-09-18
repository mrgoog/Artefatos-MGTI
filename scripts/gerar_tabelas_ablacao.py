"""Persiste em disco as tabelas de comparação de arquitetura que o artigo cita.

Materializa a ablação, a matriz pareada de variantes e a interação 2×2 em `tabelas/`.
A tabela de ablação é propriedade das **saídas dos agentes**, não da cabeça de decisão: usa
`confidence_*` e `z_agent_*`, idênticas entre `phase2_local_full_715` e `..._meanraw` e entre
valores de C_min. Rodar sobre um run basta.

Uso:
    uv run python scripts/gerar_tabelas_ablacao.py
    uv run python scripts/gerar_tabelas_ablacao.py --run phase2_local_full_715 --c-min 0.7
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = Path("tabelas")

_SPEC = importlib.util.spec_from_file_location(
    "ablacao", RAIZ / "scripts" / "ablacao_arquitetura.py"
)
assert _SPEC and _SPEC.loader
ablacao = importlib.util.module_from_spec(_SPEC)
sys.modules["ablacao"] = ablacao
_SPEC.loader.exec_module(ablacao)


def _rotulo(nome: str) -> str:
    """'A5 argmax(c_bruta) + score = MÉDIA(c_bruta)' -> 'A5'."""
    return nome.split()[0].rstrip(":")


def _markdown(df: pd.DataFrame) -> str:
    """Tabela GFM sem depender de `tabulate` — não vale uma dependência por isto."""
    cols = list(df.columns)
    linhas = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    linhas += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(linhas)


def _escrever(df: pd.DataFrame, base: Path, titulo: str) -> None:
    # Nada de with_suffix aqui: o nome contém "cmin0.7" e o ".7" seria tomado por extensão.
    parquet, md = Path(f"{base}.parquet"), Path(f"{base}.md")
    df.to_parquet(parquet, index=False)
    md.write_text(
        f"<!-- gerado por scripts/gerar_tabelas_ablacao.py — não editar à mão -->\n"
        f"# {titulo}\n\n{_markdown(df)}\n",
        encoding="utf-8",
    )
    print(f"  {parquet}  ({len(df)} linhas)")
    print(f"  {md}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="phase2_local_full_715")
    p.add_argument("--c-min", type=float, default=0.7)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d, agentes = ablacao.carregar(Path("runs") / args.run, args.c_min)
    vars_ = ablacao.montar_variantes(d, agentes)
    base = vars_[0]
    print(f"run={args.run}  n={len(d)}  agentes={agentes}\n")
    DESTINO.mkdir(parents=True, exist_ok=True)

    # (a) tabela de ablação: cada variante vs A0
    linhas = []
    for v in vars_:
        z, s = v.z_vencedor, v.score
        lo, hi = (
            (float("nan"), float("nan"))
            if v is base
            else ablacao.bootstrap_delta(v, base, args.bootstrap, args.seed)
        )
        linhas.append(
            {
                "variante": _rotulo(v.nome),
                "descricao": v.nome.split(" ", 1)[1],
                "acuracia_cob1": round(float(z.mean()), 3),
                "auroc": round(float(roc_auc_score(z, s)), 3),
                "aurc": round(float(ablacao.aurc(s, z)), 3),
                "risco_cob03": round(float(ablacao.risco_em(s, z, 0.3)), 3),
                "risco_cob05": round(float(ablacao.risco_em(s, z, 0.5)), 3),
                "delta_aurc_vs_a0": round(
                    float(ablacao.aurc(s, z) - ablacao.aurc(base.score, base.z_vencedor)), 3
                ),
                "ic_lo": round(lo, 3),
                "ic_hi": round(hi, 3),
                "significativo": bool(hi < 0 or lo > 0) if v is not base else False,
            }
        )
    sufixo = f"{args.run}_cmin{args.c_min}"
    _escrever(
        pd.DataFrame(linhas),
        DESTINO / f"ablacao_{sufixo}",
        f"Ablação de arquitetura — {args.run}, C_min={args.c_min}, n={len(d)}",
    )

    # (b) matriz pareada: toda comparação entre variantes, não só contra A0.
    # A5 vs B0 (ensemble vs agente único) só existe aqui.
    vmap = {_rotulo(v.nome): v for v in vars_}
    pares = []
    for a in vmap:
        for b in vmap:
            if a == b:
                continue
            va, vb = vmap[a], vmap[b]
            pt = ablacao.aurc(va.score, va.z_vencedor) - ablacao.aurc(vb.score, vb.z_vencedor)
            lo, hi = ablacao.bootstrap_delta(va, vb, args.bootstrap, args.seed)
            pares.append(
                {
                    "a": a,
                    "b": b,
                    "delta_aurc": round(float(pt), 3),
                    "ic_lo": round(lo, 3),
                    "ic_hi": round(hi, 3),
                    "vencedor": a if hi < 0 else (b if lo > 0 else "indistinguiveis"),
                }
            )
    _escrever(
        pd.DataFrame(pares),
        DESTINO / f"matriz_pareada_{sufixo}",
        f"ΔAURC pareado (a − b; negativo = a melhor) — {args.run}, C_min={args.c_min}",
    )

    # (c) interação do 2x2.
    if all(k in vmap for k in ("A1", "A4", "A5", "A7")):
        def _aurc(v: str, idx: object = None) -> float:
            r = vmap[v]
            if idx is None:
                return ablacao.aurc(r.score, r.z_vencedor)
            return ablacao.aurc(r.score[idx], r.z_vencedor[idx])

        # interação = (efeito do operador sob bruta) − (efeito do operador sob calibrada)
        pt = (_aurc("A5") - _aurc("A7")) - (_aurc("A4") - _aurc("A1"))
        rng = np.random.default_rng(args.seed)
        estratos = d["has_doc_ref"].to_numpy()
        indices_estratos = [
            np.flatnonzero(estratos == valor) for valor in np.unique(estratos)
        ]

        def _reamostrar() -> np.ndarray:
            return np.concatenate(
                [idx[rng.integers(0, len(idx), len(idx))] for idx in indices_estratos]
            )

        amostras = [
            (_aurc("A5", i) - _aurc("A7", i)) - (_aurc("A4", i) - _aurc("A1", i))
            for i in (_reamostrar() for _ in range(args.bootstrap))
        ]
        lo, hi = (float(x) for x in np.percentile(amostras, [2.5, 97.5]))
        inter = pd.DataFrame([{
            "termo": "interacao_operador_x_calibracao",
            "definicao": "(A5-A7) - (A4-A1)",
            "delta_aurc": round(float(pt), 3),
            "ic_lo": round(lo, 3),
            "ic_hi": round(hi, 3),
            "ic_inclui_zero": bool(lo <= 0 <= hi),
        }])
        _escrever(
            inter,
            DESTINO / f"interacao_2x2_{sufixo}",
            f"Interação operador × calibração no fatorial 2×2 — {args.run}, C_min={args.c_min}",
        )

    print("\n  Regra das duas fontes: estas tabelas são a ÚNICA fonte de comparação entre")
    print("  arquiteturas (as únicas com IC pareado). Ponto de operação e figuras vêm dos")
    print("  runs materializados. Não misturar as duas fontes na mesma frase.")


if __name__ == "__main__":
    main()
