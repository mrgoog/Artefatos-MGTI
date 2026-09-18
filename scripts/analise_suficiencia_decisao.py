"""Análise cruzada: suficiência de contexto × ação do agente × correção (custo zero).

Consome `metrics/suficiencia_contexto.parquet` (produzido por
`juiz_suficiencia_contexto.py`) e cruza com fatos já em disco para responder:
*a abstenção foi por motivo real?*

Três saídas:
  (a) Tabela de qualidade de decisão — ação (respondeu/absteve) × suficiência.
  (b) Curva observacional de adequação vs suficiência, com IC bootstrap.
  (c) Consistência vs a auditoria de recusas já feita.

Nada é modificado — leitura pura sobre artefatos persistidos.

Uso:
    uv run python scripts/analise_suficiencia_decisao.py --run phase2_local_full_715
    uv run python scripts/analise_suficiencia_decisao.py --run phase2_local_full_715 --cmin 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from auditar_recusas import PADRAO_RECUSA  # noqa: E402

from ensemble_llm.avaliacao.bootstrap_estratificado import (  # noqa: E402
    ic_bootstrap_estratificado,
)
from ensemble_llm.persistencia.manifesto import ler_manifesto  # noqa: E402

ROTULO = {2: "suficiente", 1: "parcial", 0: "insuficiente", -1: "falha"}


def _acao_por_item_agente(run_dir: Path, agentes: list[str]) -> pd.DataFrame:
    """(item_id, agent_id, refused) — recusa detectada pela MESMA regex do auditar."""
    partes = []
    for a in agentes:
        r = pd.read_parquet(run_dir / "agent_responses" / a / "responses.parquet")
        r = r[["item_id", "response_text"]].copy()
        r["agent_id"] = a
        r["refused"] = r["response_text"].fillna("").str.contains(PADRAO_RECUSA)
        partes.append(r[["item_id", "agent_id", "refused"]])
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cmin", type=float, default=0.7)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    manifesto = ler_manifesto(run_dir / "manifest.yaml")
    agentes = list(manifesto.agents)

    suf = pd.read_parquet(run_dir / "metrics" / "suficiencia_contexto.parquet")
    suf = suf[suf.veredicto_suficiencia >= 0]  # descarta falhas de parse (-1)
    cons = pd.read_parquet(run_dir / f"consolidated_test_full_cmin={args.cmin:.1f}.parquet")
    acao = _acao_por_item_agente(run_dir, agentes)

    # tabela longa: (item, agente) com suficiência (per-item), ação e z_agent
    z_long = cons.melt(
        id_vars=["item_id"],
        value_vars=[f"z_agent_{a}" for a in agentes],
        var_name="agent_id",
        value_name="z_agent",
    )
    z_long["agent_id"] = z_long.agent_id.str.removeprefix("z_agent_")
    long = (
        acao.merge(z_long, on=["item_id", "agent_id"])
        .merge(suf[["item_id", "veredicto_suficiencia"]], on="item_id")
    )
    long["suf"] = long.veredicto_suficiencia.map(ROTULO)
    long["acao"] = np.where(long.refused, "absteve", "respondeu")

    print("=" * 72)
    print(f"(a) QUALIDADE DE DECISÃO — ação × suficiência  (n={len(long)} pares item×agente)")
    print("=" * 72)
    ordem = ["insuficiente", "parcial", "suficiente"]
    cont = pd.crosstab(long.acao, long.suf).reindex(columns=ordem)
    print("\ncontagens:")
    print(cont.to_string())
    print("\ntaxa por linha (como cada ação se distribui pela suficiência):")
    print((cont.div(cont.sum(1), axis=0) * 100).round(1).to_string())
    print("\nadequação (z_agent médio) dentro de cada célula:")
    piv = long.pivot_table(index="acao", columns="suf", values="z_agent", aggfunc="mean")
    print((piv.reindex(columns=ordem) * 100).round(1).to_string())
    print("\ninterpretação das células:")
    print("  absteve × insuficiente = ABSTENÇÃO CORRETA  | absteve × suficiente = recusa EVASIVA")
    print("  respondeu × insuficiente = resp. ARRISCADA | respondeu × suficiente = resp. PRÓPRIA")

    # forma binária colapsada (suficiente = veredicto>=1) para o 2×2
    long["suf_bin"] = np.where(long.veredicto_suficiencia >= 1, "suf(>=parcial)", "insuf")
    print("\n2×2 colapsado (contagens):")
    print(pd.crosstab(long.acao, long.suf_bin).to_string())

    print("\n" + "=" * 72)
    print(f"(b) ADEQUAÇÃO DO SISTEMA vs SUFICIÊNCIA  (por item, cmin={args.cmin})")
    print("=" * 72)
    m = cons.merge(suf[["item_id", "veredicto_suficiencia"]], on="item_id")
    print(f"\n{'suficiência':14s} {'n':>4s} {'z_winner (IC95%)':>22s} {'cobertura':>10s}")
    for v in (0, 1, 2):
        sub = m[m.veredicto_suficiencia == v]
        if sub.empty:
            continue
        ic = ic_bootstrap_estratificado(
            sub, lambda d: float(d.z_winner.mean()), n_reamostras=2000
        )
        cob = float((~sub.abstained).mean())
        print(f"  {ROTULO[v]:12s} {len(sub):4d}   {ic.point:.3f} [{ic.ci_lo:.3f},{ic.ci_hi:.3f}]"
              f"   {cob:8.1%}")
    print("  (z_winner = adequação do agente vencedor; independe de abstenção. "
          "cobertura = fração respondida.)")

    print("\n" + "=" * 72)
    print("(c) CONSISTÊNCIA vs auditoria_recusas.parquet (deve reproduzir ~77%)")
    print("=" * 72)
    aud_path = run_dir / "metrics" / "auditoria_recusas.parquet"
    if not aud_path.exists():
        print(f"  (arquivo {aud_path} não existe — pule esta checagem)")
        return
    aud = pd.read_parquet(aud_path).merge(
        suf[["item_id", "veredicto_suficiencia"]], on="item_id", how="left"
    )
    # dos (item,agente) que recusaram: fração com contexto julgado insuficiente
    frac_insuf = float((aud.veredicto_suficiencia == 0).mean())
    frac_nao_suf = float(aud.veredicto_suficiencia.isin([0, 1]).mean())
    print(f"\n  das {len(aud)} recusas auditadas, fração com suficiência=INSUFICIENTE: "
          f"{frac_insuf:.1%}")
    print(f"  fração com contexto NÃO-suficiente (insuf+parcial ~ 'justificada' binária): "
          f"{frac_nao_suf:.1%}")
    print("  (auditar_recusas achou 76.7% JUSTIFICADAS — o binário absorve o 'parcial' do ordinal)")
    print("\n  cruzamento veredicto_recusa (auditar) × suficiência (este juiz):")
    rot_rec = {2: "justificada", 1: "evasiva", 0: "nao-recusa", -1: "falha"}
    tab = pd.crosstab(
        aud.veredicto_recusa.map(rot_rec), aud.veredicto_suficiencia.map(ROTULO)
    )
    print(tab.to_string())
    # concordância: justificada(2) <-> insuficiente(0)
    val = aud[(aud.veredicto_recusa >= 0) & (aud.veredicto_suficiencia >= 0)]
    conc = float(
        ((val.veredicto_recusa == 2) == (val.veredicto_suficiencia == 0)).mean()
    )
    print(f"\n  concordância (recusa justificada ⟺ contexto insuficiente): {conc:.1%}")


if __name__ == "__main__":
    main()
