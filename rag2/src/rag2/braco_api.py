"""Braço de agentes de API sobre R3 (N=200; decisions 2026-09-14) — exploratório, fora do pré-registro: contrastes pareados, corte por suficiência, decomposição do Δ por estrato e custo com a geração incluída. CPU, zero chamadas; lê só artefatos versionados em runs/confirmacao/."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from rag2.bootstrap import ic_percentil
from rag2.confirmacao import DIR_CONFIRMACAO, NOME_NOVO
from rag2.constantes import AGENTES, B_BOOTSTRAP

NOME_API = "novo_api"
"""runs/confirmacao/novo_api/ — mesmo contexto e suficiência de `novo` (copiados byte a byte), agentes de API; run_api.py ao lado."""
AGENTES_API = ("claude-haiku-4.5", "gemini-3.1-flash-lite", "gpt-5.4-nano")
"""Ordem dos agentes de API (= phase1_api_pilot_200_1 na origem)."""
ESTRATOS = {2: "suficiente", 1: "parcial", 0: "insuficiente"}
"""Veredito do juiz de suficiência (coluna `suficiencia` do por_item); -1 = contexto vazio/parse falho, ausente nos 200."""
VEREDITOS_AUDITOR = {2: "justificada", 1: "evasiva", 0: "nao_recusa"}
ARQUIVO_CONTRASTES = "contrastes.json"


def carregar_bracos(dir_confirmacao: Path = DIR_CONFIRMACAO) -> pd.DataFrame:
    """Uma linha por item (ordem de pareado.parquet): has_doc_ref, suficiencia (de novo_api ≡ novo), z_media_{atual,local,api} e z_<agente>_{local,api}. Exige contexto e suficiência idênticos entre local e API."""
    par = pd.read_parquet(dir_confirmacao / "pareado.parquet", columns=["item_id", "has_doc_ref", "z_media_atual"])
    loc = pd.read_parquet(dir_confirmacao / NOME_NOVO / "por_item.parquet").set_index("item_id")
    api = pd.read_parquet(dir_confirmacao / NOME_API / "por_item.parquet").set_index("item_id")
    ids = par["item_id"].tolist()
    if sorted(ids) != sorted(loc.index) or sorted(ids) != sorted(api.index):
        raise ValueError("pareado, novo e novo_api não têm os mesmos itens")
    loc, api = loc.loc[ids], api.loc[ids]
    if not (loc["suficiencia"].to_numpy() == api["suficiencia"].to_numpy()).all():
        raise ValueError("suficiência difere entre novo e novo_api — o contexto deveria ser o mesmo")
    df = par.assign(
        suficiencia=loc["suficiencia"].to_numpy(),
        z_media_local=loc["z_media"].to_numpy(dtype=float),
        z_media_api=api["z_media"].to_numpy(dtype=float),
        **{f"z_{aid}_local": loc[f"z_{aid}"].to_numpy(dtype=float) for aid in AGENTES},
        **{f"z_{aid}_api": api[f"z_{aid}"].to_numpy(dtype=float) for aid in AGENTES_API},
        recusa_local=loc["recusa_media"].to_numpy(dtype=float) * len(AGENTES),
        recusa_api=api["recusa_media"].to_numpy(dtype=float) * len(AGENTES_API),
    )
    return df


def _contraste(df: pd.DataFrame, a: str, b: str, *, b_boot: int, estrato: str | None = "has_doc_ref") -> dict:
    """Δ = média por item de (a − b) com IC percentil pareado (reamostra itens; estratos has_doc_ref, semente 42)."""
    d = df.assign(delta=df[a] - df[b])
    ic = ic_percentil(d, lambda x: float(x["delta"].mean()), b=b_boot, coluna_estrato=estrato)
    return {"n": len(d), "a": float(d[a].mean()), "b": float(d[b].mean()), "delta": float(d["delta"].mean()), "ic": [float(ic.ci_lo), float(ic.ci_hi)]}


def custo_api(dir_api: Path) -> dict:
    """Geração (cost_usd dos responses.parquet, por agente) + avaliação (custo.json: juiz, auditor, suficiência) = total; a produção local não tem cobrança de API na geração."""
    geracao = {aid: round(float(pd.read_parquet(dir_api / "agent_responses" / aid / "responses.parquet", columns=["cost_usd"])["cost_usd"].sum()), 4) for aid in AGENTES_API}
    e = json.loads((dir_api / "custo.json").read_text(encoding="utf-8"))["etapas"]
    avaliacao = {k: float(e[k].get("custo_usd", 0.0)) for k in ("juiz", "auditor", "suficiencia")}
    return {
        "geracao": geracao,
        "geracao_total": round(sum(geracao.values()), 4),
        "avaliacao": avaliacao,
        "avaliacao_total": round(sum(avaliacao.values()), 4),
        "total": round(sum(geracao.values()) + sum(avaliacao.values()), 4),
    }


def contrastes(dir_confirmacao: Path = DIR_CONFIRMACAO, *, b: int = B_BOOTSTRAP) -> dict:
    """Tudo o que o apêndice cita do braço de API, apurado dos artefatos: contrastes pareados (api−local, local−atual, api−atual), adequação por agente, corte por suficiência (por braço e por agente), Δ api−local dentro do estrato suficiente, decomposição do Δ global por estrato, recusas e custo."""
    df = carregar_bracos(dir_confirmacao)
    n = len(df)
    por_estrato = {}
    for cod, nome in ESTRATOS.items():
        s = df[df["suficiencia"] == cod]
        por_estrato[nome] = {
            "n": len(s),
            "local": float(s["z_media_local"].mean()),
            "api": float(s["z_media_api"].mean()),
            "por_agente_local": {aid: float(s[f"z_{aid}_local"].mean()) for aid in AGENTES},
            "por_agente_api": {aid: float(s[f"z_{aid}_api"].mean()) for aid in AGENTES_API},
            # contribuição do estrato ao Δ global api−local: (n_s/N)·(api_s − local_s); as três somam o Δ global
            "contribuicao_delta": float(len(s) / n * (s["z_media_api"].mean() - s["z_media_local"].mean())),
        }
    suf = df[df["suficiencia"] == 2]
    aud = pd.read_parquet(dir_confirmacao / NOME_API / "auditoria_recusas.parquet", columns=["veredicto_recusa"])
    return {
        "n_itens": n,
        "b_bootstrap": b,
        "agentes_api": list(AGENTES_API),
        "adequacao_media": {"atual": float(df["z_media_atual"].mean()), "local": float(df["z_media_local"].mean()), "api": float(df["z_media_api"].mean())},
        "por_agente_api": {aid: {"adequacao": float(df[f"z_{aid}_api"].mean()), "n_adequadas": int(df[f"z_{aid}_api"].sum())} for aid in AGENTES_API},
        "contrastes": {
            "api_menos_local": _contraste(df, "z_media_api", "z_media_local", b_boot=b),
            "local_menos_atual": _contraste(df, "z_media_local", "z_media_atual", b_boot=b),
            "api_menos_atual": _contraste(df, "z_media_api", "z_media_atual", b_boot=b),
            # dentro do estrato suficiente: os estratos has_doc_ref ficam pequenos; reamostragem simples (sem estratificação), declarada
            "api_menos_local_suficiente": _contraste(suf, "z_media_api", "z_media_local", b_boot=b, estrato=None),
        },
        "por_suficiencia": por_estrato,
        "recusas": {
            "local": int(df["recusa_local"].sum()),
            "api": int(df["recusa_api"].sum()),
            "auditor_api": {nome: int((aud["veredicto_recusa"] == cod).sum()) for cod, nome in VEREDITOS_AUDITOR.items()},
        },
        "custo": custo_api(dir_confirmacao / NOME_API),
    }


def _f(x: float) -> str:
    return f"{x:.4f}"


def _ic(c: Mapping) -> str:
    return f"{c['delta']:+.4f} [{c['ic'][0]:+.4f}; {c['ic'][1]:+.4f}]"


def contrastes_md(r: Mapping) -> str:
    """contrastes.md: leitura humana do contrastes.json (mesmas cifras)."""
    c, s, k = r["contrastes"], r["por_suficiencia"], r["custo"]
    linhas = [
        f"# Braço de agentes de API sobre R3 — exploratório, fora do pré-registro (N={r['n_itens']})\n",
        (
            "Contexto e suficiência de R3 reusados byte a byte do braço local; agentes de API com `max_tokens` 4096 (piloto da origem) contra 1024 dos locais — "
            "comparação de configurações de geração sob contexto fixo, não só de família de agentes. IC95 percentil pareado por item, "
            f"B = {r['b_bootstrap']}, semente 42, estratificado por has_doc_ref (sem estratificação dentro do estrato suficiente).\n"
        ),
        "| contraste | n | a | b | Δ a − b [IC95] |", "|---|---:|---:|---:|---:|",
        f"| API@R3 − locais@R3 | {c['api_menos_local']['n']} | {_f(c['api_menos_local']['a'])} | {_f(c['api_menos_local']['b'])} | {_ic(c['api_menos_local'])} |",
        f"| locais@R3 − produção | {c['local_menos_atual']['n']} | {_f(c['local_menos_atual']['a'])} | {_f(c['local_menos_atual']['b'])} | {_ic(c['local_menos_atual'])} |",
        f"| API@R3 − produção | {c['api_menos_atual']['n']} | {_f(c['api_menos_atual']['a'])} | {_f(c['api_menos_atual']['b'])} | {_ic(c['api_menos_atual'])} |",
        f"| API@R3 − locais@R3, só contexto suficiente | {c['api_menos_local_suficiente']['n']} | {_f(c['api_menos_local_suficiente']['a'])} | {_f(c['api_menos_local_suficiente']['b'])} | {_ic(c['api_menos_local_suficiente'])} |",
        "\n| suficiência | n | API | local | contribuição ao Δ global |", "|---|---:|---:|---:|---:|",
        *[f"| {nome} | {e['n']} | {_f(e['api'])} | {_f(e['local'])} | {e['contribuicao_delta']:+.4f} |" for nome, e in s.items()],
        "\n| agente | adequação (200) | adequação no estrato suficiente |", "|---|---:|---:|",
        *[f"| {aid} | {_f(r['por_agente_api'][aid]['adequacao'])} | {_f(s['suficiente']['por_agente_api'][aid])} |" for aid in r["agentes_api"]],
        *[f"| {aid} (local) | — | {_f(s['suficiente']['por_agente_local'][aid])} |" for aid in AGENTES],
        f"\nRecusas: API {r['recusas']['api']} (auditor: {r['recusas']['auditor_api']}); locais {r['recusas']['local']}.\n",
        "| custo US$ | valor |", "|---|---:|",
        *[f"| geração {aid} | {v:.4f} |" for aid, v in k["geracao"].items()],
        f"| geração total | {k['geracao_total']:.4f} |",
        *[f"| avaliação {e} | {v:.4f} |" for e, v in k["avaliacao"].items()],
        f"| avaliação total | {k['avaliacao_total']:.4f} |",
        f"| total | {k['total']:.4f} |",
    ]
    return "\n".join(linhas) + "\n"


def gravar_contrastes(dir_confirmacao: Path = DIR_CONFIRMACAO, *, b: int = B_BOOTSTRAP) -> Path:
    """contrastes.json (registrar uma vez; regenerar ≡ versionado) e contrastes.md em runs/confirmacao/novo_api/."""
    r = contrastes(dir_confirmacao, b=b)
    saida = Path(dir_confirmacao) / NOME_API
    (saida / ARQUIVO_CONTRASTES).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "contrastes.md").write_text(contrastes_md(r), encoding="utf-8")
    return saida
