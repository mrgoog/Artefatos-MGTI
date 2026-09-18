"""Auditoria complementar de não resposta, adequação e suficiência do Artigo 2.

Esta análise usa somente respostas, rótulos e scores já persistidos. Ela não chama
agentes nem juízes. As métricas aqui definidas complementam AURC e risco seletivo; não
os redefinem.

Saídas em ``tabelas/``:

* ``auditoria_decisao_itens_*.parquet``: item × variante;
* ``metricas_decisao_cobertura_*.{parquet,md}``: curvas em coberturas comuns;
* ``contrastes_decisao_*.{parquet,md}``: contrastes pareados com IC percentil;
* ``pontos_operacao_decisao_*.{parquet,md}``: runs A0 e A5 materializados.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "tabelas"

_SPEC = importlib.util.spec_from_file_location(
    "ablacao", RAIZ / "scripts" / "ablacao_arquitetura.py"
)
assert _SPEC and _SPEC.loader
ablacao = importlib.util.module_from_spec(_SPEC)
sys.modules["ablacao"] = ablacao
_SPEC.loader.exec_module(ablacao)

COBERTURAS_PADRAO = (0.3, 0.5, 0.7, 0.9, 1.0)
CONTRASTES_PADRAO = (
    ("A5", "A7"),
    ("A4", "A1"),
    ("A5", "A0"),
    ("A5", "B0"),
)
METRICAS_CONTRASTE = (
    "cobertura_efetiva",
    "risco_substantivo",
    "utilidade_convencional",
    "utilidade_erro_efetiva",
    "utilidade_contexto_estrita",
    "utilidade_contexto_ampla",
    "penalizacao_contextual_estrita",
    "penalizacao_contextual_ampla",
)
METRICAS_PONTO_CONTRASTE = (
    "cobertura_nominal",
    "cobertura_efetiva",
    "risco_seletivo_convencional",
    "risco_substantivo",
    "precisao_abstencao_erro",
    "recall_erros_abstencao",
    "taxa_falsa_abstencao",
    *METRICAS_CONTRASTE[2:],
)


def _rotulo(nome: str) -> str:
    return nome.split()[0].rstrip(":")


def pesos_resposta_para_cobertura(score: np.ndarray, cobertura: float) -> np.ndarray:
    """Probabilidade de resposta por item, com randomização esperada no empate de corte."""
    score = np.asarray(score, dtype=float)
    if not 0.0 <= cobertura <= 1.0:
        raise ValueError(f"cobertura deve estar em [0,1], recebeu {cobertura}")
    n = len(score)
    if n == 0:
        raise ValueError("score vazio")

    alvo = cobertura * n
    pesos = np.zeros(n, dtype=float)
    restante = alvo
    for valor in np.unique(score)[::-1]:
        idx = np.flatnonzero(score == valor)
        if restante <= 0:
            break
        if restante >= len(idx):
            pesos[idx] = 1.0
            restante -= len(idx)
        else:
            pesos[idx] = restante / len(idx)
            restante = 0.0
    return pesos


def _div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def calcular_metricas(dados: pd.DataFrame, prob_resposta: np.ndarray) -> dict[str, float]:
    """Calcula métricas complementares para uma política de resposta probabilística.

    ``prob_resposta`` vale 0/1 para um threshold sem empate no corte. Valores fracionários
    representam a esperança sob desempate aleatório dentro do grupo de score empatado.
    """
    a = np.asarray(prob_resposta, dtype=float)
    z = dados["z_winner"].to_numpy(dtype=float)
    t = dados["textual_refusal"].to_numpy(dtype=bool)
    k = dados["context_sufficiency"].to_numpy(dtype=int)
    verdict = dados["refusal_verdict"].to_numpy(dtype=float)
    if len(a) != len(dados):
        raise ValueError("prob_resposta e dados têm tamanhos diferentes")

    substantiva = (~t).astype(float)
    resposta_efetiva = a * substantiva
    nao_resposta_efetiva = 1.0 - resposta_efetiva
    abstencao = 1.0 - a
    erro = 1.0 - z

    recusa_estrita = t & (verdict == 2)
    recusa_ampla = t & np.isin(verdict, [1, 2])
    contexto_estrito = k == 0
    contexto_amplo = k <= 1

    utilidade_convencional = float(np.mean(resposta_efetiva * z))
    utilidade_erro_efetiva = float(
        np.mean(resposta_efetiva * z + nao_resposta_efetiva * erro)
    )
    utilidade_contexto_estrita = float(
        np.mean(
            abstencao * contexto_estrito
            + a * (recusa_estrita.astype(float) + substantiva * z)
        )
    )
    utilidade_contexto_ampla = float(
        np.mean(
            abstencao * contexto_amplo
            + a * (recusa_ampla.astype(float) + substantiva * z)
        )
    )

    n_abstencoes = float(abstencao.sum())
    n_nao_respostas = float(nao_resposta_efetiva.sum())
    out = {
        "n": float(len(dados)),
        "cobertura_nominal": float(a.mean()),
        "cobertura_efetiva": float(resposta_efetiva.mean()),
        "risco_seletivo_convencional": _div(float((a * erro).sum()), float(a.sum())),
        "risco_substantivo": _div(
            float((resposta_efetiva * erro).sum()), float(resposta_efetiva.sum())
        ),
        "precisao_abstencao_erro": _div(
            float((abstencao * erro).sum()), n_abstencoes
        ),
        "recall_erros_abstencao": _div(
            float((abstencao * erro).sum()), float(erro.sum())
        ),
        "taxa_falsa_abstencao": _div(float((abstencao * z).sum()), float(z.sum())),
        "precisao_abstencao_contexto_estrita": _div(
            float((abstencao * contexto_estrito).sum()), n_abstencoes
        ),
        "precisao_abstencao_contexto_ampla": _div(
            float((abstencao * contexto_amplo).sum()), n_abstencoes
        ),
        "precisao_nao_resposta_contexto_estrita": _div(
            float((nao_resposta_efetiva * contexto_estrito).sum()), n_nao_respostas
        ),
        "precisao_nao_resposta_contexto_ampla": _div(
            float((nao_resposta_efetiva * contexto_amplo).sum()), n_nao_respostas
        ),
        "abstencao_dado_contexto_insuficiente": _div(
            float((abstencao * (k == 0)).sum()), float((k == 0).sum())
        ),
        "abstencao_dado_contexto_parcial": _div(
            float((abstencao * (k == 1)).sum()), float((k == 1).sum())
        ),
        "abstencao_dado_contexto_suficiente": _div(
            float((abstencao * (k == 2)).sum()), float((k == 2).sum())
        ),
        "recusas_textuais_selecionadas": float(t.sum()),
        "recusas_textuais_liberadas": float((a * t).sum()),
        "utilidade_convencional": utilidade_convencional,
        "utilidade_erro_efetiva": utilidade_erro_efetiva,
        "utilidade_contexto_estrita": utilidade_contexto_estrita,
        "utilidade_contexto_ampla": utilidade_contexto_ampla,
        "penalizacao_contextual_estrita": (
            utilidade_contexto_estrita - utilidade_convencional
        ),
        "penalizacao_contextual_ampla": (
            utilidade_contexto_ampla - utilidade_convencional
        ),
    }
    return out


def montar_tabela_itens(
    run_dir: Path,
    c_min: float,
    audit_run_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Monta a tabela item × variante a partir dos artefatos existentes."""
    d, agentes = ablacao.carregar(run_dir, c_min)
    variantes = ablacao.montar_variantes(d, agentes)
    suficiencia = pd.read_parquet(run_dir / "metrics" / "suficiencia_contexto.parquet")
    mapa_suf = suficiencia.set_index("item_id")["veredicto_suficiencia"]
    auditoria = pd.read_parquet(audit_run_dir / "metrics" / "auditoria_recusas.parquet")
    mapa_auditoria = auditoria.set_index(["item_id", "agent_id"])["veredicto_recusa"]
    mat_recusas = d[[f"recusa_{a}" for a in agentes]].to_numpy(dtype=bool)
    linhas = np.arange(len(d))

    blocos: list[pd.DataFrame] = []
    for resultado in variantes:
        escolhidos = resultado.escolhido.astype(int)
        agentes_escolhidos = np.asarray(agentes, dtype=object)[escolhidos]
        recusas = mat_recusas[linhas, escolhidos]
        pares = pd.MultiIndex.from_arrays([d["item_id"], agentes_escolhidos])
        veredictos = mapa_auditoria.reindex(pares).to_numpy(dtype=float)
        faltantes = recusas & np.isnan(veredictos)
        if faltantes.any():
            exemplos = d.loc[faltantes, "item_id"].head(5).tolist()
            raise RuntimeError(
                "Recusas textuais selecionadas sem auditoria: " + ", ".join(exemplos)
            )

        blocos.append(
            pd.DataFrame(
                {
                    "item_id": d["item_id"].to_numpy(),
                    "fold_k": d["fold_k"].to_numpy(),
                    "has_doc_ref": d["has_doc_ref"].to_numpy(dtype=bool),
                    "variant": _rotulo(resultado.nome),
                    "variant_description": resultado.nome,
                    "score": resultado.score.astype(float),
                    "agent_chosen": agentes_escolhidos,
                    "z_winner": resultado.z_vencedor.astype(int),
                    "textual_refusal": recusas,
                    "refusal_verdict": veredictos,
                    "context_sufficiency": d["item_id"].map(mapa_suf).to_numpy(dtype=int),
                }
            )
        )
    itens = pd.concat(blocos, ignore_index=True)
    if not bool(itens["context_sufficiency"].isin([0, 1, 2]).all()):
        raise RuntimeError("Há itens sem rótulo válido de suficiência")
    return itens, agentes


def metricas_por_cobertura(
    itens: pd.DataFrame, coberturas: tuple[float, ...]
) -> pd.DataFrame:
    linhas: list[dict[str, float | str]] = []
    for variante, dados in itens.groupby("variant", sort=False):
        dados = dados.reset_index(drop=True)
        for cobertura in coberturas:
            prob = pesos_resposta_para_cobertura(dados["score"].to_numpy(), cobertura)
            linhas.append(
                {
                    "variant": variante,
                    "target_coverage": cobertura,
                    **calcular_metricas(dados, prob),
                }
            )
    return pd.DataFrame(linhas)


def _reamostra_estratificada(
    estrato: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    partes = []
    for valor in np.unique(estrato):
        idx = np.flatnonzero(estrato == valor)
        partes.append(idx[rng.integers(0, len(idx), len(idx))])
    return np.concatenate(partes)


def contrastes_bootstrap(
    itens: pd.DataFrame,
    coberturas: tuple[float, ...],
    n_reamostras: int,
    seed: int,
) -> pd.DataFrame:
    """Contrastes A−B pareados; negativo significa valor menor em A."""
    por_variante = {
        nome: dados.reset_index(drop=True)
        for nome, dados in itens.groupby("variant", sort=False)
    }
    necessarias = {v for par in CONTRASTES_PADRAO for v in par}
    ausentes = necessarias - por_variante.keys()
    if ausentes:
        raise RuntimeError(f"Variantes ausentes para contrastes: {sorted(ausentes)}")

    referencia = por_variante[next(iter(necessarias))]
    estrato = referencia["has_doc_ref"].to_numpy(dtype=bool)
    rng = np.random.default_rng(seed)
    chaves = [
        (a, b, cobertura, metrica)
        for a, b in CONTRASTES_PADRAO
        for cobertura in coberturas
        for metrica in METRICAS_CONTRASTE
    ]
    amostras = {chave: np.empty(n_reamostras, dtype=float) for chave in chaves}

    for b_idx in range(n_reamostras):
        idx = _reamostra_estratificada(estrato, rng)
        cache: dict[tuple[str, float], dict[str, float]] = {}
        for nome in necessarias:
            dados = por_variante[nome].iloc[idx].reset_index(drop=True)
            for cobertura in coberturas:
                prob = pesos_resposta_para_cobertura(
                    dados["score"].to_numpy(), cobertura
                )
                cache[(nome, cobertura)] = calcular_metricas(dados, prob)
        for chave in chaves:
            a, base, cobertura, metrica = chave
            amostras[chave][b_idx] = (
                cache[(a, cobertura)][metrica] - cache[(base, cobertura)][metrica]
            )

    linhas = []
    for chave in chaves:
        a, base, cobertura, metrica = chave
        da = por_variante[a]
        db = por_variante[base]
        ma = calcular_metricas(
            da, pesos_resposta_para_cobertura(da["score"].to_numpy(), cobertura)
        )[metrica]
        mb = calcular_metricas(
            db, pesos_resposta_para_cobertura(db["score"].to_numpy(), cobertura)
        )[metrica]
        validas = amostras[chave][np.isfinite(amostras[chave])]
        lo, hi = np.quantile(validas, [0.025, 0.975])
        linhas.append(
            {
                "a": a,
                "b": base,
                "target_coverage": cobertura,
                "metric": metrica,
                "delta_a_minus_b": ma - mb,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_resamples": n_reamostras,
                "method": "paired stratified percentile bootstrap",
            }
        )
    return pd.DataFrame(linhas)


def montar_dados_ponto_materializado(
    run_dir: Path,
    c_min: float,
    audit_run_dir: Path,
) -> pd.DataFrame:
    caminho = run_dir / f"consolidated_test_full_cmin={c_min}.parquet"
    d = pd.read_parquet(caminho)
    suficiencia = pd.read_parquet(run_dir / "metrics" / "suficiencia_contexto.parquet")
    d = d.merge(
        suficiencia[["item_id", "veredicto_suficiencia"]], on="item_id", how="left"
    )
    auditoria = pd.read_parquet(audit_run_dir / "metrics" / "auditoria_recusas.parquet")
    auditoria = auditoria.rename(columns={"agent_id": "agent_chosen"})

    respostas = []
    for agente in sorted(d["agent_chosen"].unique()):
        r = pd.read_parquet(run_dir / "agent_responses" / agente / "responses.parquet")
        r["textual_refusal"] = r["response_text"].fillna("").str.contains(
            ablacao.PADRAO_RECUSA
        )
        respostas.append(r[["item_id", "agent_id", "textual_refusal"]])
    resp = pd.concat(respostas).rename(columns={"agent_id": "agent_chosen"})
    d = d.merge(resp, on=["item_id", "agent_chosen"], how="left")
    d = d.merge(
        auditoria[["item_id", "agent_chosen", "veredicto_recusa"]],
        on=["item_id", "agent_chosen"],
        how="left",
    )
    analitica = pd.DataFrame(
        {
            "item_id": d["item_id"],
            "has_doc_ref": d["has_doc_ref"].astype(bool),
            "z_winner": d["z_winner"].astype(int),
            "textual_refusal": d["textual_refusal"].astype(bool),
            "refusal_verdict": d["veredicto_recusa"],
            "context_sufficiency": d["veredicto_suficiencia"].astype(int),
            "prob_response": (~d["abstained"].astype(bool)).astype(float),
        }
    )
    return analitica


def montar_ponto_materializado(
    dados: pd.DataFrame,
    run: str,
    c_min: float,
    nome: str,
) -> dict[str, float | str]:
    return {
        "run": run,
        "variant": nome,
        "c_min": c_min,
        **calcular_metricas(dados, dados["prob_response"].to_numpy(dtype=float)),
    }


def contraste_pontos_materializados(
    dados_a: pd.DataFrame,
    nome_a: str,
    dados_b: pd.DataFrame,
    nome_b: str,
    n_reamostras: int,
    seed: int,
) -> pd.DataFrame:
    """Contraste pareado entre dois pontos de operação materializados."""
    a = dados_a.set_index("item_id").sort_index()
    b = dados_b.set_index("item_id").sort_index()
    if not a.index.equals(b.index):
        raise RuntimeError("Os runs materializados não contêm os mesmos itens")
    if not np.array_equal(a["has_doc_ref"], b["has_doc_ref"]):
        raise RuntimeError("Estratos has_doc_ref divergem entre os runs")

    def todas_metricas(dados: pd.DataFrame) -> dict[str, float]:
        return calcular_metricas(
            dados, dados["prob_response"].to_numpy(dtype=float)
        )

    ponto_a = todas_metricas(a)
    ponto_b = todas_metricas(b)
    estrato = a["has_doc_ref"].to_numpy(dtype=bool)
    rng = np.random.default_rng(seed)
    amostras = {
        metrica: np.empty(n_reamostras, dtype=float)
        for metrica in METRICAS_PONTO_CONTRASTE
    }
    for i in range(n_reamostras):
        idx = _reamostra_estratificada(estrato, rng)
        ma = todas_metricas(a.iloc[idx])
        mb = todas_metricas(b.iloc[idx])
        for metrica in METRICAS_PONTO_CONTRASTE:
            amostras[metrica][i] = ma[metrica] - mb[metrica]

    linhas = []
    for metrica in METRICAS_PONTO_CONTRASTE:
        validas = amostras[metrica][np.isfinite(amostras[metrica])]
        lo, hi = np.quantile(validas, [0.025, 0.975])
        linhas.append(
            {
                "a": nome_a,
                "b": nome_b,
                "metric": metrica,
                "delta_a_minus_b": ponto_a[metrica] - ponto_b[metrica],
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_resamples": n_reamostras,
                "method": "paired stratified percentile bootstrap",
            }
        )
    return pd.DataFrame(linhas)


def _markdown(df: pd.DataFrame) -> str:
    def fmt(valor: object) -> str:
        if isinstance(valor, (float, np.floating)):
            return "NA" if not np.isfinite(valor) else f"{valor:.4f}"
        return str(valor)

    cols = list(df.columns)
    linhas = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    linhas.extend(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join(linhas)


def _salvar(df: pd.DataFrame, base: Path, titulo: str) -> None:
    parquet = Path(f"{base}.parquet")
    md = Path(f"{base}.md")
    df.to_parquet(parquet, index=False)
    md.write_text(
        "<!-- gerado por scripts/mensuracoes_decisao_artigo2.py; não editar à mão -->\n"
        f"# {titulo}\n\n{_markdown(df)}\n",
        encoding="utf-8",
    )
    print(parquet)
    print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="phase2_local_full_715")
    p.add_argument("--meanraw-run", default="phase2_local_full_715_meanraw")
    p.add_argument("--audit-run", default="phase2_local_full_715")
    p.add_argument("--c-min", type=float, default=0.7)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=list(COBERTURAS_PADRAO),
    )
    args = p.parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap deve ser >= 100")
    coberturas = tuple(args.coverages)
    run_dir = RAIZ / "runs" / args.run
    meanraw_dir = RAIZ / "runs" / args.meanraw_run
    audit_dir = RAIZ / "runs" / args.audit_run
    DESTINO.mkdir(exist_ok=True)

    itens, _ = montar_tabela_itens(run_dir, args.c_min, audit_dir)
    sufixo = f"{args.run}_cmin{args.c_min}"
    caminho_itens = DESTINO / f"auditoria_decisao_itens_{sufixo}.parquet"
    itens.to_parquet(caminho_itens, index=False)
    print(caminho_itens)

    metricas = metricas_por_cobertura(itens, coberturas)
    _salvar(
        metricas,
        DESTINO / f"metricas_decisao_cobertura_{sufixo}",
        f"Métricas de decisão por cobertura — {args.run}",
    )

    contrastes = contrastes_bootstrap(
        itens, coberturas, n_reamostras=args.bootstrap, seed=args.seed
    )
    _salvar(
        contrastes,
        DESTINO / f"contrastes_decisao_{sufixo}",
        f"Contrastes pareados de decisão — {args.run}",
    )

    dados_a0 = montar_dados_ponto_materializado(run_dir, args.c_min, audit_dir)
    dados_a5 = montar_dados_ponto_materializado(meanraw_dir, args.c_min, audit_dir)
    pontos = pd.DataFrame(
        [
            montar_ponto_materializado(dados_a0, run_dir.name, args.c_min, "A0"),
            montar_ponto_materializado(dados_a5, meanraw_dir.name, args.c_min, "A5"),
        ]
    )
    _salvar(
        pontos,
        DESTINO / f"pontos_operacao_decisao_cmin{args.c_min}",
        f"Métricas complementares nos pontos materializados — C_min={args.c_min}",
    )
    contraste_pontos = contraste_pontos_materializados(
        dados_a5,
        "A5",
        dados_a0,
        "A0",
        n_reamostras=args.bootstrap,
        seed=args.seed,
    )
    _salvar(
        contraste_pontos,
        DESTINO / f"contraste_pontos_operacao_decisao_cmin{args.c_min}",
        f"Contraste pareado A5 − A0 nos pontos materializados — C_min={args.c_min}",
    )


if __name__ == "__main__":
    main()
