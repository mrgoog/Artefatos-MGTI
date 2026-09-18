"""Números canônicos do RAG-2 (arch §5): gera o ledger docs/numeros_canonicos_rag2.md e as tabelas do apêndice a partir dos artefatos versionados em runs/ e data/, e confere que toda cifra do relatório tem lastro. Só leitura; sem GPU, sem modelo; determinístico (sem datas)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from rag2.artefatos import ler_resumo
from rag2.braco_api import AGENTES_API, ARQUIVO_CONTRASTES, NOME_API
from rag2.config import RAIZ
from rag2.confirmacao import DIR_CONFIRMACAO, NOME_NOVO
from rag2.constantes import (
    AGENTES,
    B_BOOTSTRAP,
    CANONICO_R0,
    COTA_CARF,
    COTA_NORMAS,
    MODELO_EMBEDDER,
    MODELO_JUIZ,
    MODELO_REFORMULADOR,
    MODELO_RERANKER,
    N_FINAL,
    N_REFORMULACOES,
    POOL_POR_RAMO,
    POOL_RERANK,
    RRF_K,
    RUN_PILOTO,
    RUN_PRODUCAO,
    SEMENTE,
    TAMANHO_AMOSTRA_B,
    TAMANHO_AMOSTRA_CONFIRMACAO,
    TOLERANCIA_CANONICO,
)
from rag2.escada import DIR_ESCADA
from rag2.geracao import DIR_PROMPTS

ARQUIVO_NUMEROS = RAIZ / "docs" / "numeros_canonicos_rag2.md"
"""O ledger (versionado, gerado): nenhuma cifra entra em docs/relatorio_apendice.md sem linha aqui (arch §5)."""
ARQUIVO_RELATORIO = RAIZ / "docs" / "relatorio_apendice.md"
ARQUIVO_REPARO = RAIZ / "data" / "reparado" / "relatorio_reparo.md"
DEGRAUS = ("R0", "R0'", "R1", "R2", "R3", "R5")
"""A escada executada, na ordem; R4 não foi executado (decisão 2026-09-12)."""
REGEX_CIFRA = re.compile(r"\d+,\d{4}")
"""Cifra com 4 casas e vírgula decimal — o que o gate confere no relatório (o registro da dissertação usa vírgula)."""
REGEX_PONTO = re.compile(r"\d+\.\d{4}")
"""Cifra com 4 casas e PONTO decimal — formato proibido no apêndice; denuncia colagem de `rag2 tabela`/`tabela-prompts` (ponto) num texto que deveria usar vírgula."""
PALAVRAS_CANONICO_ANTIGO = 5009
PILOTO_API_ADEQUACAO = 0.245
"""Média dos três agentes de API do piloto da origem (phase1_api_pilot_200_1: 0,260; 0,240; 0,235) — `../runs/phase1_api_pilot_200_1/judge_labels/` (média de z por agente); três casas, como na origem."""
"""'Orçamento de contexto da produção' do ledger da origem (coluna tokens_aprox) — outra definição de 'palavras'; declarado, nunca comparado com contar_palavras."""
NOMES_METRICAS = {
    "recall_passagem": "recall de passagem", "recall_arquivo": "recall de arquivo",
    "z_media": "adequação média (3 agentes)", "z_qwen35_9b": "adequação qwen35_9b",
    "z_granite41_8b": "adequação granite41_8b", "z_gemma4_e4b": "adequação gemma4_e4b",
    "z_b0": "adequação B0 (melhor agente fixo)", "recusa": "recusa textual",
    "evasiva": "recusa evasiva (auditor = 1)", "falha_parse": "falha de parse",
    "suficiente": "contexto suficiente (= 2)", "insuficiente": "contexto insuficiente (= 0)",
    "palavras": "palavras/item", "n_chunks": "chunks/item",
}
"""Nome em português de cada métrica de comparacao.json, na tabela do apêndice."""
NOMES_TABELAS = ("escada", "hipoteses", "geracao", "comparacao", "custo")

_TROCA = str.maketrans(",.", ".,")


def fmt(x: float, casas: int = 4) -> str:
    """Vírgula decimal e ponto de milhar: fmt(0.23549) = '0,2355'; fmt(3264.68, 1) = '3.264,7'."""
    return f"{float(x):,.{casas}f}".translate(_TROCA)


def fmt_int(x: float) -> str:
    """Inteiro arredondado com ponto de milhar: fmt_int(3199.86) = '3.200'."""
    return f"{round(float(x)):,}".translate(_TROCA)


def fmt_sinal(x: float, casas: int = 4) -> str:
    """Sinal explícito, menos tipográfico (U+2212): '+0,1171', '−0,0100'; zero é '+0,0000'."""
    return ("−" if x < 0 else "+") + fmt(abs(x), casas)


def fmt_ic(lo: float, hi: float, casas: int = 4, *, sinal: bool = False) -> str:
    """IC no registro da dissertação: '[0,2072; 0,2658]' ou, com sinal, '[+0,0712; +0,1661]'."""
    f = (lambda v: fmt_sinal(v, casas)) if sinal else (lambda v: fmt(v, casas))
    return f"[{f(lo)}; {f(hi)}]"


def carregar() -> dict:
    """Todos os artefatos que o ledger cita (versionados). Falta de qualquer um é FileNotFoundError com o caminho."""
    escada = {d: ler_resumo(DIR_ESCADA / d) for d in DEGRAUS}
    if faltam := [d for d, r in escada.items() if r is None]:
        raise FileNotFoundError(f"resumo.json ausente em runs/escada/ para {faltam}")
    novo = ler_resumo(DIR_CONFIRMACAO / NOME_NOVO)
    if novo is None:
        raise FileNotFoundError(f"runs/confirmacao/{NOME_NOVO}/resumo.json ausente: M5.1 não registrado")
    comparacao = json.loads((DIR_CONFIRMACAO / "comparacao.json").read_text(encoding="utf-8"))
    indice = json.loads((DIR_ESCADA / "R0'" / "indice_metadata.json").read_text(encoding="utf-8"))
    m = re.search(r"pares que passam a casar: (\d+)", ARQUIVO_REPARO.read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"{ARQUIVO_REPARO} sem a linha 'pares que passam a casar: N'")
    producao = ler_resumo(DIR_ESCADA / f"producao_{RUN_PRODUCAO}")
    smoke = ler_resumo(DIR_PROMPTS / "P0")
    if faltam := [p for p, r in ((f"runs/escada/producao_{RUN_PRODUCAO}", producao), ("runs/prompts/P0", smoke)) if r is None]:
        raise FileNotFoundError(f"resumo.json ausente para {faltam} (M0.2/M4.1 não registrados)")
    p_api = DIR_CONFIRMACAO / NOME_API / ARQUIVO_CONTRASTES
    if not p_api.exists():
        raise FileNotFoundError(f"{p_api} ausente: rode `rag2 confirmacao braco-api`")
    return {
        "escada": escada,
        "producao": producao,
        "smoke": smoke,
        "novo": novo,
        "comparacao": comparacao,
        "indice": indice,
        "pares_reparo": int(m.group(1)),
        "api": json.loads(p_api.read_text(encoding="utf-8")),
    }


def veredito(degrau: Mapping, anterior: Mapping) -> str:
    """Critério pré-registrado §6: 'confirmada' se o IC95 do degrau não inclui o recall de passagem do anterior (ic_lo > ponto anterior)."""
    return "confirmada" if degrau["ic_recall_passagem"][0] > anterior["recall_passagem"] else "não confirmada"


def _rp(r: Mapping) -> str:
    """'0,3426 [0,3106; 0,3754]' de um resumo da escada."""
    return f"{fmt(r['recall_passagem'])} {fmt_ic(*r['ic_recall_passagem'])}"


def _inclui(v: str) -> str:
    return "exclui" if v == "confirmada" else "inclui"


def tabelas_relatorio(a: Mapping | None = None) -> dict[str, str]:
    """As cinco tabelas do apêndice (markdown, registro da dissertação), na ordem de NOMES_TABELAS; o relatório as cola verbatim."""
    a = a or carregar()
    e = a["escada"]
    t: dict[str, str] = {}
    linhas = [
        "| degrau | recall de passagem [IC95] | recall de arquivo | Hit@K | itens sem dispositivo | chunks/item | palavras/item | teto oracular |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in DEGRAUS:
        r = e[d]
        linhas.append(
            f"| {d} | {_rp(r)} | {fmt(r['recall_arquivo'])} | {fmt(r['hit_passagem'])} | {fmt(r['frac_zero'])} "
            f"| {fmt(r['n_chunks_medio'], 2)} | {fmt_int(r['palavras_medias'])} | {fmt(r['teto_oracular'])} |"
        )
    t["escada"] = "\n".join(linhas)
    v1, v2, v3, v5 = veredito(e["R1"], e["R0"]), veredito(e["R2"], e["R1"]), veredito(e["R3"], e["R2"]), veredito(e["R5"], e["R3"])
    h1 = "confirmada" if "confirmada" in (v1, v2) else "não confirmada"
    h3 = "confirmada" if e["R0'"]["teto_oracular"] > e["R0"]["teto_oracular"] else "não confirmada"
    hip = [
        ("H1", "relaxar a dedup e reranquear (R1, R2) recupera os itens zerados por ordenação",
         f"R1 {_rp(e['R1'])}: IC {_inclui(v1)} R0 ({fmt(e['R0']['recall_passagem'])}); R2 {_rp(e['R2'])}: IC {_inclui(v2)} R1 ({fmt(e['R1']['recall_passagem'])})", h1),
        ("H2", "a cota por tipo de fonte (R3) corrige a dominância dos acórdãos",
         f"R3 {_rp(e['R3'])}: IC {_inclui(v3)} R2 ({fmt(e['R2']['recall_passagem'])}); itens sem dispositivo {fmt(e['R2']['frac_zero'])} → {fmt(e['R3']['frac_zero'])}", v3),
        ("H3", "o reparo do corpus eleva o teto oracular acima de 0,9145",
         f"teto oracular {fmt(e['R0']['teto_oracular'])} → {fmt(e[chr(82) + chr(48) + chr(39)]['teto_oracular'])} ({a['pares_reparo']} pares passam a casar); R0' {_rp(e[chr(82) + chr(48) + chr(39)])}", h3),
        ("H4", "metadados taxonômicos como boost (R4) dão ganho pequeno", "R4 não executado (decisão 2026-09-12; M2.1–M2.3 abandonados)", "não testada"),
        ("H5", "multi-query (R5) ataca a classe \"pool\"", f"R5 {_rp(e['R5'])}: IC {_inclui(v5)} R3 ({fmt(e['R3']['recall_passagem'])})", v5),
    ]
    t["hipoteses"] = "\n".join(
        ["| hipótese | o que testa | resultado | veredito |", "|---|---|---|---|"] + [f"| {h} | {o} | {r} | {v} |" for h, o, r, v in hip]
    )
    n = a["novo"]
    s = n["suficiencia"]
    pa = n["por_agente"]
    t["geracao"] = "\n".join(
        [
            "| variante | degrau | N | adequação média [IC95] | qwen35_9b | granite41_8b | gemma4_e4b | recusa [IC95] | evasiva [IC95] | falha de parse | suficiente [IC95] | parcial | insuficiente | palavras/item |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {n['variante']} | {n['degrau']} | {n['n_itens']} | {fmt(n['adequacao_media'])} {fmt_ic(*n['ic_adequacao_media'])} "
            + " ".join(f"| {fmt(pa[aid]['adequacao'])}" for aid in AGENTES)
            + f" | {fmt(n['taxa_recusa'])} {fmt_ic(*n['ic_taxa_recusa'])} | {fmt(n['taxa_evasiva'])} {fmt_ic(*n['ic_taxa_evasiva'])} | {fmt(n['taxa_falha_parse'])} "
            f"| {fmt(s['frac_suficiente'])} {fmt_ic(*s['ic_frac_suficiente'])} | {s['n_parcial']} | {s['n_insuficiente']} | {fmt_int(n['palavras_medias'])} |",
        ]
    )
    c = a["comparacao"]
    t["comparacao"] = "\n".join(
        ["| métrica | n | atual | novo | Δ novo − atual [IC95] |", "|---|---:|---:|---:|---:|"]
        + [f"| {NOMES_METRICAS[m['metrica']]} | {m['n']} | {fmt(m['atual'])} | {fmt(m['novo'])} | {fmt_sinal(m['delta'])} {fmt_ic(m['ic_lo'], m['ic_hi'], sinal=True)} |" for m in c["metricas"]]
    )
    at, nv = c["custo"]["atual"], c["custo"]["novo"]
    seg = nv["segundos"]
    seg_agentes = sum(v for k, v in seg.items() if k.startswith("agente_"))
    linhas = ["| etapa | atual: chamadas | atual: US$ | novo: chamadas | novo: US$ | novo: segundos |", "|---|---:|---:|---:|---:|---:|"]
    for etapa, rotulo, s_ in (("agentes", "agentes", seg_agentes), ("juiz", "juiz", seg["juiz"]), ("auditor", "auditor", seg["auditor"]), ("suficiencia", "suficiência", seg["suficiencia"])):
        linhas.append(f"| {rotulo} | {at[etapa]['chamadas']} | {fmt(at[etapa]['custo_usd'])} | {nv[etapa]['chamadas']} | {fmt(nv[etapa]['custo_usd'])} | {fmt_int(s_)} |")
    linhas.append(f"| contexto (recuperação) | — | — | — | — | {fmt_int(seg['contexto'])} |")
    linhas.append(f"| total | | {fmt(at['custo_usd_total'])} | | {fmt(nv['custo_usd_total'])} | {fmt_int(sum(seg.values()))} |")
    t["custo"] = "\n".join(linhas)
    return t


def _linhas(rows: Sequence[tuple[str, str, str]]) -> str:
    """Tabela grandeza | valor | estado | procedência (estado ✅ = apurado de artefato versionado)."""
    return "\n".join(["| grandeza | valor | estado | procedência |", "|---|---|---|---|"] + [f"| {g} | {v} | ✅ | {p} |" for g, v, p in rows])


def ledger(a: Mapping | None = None) -> str:
    """docs/numeros_canonicos_rag2.md inteiro: cabeçalho (regra de uso, legenda), §1–§9 com as grandezas e procedências, §10 as cinco tabelas do apêndice (embutidas, para que toda célula tenha lastro), §11 como manter. Sem datas: determinístico."""
    a = a or carregar()
    e, n, c, s = a["escada"], a["novo"], a["comparacao"], a["novo"]["suficiencia"]
    R = "runs/escada"

    def esc(d: str) -> str:
        return f"`{R}/{d}/resumo.json`"

    nov = f"`runs/confirmacao/{NOME_NOVO}/resumo.json`"
    cmp_ = "`runs/confirmacao/comparacao.json`"
    api = a["api"]
    api_ = f"`runs/confirmacao/{NOME_API}/{ARQUIVO_CONTRASTES}`"
    cst = "`src/rag2/constantes.py`"
    partes = [
        "# Números canônicos — RAG-2 (apêndice)\n",
        (
            "**Regra de uso:** nenhuma cifra entra em `docs/relatorio_apendice.md` sem constar deste arquivo (arch §5). Se um "
            "número precisar existir e não estiver aqui, ele é primeiro apurado de um artefato versionado, registrado aqui com "
            "procedência, e só então escrito no texto — nunca o caminho inverso.\n"
        ),
        (
            "**Estado:** **gerado** por `uv run rag2 numeros` a partir dos artefatos versionados em `runs/` e `data/`; não edite à "
            "mão — regenere. `uv run rag2 numeros --conferir docs/relatorio_apendice.md` confere que toda cifra com 4 casas do "
            "relatório está aqui e que as cinco tabelas de §11 aparecem nele verbatim. As cifras do braço de API (§10) são "
            "exploratórias: conferíveis, mas fora do pré-registro.\n"
        ),
        (
            "**Legenda de estado:**\n\n| símbolo | significado |\n|---|---|\n| ✅ | apurado de artefato versionado neste repositório; "
            "reproduzível por `rag2 numeros` |\n| ⚠️ | apurado, mas diverge do que o relatório publica — exige edição |\n| ❌ | citado "
            "e sem lastro — não usar |\n"
        ),
        (
            "**Duas armadilhas que este arquivo existe para evitar:** (1) há **duas definições de \"palavras\"** — ver §8; "
            "(2) a comparação atual × novo (§6) **não é evidência causal** — deriva de runtime dos agentes, decisão B3 de "
            "2026-09-13 (`docs/decisions.md`); o apêndice a publica com esse caveat.\n"
        ),
        "---\n\n## 1. Desenho experimental\n\n"
        + _linhas(
            [
                ("itens do BR-TaxQA-R", "715", "`questions_QA_2024_v1.1.json` (origem)"),
                ("itens com dispositivo identificável (escada)", str(e["R0"]["n_itens"]), esc("R0")),
                ("dispositivos esperados (pares arquivo–artigo)", fmt_int(e["R0"]["n_dispositivos"]), esc("R0")),
                ("amostra da confirmação (piloto)", f"{TAMANHO_AMOSTRA_CONFIRMACAO} ({c['n_has_doc_ref']} has_doc_ref; {c['n_com_dispositivo']} com dispositivo)", f"{cmp_}; `{RUN_PILOTO}` (origem)"),
                ("frente B (60 itens)", f"{TAMANHO_AMOSTRA_B} — **não executada** (decisão 2026-09-13)", "`docs/decisions.md`"),
                ("agentes", ", ".join(f"{k} ({v})" for k, v in AGENTES.items()), f"{cst}; `manifest.yaml` da produção"),
                ("juiz, auditor e juiz de suficiência", f"`{MODELO_JUIZ}`, temperatura 0", cst),
                ("embedder / reranker / reformulador", f"`{MODELO_EMBEDDER}` / `{MODELO_RERANKER}` / `{MODELO_REFORMULADOR}`", cst),
                ("hiperparâmetros pré-registrados", f"pool {POOL_POR_RAMO}+{POOL_POR_RAMO}; RRF k = {RRF_K}; rerank {POOL_RERANK}; cota {COTA_NORMAS} normas + {COTA_CARF} CARF; N_FINAL = {N_FINAL}; reformulações {N_REFORMULACOES}", f"{cst} ≡ `docs/pre-registro.md` §4"),
                ("bootstrap", f"percentil, B = {fmt_int(B_BOOTSTRAP)}, semente {SEMENTE}, estratificado por has_doc_ref na geração", cst),
                ("índice reparado", f"{fmt_int(a['indice']['n_chunks'])} chunks ({fmt_int(a['indice']['chunks']['norma'])} normas + {fmt_int(a['indice']['chunks']['acordao_carf'])} CARF), sem pseudo-docs", "`runs/escada/R0'/indice_metadata.json`"),
            ]
        ),
        "\n## 2. Canônico e R0\n\n"
        + _linhas(
            [
                ("recall de passagem canônico (diagnóstico)", fmt(CANONICO_R0), f"{cst} (`CANONICO_R0`); `reference/diagnostico_recall_pool.md` §1"),
                ("R0 pela cadeia nova (mesmo índice de produção)", _rp(e["R0"]), esc("R0")),
                ("Δ R0 − canônico", f"{fmt_sinal(e['R0']['recall_passagem'] - CANONICO_R0)} (tolerância {fmt(TOLERANCIA_CANONICO, 3)})", esc("R0")),
                ("produção recomputada dos artefatos (`metricas-run`)", fmt(a["producao"]["recall_passagem"]), esc(f"producao_{RUN_PRODUCAO}")),
            ]
        ),
        "\n## 3. Escada de recuperação (556 itens)\n\n"
        + _linhas(
            [
                (f"{d} — recall de passagem [IC95]; arquivo; Hit@K; sem dispositivo; chunks; palavras; teto",
                 f"{_rp(e[d])}; {fmt(e[d]['recall_arquivo'])}; {fmt(e[d]['hit_passagem'])}; {fmt(e[d]['frac_zero'])}; {fmt(e[d]['n_chunks_medio'], 2)}; {fmt_int(e[d]['palavras_medias'])}; {fmt(e[d]['teto_oracular'])}",
                 esc(d))
                for d in DEGRAUS
            ]
            + [("R4", "não executado (decisão 2026-09-12); β = 0,10 e γ = 0,05 sem uso", "`docs/decisions.md`; `docs/pre-registro.md` §3"),
               ("pares esperados que passam a casar com o reparo", str(a["pares_reparo"]), "`data/reparado/relatorio_reparo.md`"),
               ("melhor degrau de A (critério §6)", "R3", "`docs/decisions.md` 2026-09-13; §4")]
        ),
        "\n## 4. Hipóteses (critério pré-registrado §6: o IC95 do degrau não inclui o ponto do anterior)\n\n" + tabelas_relatorio(a)["hipoteses"] + "\n",
        "\n## 5. Geração no pipeline novo — P0 sobre R3, N = 200\n\n"
        + _linhas(
            [
                ("adequação média (3 agentes) [IC95]", f"{fmt(n['adequacao_media'])} {fmt_ic(*n['ic_adequacao_media'])}", nov),
            ]
            + [(f"adequação {aid}", f"{fmt(n['por_agente'][aid]['adequacao'])} ({n['por_agente'][aid]['n_adequadas']}/{n['n_itens']})", nov) for aid in AGENTES]
            + [
                ("recusa textual [IC95]", f"{fmt(n['taxa_recusa'])} {fmt_ic(*n['ic_taxa_recusa'])}", nov),
                ("recusa evasiva (auditor = 1) [IC95]", f"{fmt(n['taxa_evasiva'])} {fmt_ic(*n['ic_taxa_evasiva'])}", nov),
                ("falha de parse", fmt(n["taxa_falha_parse"]), nov),
                ("suficiência de contexto: suficiente [IC95]; parcial; insuficiente; parse falho", f"{s['n_suficiente']} ({fmt(s['frac_suficiente'])} {fmt_ic(*s['ic_frac_suficiente'])}); {s['n_parcial']}; {s['n_insuficiente']}; {s['n_falha_parse']}", nov),
                ("palavras/item; chunks/item", f"{fmt_int(n['palavras_medias'])}; {fmt(n['n_chunks_medio'], 2)}", nov),
                ("smoke test de M4.1 (20 itens) — **mecanismo, não resultado**", f"{fmt(a['smoke']['adequacao_media'])} {fmt_ic(*a['smoke']['ic_adequacao_media'])}", "`runs/prompts/P0/resumo.json`; `docs/decisions.md` 2026-09-13"),
                ("P1–P3", "não construídos (condicionais, a jusante do N = 200)", "`docs/decisions.md` 2026-09-12"),
            ]
        ),
        "\n## 6. Comparação pareada atual × novo (N = 200; recall nos 151 com dispositivo)\n\n"
        + _linhas(
            [(NOMES_METRICAS[m["metrica"]], f"atual {fmt(m['atual'])}; novo {fmt(m['novo'])}; Δ {fmt_sinal(m['delta'])} {fmt_ic(m['ic_lo'], m['ic_hi'], sinal=True)}; n = {m['n']}", cmp_) for m in c["metricas"]]
            + [("B0 (melhor agente fixo, escolhido na produção)", c["b0"], cmp_),
               ("condição de revisão B3 (IC de Δ adequação média exclui 0 por cima?)", "não" if not c["revisao_b3"]["significativamente_melhor"] else "**sim — revisar**", f"{cmp_}; `docs/decisions.md` 2026-09-13")]
        ),
        "\n## 7. Custo nos 200\n\n"
        + _linhas(
            [(f"{lado}: {etapa} — chamadas; US$", f"{c['custo'][lado][etapa]['chamadas']}; {fmt(c['custo'][lado][etapa]['custo_usd'])}", cmp_)
             for lado in ("atual", "novo") for etapa in ("agentes", "juiz", "auditor", "suficiencia")]
            + [(f"{lado}: total US$", fmt(c["custo"][lado]["custo_usd_total"]), cmp_) for lado in ("atual", "novo")]
            + [("novo: segundos de parede por etapa", "; ".join(f"{k} {fmt_int(v)}" for k, v in c["custo"]["novo"]["segundos"].items()), f"{cmp_} (de `runs/confirmacao/{NOME_NOVO}/custo.json`)"),
               ("atual: segundos de parede", "não registrados na produção", "—"),
               ("palavras/item; chunks/item", f"atual {fmt(c['medias']['palavras_atual'], 1)}; {fmt(c['medias']['n_chunks_atual'], 2)} · novo {fmt(c['medias']['palavras_novo'], 1)}; {fmt(c['medias']['n_chunks_novo'], 2)}", cmp_)]
        ),
        "\n## 8. Duas definições de \"palavras\"\n\n"
        + _linhas(
            [
                ("palavras de contexto por item — este projeto (R0 = produção)", f"{fmt_int(e['R0']['palavras_medias'])} = `contar_palavras` (split por espaço em branco do texto dos chunks servidos)", esc("R0")),
                ("palavras de contexto por item — canônico antigo", f"{fmt_int(PALAVRAS_CANONICO_ANTIGO)} = coluna `tokens_aprox` de `varredura_recuperacao.parquet`", "`../runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/metrics/varredura_recuperacao.parquet`"),
                ("regra", "definições diferentes para o mesmo contexto: **nunca comparar uma com a outra**; o apêndice usa só a primeira", "`docs/tooling-and-debugging.md`"),
            ]
        ),
        "\n## 9. Desvios do pré-registro e vieses declarados\n\n"
        + _linhas(
            [
                ("R4 não executado; M2.1–M2.3 abandonados; H4 não testada", "2026-09-12", "`docs/decisions.md`; `specs/progress.md`"),
                ("R5 construído sobre R3 (não R4), cota 5+2 e corte 7 mantidos", "2026-09-12", "`docs/decisions.md`"),
                ("parâmetros do reformulador fixados (temp 0, seed 42, max_tokens 512)", "2026-09-12", "`docs/decisions.md`"),
                ("frente B (60) não executada; M4.1 = smoke de 20 itens; P1–P3 não construídos", "2026-09-13", "`docs/decisions.md`"),
                ("MAX_TOKENS_AUDITOR = 4096 e MAX_TOKENS_SUFICIENCIA = 4096, seed 42 (valores dos scripts da origem)", "2026-09-13 / 2026-09-14", "`docs/decisions.md`"),
                ("comparabilidade: deriva de runtime dos agentes (1/9 respostas idênticas com o mesmo prompt); sem braço-controle P0×R0; comparação fora da dissertação (B3 waived)", "2026-09-13; condição de revisão não disparada em M5.2", "`docs/decisions.md`; §6"),
                ("config do llama-swap mudou de sha entre M4.1 e M5.1; flags dos três agentes idênticas", "2026-09-14", "`docs/decisions.md`"),
                ("reparo do corpus escolhe documentos a partir dos dispositivos esperados do conjunto de teste — viés a favor do novo", "pré-registro §7", "`docs/pre-registro.md`; `data/reparado/relatorio_reparo.md`"),
                ("taxonomias RFB/C não validadas por dupla codificação", "sem uso: R4 não executado", "`docs/pre-registro.md` §7"),
                ("amostra de 200 é a do piloto, não uma amostra nova; 49 itens sem dispositivo (sem recall)", "pré-registro §2/§7", f"{cmp_}"),
            ]
        ),
        "\n## 10. Braço de agentes de API sobre R3 — **exploratório, fora do pré-registro** (N = 200)\n\n"
        "Etiqueta: cifras apuradas de artefato e conferíveis, mas de um experimento fora do pré-registro (decisão 2026-09-14) — "
        "peso inferencial de exploração, não de confirmação. Contexto e suficiência reusados byte a byte de `novo`; agentes de API com "
        "`max_tokens` 4096 (piloto da origem) contra 1024 dos locais: comparação de configurações de geração sob contexto fixo.\n\n"
        + _linhas(
            [
                ("agentes de API", ", ".join(AGENTES_API), f"`runs/confirmacao/{NOME_API}/run_api.py`"),
                ("adequação média: produção; locais@R3; API@R3", "; ".join(fmt(api["adequacao_media"][k]) for k in ("atual", "local", "api")), api_),
                ("piloto de API sobre a recuperação antiga (média dos três agentes)", f"{fmt(PILOTO_API_ADEQUACAO, 3)} (0,260; 0,240; 0,235)", "`../runs/phase1_api_pilot_200_1/judge_labels/openai_gpt-5.4-mini/cache.parquet` (média de z por agente)"),
            ]
            + [(f"adequação {aid}", f"{fmt(api['por_agente_api'][aid]['adequacao'])} ({api['por_agente_api'][aid]['n_adequadas']}/{api['n_itens']})", api_) for aid in AGENTES_API]
            + [
                (f"Δ {rotulo} [IC95]; n", f"{fmt_sinal(c['delta'])} {fmt_ic(*c['ic'], sinal=True)}; {c['n']}", api_)
                for rotulo, c in (
                    ("API@R3 − locais@R3 (mesmo contexto; agente e orçamento de saída variam)", api["contrastes"]["api_menos_local"]),
                    ("locais@R3 − produção (≡ §6, adequação média)", api["contrastes"]["local_menos_atual"]),
                    ("API@R3 − produção (as duas mudanças)", api["contrastes"]["api_menos_atual"]),
                    ("API@R3 − locais@R3 dentro do estrato suficiente (reamostragem simples)", api["contrastes"]["api_menos_local_suficiente"]),
                )
            ]
            + [
                (f"adequação com contexto {nome}: API; local; n; contribuição ao Δ global",
                 f"{fmt(e['api'])}; {fmt(e['local'])}; {e['n']}; {fmt_sinal(e['contribuicao_delta'])}", api_)
                for nome, e in api["por_suficiencia"].items()
            ]
            + [
                ("adequação por agente no estrato suficiente (API)", "; ".join(f"{aid} {fmt(v)}" for aid, v in api["por_suficiencia"]["suficiente"]["por_agente_api"].items()), api_),
                ("adequação por agente no estrato suficiente (locais)", "; ".join(f"{aid} {fmt(v)}" for aid, v in api["por_suficiencia"]["suficiente"]["por_agente_local"].items()), api_),
                ("recusas textuais: API; locais; auditor da API (justificada; evasiva; não recusa)",
                 f"{api['recusas']['api']}; {api['recusas']['local']}; {api['recusas']['auditor_api']['justificada']}; {api['recusas']['auditor_api']['evasiva']}; {api['recusas']['auditor_api']['nao_recusa']}", api_),
                ("custo US$: geração por agente", "; ".join(f"{aid} {fmt(v)}" for aid, v in api["custo"]["geracao"].items()), f"{api_}; `responses.parquet` de cada agente"),
                ("custo US$: geração; avaliação (juiz; auditor; suficiência); total",
                 f"{fmt(api['custo']['geracao_total'])}; {fmt(api['custo']['avaliacao_total'])} ({fmt(api['custo']['avaliacao']['juiz'])}; {fmt(api['custo']['avaliacao']['auditor'])}; {fmt(api['custo']['avaliacao']['suficiencia'])}); {fmt(api['custo']['total'])}",
                 f"{api_}; `runs/confirmacao/{NOME_API}/custo.json`"),
            ]
        ),
        "\n## 11. Tabelas do apêndice (geradas — coladas verbatim em `docs/relatorio_apendice.md`)\n",
    ]
    for nome, tab in tabelas_relatorio(a).items():
        partes.append(f"\n### {nome}\n\n{tab}\n")
    partes.append(
        "\n## 12. Como manter este arquivo\n\n1. Nunca edite à mão: `uv run rag2 numeros` regenera tudo dos artefatos.\n"
        "2. Antes de commitar o apêndice: `uv run rag2 numeros --conferir docs/relatorio_apendice.md` (exit 0).\n"
        "3. Um número novo no apêndice nasce de um artefato versionado, entra aqui por `numeros.py`, e só então no texto.\n\n"
        "Comandos que produziram os artefatos citados:\n\n```bash\nuv run rag2 degrau R0   # … R0', R1, R2, R3, R5 (goldens em runs/escada/)\n"
        "uv run rag2 metricas-run phase2_local_full_715\nuv run rag2 variante P0 --degrau R3           # smoke (M4.1)\n"
        "RAG2_DEVICE=cuda:0 uv run rag2 confirmacao executar   # N=200 (M5.1)\nuv run rag2 confirmacao comparar              # comparação e custo (M5.2)\n"
        "uv run rag2 confirmacao braco-api             # contrastes do braço de API (exploratório, §10)\n"
        "uv run rag2 numeros                           # este arquivo\n```\n"
    )
    return "\n".join(partes)


def cifras(texto: str) -> list[str]:
    """Cifras com 4 casas e vírgula, únicas, na ordem de aparição."""
    return list(dict.fromkeys(REGEX_CIFRA.findall(texto)))


def conferir(relatorio: str, ledger_texto: str, tabelas: Mapping[str, str]) -> dict:
    """O gate de arch §5: {'cifras': n, 'sem_lastro': [cifras (vírgula) do relatório ausentes do ledger], 'formato_invalido': [cifras com ponto decimal — formato proibido], 'tabelas_ausentes': [tabelas geradas que não aparecem verbatim]}."""
    cs = cifras(relatorio)
    return {
        "cifras": len(cs),
        "sem_lastro": [x for x in cs if x not in ledger_texto],
        "formato_invalido": list(dict.fromkeys(REGEX_PONTO.findall(relatorio))),
        "tabelas_ausentes": [k for k, t in tabelas.items() if t not in relatorio],
    }
