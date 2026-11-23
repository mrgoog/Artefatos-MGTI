"""UI Streamlit para validação humana do juiz LLM.

Lê `data/validacao_humana/amostra.parquet` (gerado por `sortear_amostra_validacao.py`) e
grava `data/validacao_humana/anotacoes.parquet` a cada anotação salva.

Protocolo e limitação histórica:
  * o rótulo explícito do juiz e seu raciocínio ficam OCULTOS até a anotação ser salva;
  * nos pares substantivos, a interface exibe o estrato `adequada` ou `inadequada`,
    derivado do rótulo automático; portanto, o cegamento não é assegurado;
  * depois de salvar, um botão revela o juiz e abre um campo separado de comentário
    pós-revelação; o arquivo conserva apenas o estado mais recente da anotação;
  * respostas substantivas são julgadas contra a referência (o que o juiz binário viu);
    recusas são julgadas contra o contexto servido (o que o auditor viu).

O número de itens por estrato é escolhido na barra lateral; como a ordem de sorteio é
fixa, aumentar N depois preserva o que já foi anotado.

Uso:
    uv run streamlit run scripts/anotar_validacao.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA = Path("data/validacao_humana")
AMOSTRA = PASTA / "amostra.parquet"
ANOTACOES = PASTA / "anotacoes.parquet"

ROTULOS_SUBSTANTIVA = {
    1: "Adequada",
    0: "Inadequada",
    -1: "Não sei julgar",
}
ROTULOS_RECUSA = {
    2: "Justificada (contexto não permitia responder)",
    1: "Evasiva (contexto permitia responder)",
    0: "Não é recusa (o agente respondeu)",
    -1: "Não sei julgar",
}
CATEGORIAS_ERRO = [
    "Contradiz a referência em ponto material",
    "Incompleta: não esgota o que a pergunta exige",
    "Afirma algo não sustentado pelo material",
    "Fora do tópico / evasiva",
    "Correta com redação diferente da referência",
    "Referência (ground truth) parece problemática",
    "Outro (descrever)",
]
COLUNAS_ANOTACAO = [
    "item_id", "agent_id", "estrato", "rotulo_humano", "categorias_erro", "comentario",
    "revelado", "comentario_pos_revelacao", "timestamp",
]


# Persistência.

@st.cache_data
def carregar_amostra() -> pd.DataFrame:
    return pd.read_parquet(AMOSTRA)


def carregar_anotacoes() -> pd.DataFrame:
    if ANOTACOES.exists():
        return pd.read_parquet(ANOTACOES)
    return pd.DataFrame(columns=COLUNAS_ANOTACAO)


def salvar_anotacao(registro: dict) -> None:
    df = carregar_anotacoes()
    chave = (df["item_id"] == registro["item_id"]) & (df["agent_id"] == registro["agent_id"])
    df = df[~chave]
    df = pd.concat([df, pd.DataFrame([registro])], ignore_index=True)
    ANOTACOES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ANOTACOES, index=False)


def anotacao_de(df: pd.DataFrame, item_id: str, agent_id: str) -> dict | None:
    sel = df[(df["item_id"] == item_id) & (df["agent_id"] == agent_id)]
    return None if sel.empty else sel.iloc[0].to_dict()


# Sequência de exibição.

def montar_sequencia(amostra: pd.DataFrame, n: dict[str, int], modo: str) -> pd.DataFrame:
    partes = [
        amostra[(amostra["estrato"] == e) & (amostra["ordem_sorteio"] <= k)].sort_values("ordem_sorteio")
        for e, k in n.items()
        if k > 0
    ]
    if not partes:
        return amostra.iloc[0:0]
    if modo == "Por estrato":
        return pd.concat(partes, ignore_index=True)
    # intercalada: round-robin pela ordem de sorteio
    seq = pd.concat(partes, ignore_index=True)
    return seq.sort_values(["ordem_sorteio", "estrato"]).reset_index(drop=True)


# Renderização.

def _dispositivos_compactos(js: str) -> list[str]:
    """Extrai 'título — art. X' do dump JSON que o juiz recebeu."""
    try:
        refs = json.loads(js)
    except Exception:
        return [js]
    linhas = []
    for ref in refs:
        try:
            obj = json.loads(ref)
        except Exception:
            linhas.append(str(ref))
            continue
        for titulo, lista in obj.items():
            for ent in lista:
                arts = ", ".join(f"art. {a.get('artigo')}" for a in ent.get("artigos", []))
                linhas.append(f"{titulo}" + (f" — {arts}" if arts else ""))
    return linhas or [js]


def render_item(row: pd.Series, anot: dict | None) -> None:
    eh_recusa = row["estrato"] == "recusa"
    st.subheader(f"{row['item_id']} · {row['agent_id']} · estrato: **{row['estrato']}** (#{int(row['ordem_sorteio'])})")

    st.markdown("#### Pergunta")
    st.info(row["pergunta"])

    if eh_recusa:
        st.markdown("#### Resposta do agente (recusa textual)")
        st.warning(row["response_text"] or "(vazia)")
        st.caption(f"confiança declarada: {row['confidence_raw']:.2f}")
        with st.expander("Referência (o que uma resposta correta precisaria afirmar) — NÃO estava disponível ao agente", expanded=False):
            st.write(row["ground_truth"])
        st.markdown("#### Contexto servido ao agente (única fonte que ele viu)")
        st.text_area("contexto", row["contexto_servido"], height=420, label_visibility="collapsed", key=f"ctx_{row['item_id']}_{row['agent_id']}")
    else:
        col_ref, col_cand = st.columns(2)
        with col_ref:
            st.markdown("#### Referência (ground truth)")
            st.success(row["ground_truth"])
            disp = _dispositivos_compactos(row["dispositivos_legais"])
            with st.expander(f"Dispositivos citados na referência ({len(disp)})", expanded=False):
                for d in disp:
                    st.markdown(f"- {d}")
            ementas = json.loads(row["ementas_carf"]) if row["ementas_carf"] else []
            if ementas:
                with st.expander(f"Ementas CARF ({len(ementas)})"):
                    for e in ementas:
                        st.markdown(f"- {e}")
        with col_cand:
            st.markdown("#### Resposta candidata")
            st.warning(row["response_text"] or "(vazia)")
            st.caption(f"confiança declarada: {row['confidence_raw']:.2f}")
        st.caption("O juiz binário NÃO viu o contexto recuperado; julgue apenas pela consistência com a referência, como na rubrica.")


def render_formulario(row: pd.Series, anot: dict | None) -> None:
    eh_recusa = row["estrato"] == "recusa"
    rotulos = ROTULOS_RECUSA if eh_recusa else ROTULOS_SUBSTANTIVA
    chave = f"{row['item_id']}_{row['agent_id']}"

    st.markdown("---")
    st.markdown("#### Sua anotação" + (" (já salva — pode alterar)" if anot else ""))
    opcoes = list(rotulos.values())
    codigo_por_rotulo = {v: k for k, v in rotulos.items()}
    idx = opcoes.index(rotulos[int(anot["rotulo_humano"])]) if anot and int(anot["rotulo_humano"]) in rotulos else None
    escolha = st.radio(
        "Recusa foi:" if eh_recusa else "A resposta candidata é:",
        opcoes, index=idx, key=f"rot_{chave}", horizontal=not eh_recusa,
    )
    rotulo = codigo_por_rotulo.get(escolha)
    categorias: list[str] = []
    if not eh_recusa:
        prev = list(anot["categorias_erro"]) if anot and anot.get("categorias_erro") is not None else []
        categorias = st.multiselect("Problemas observados (se houver)", CATEGORIAS_ERRO, default=[c for c in prev if c in CATEGORIAS_ERRO], key=f"cat_{chave}")
    comentario = st.text_area("Comentário (opcional)", value=(anot or {}).get("comentario") or "", key=f"com_{chave}", height=80)

    if st.button("💾 Salvar anotação", type="primary", key=f"save_{chave}", disabled=rotulo is None):
        salvar_anotacao({
            "item_id": row["item_id"], "agent_id": row["agent_id"], "estrato": row["estrato"],
            "rotulo_humano": int(rotulo), "categorias_erro": categorias, "comentario": comentario,
            "revelado": bool((anot or {}).get("revelado", False)),
            "comentario_pos_revelacao": (anot or {}).get("comentario_pos_revelacao") or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        st.toast("Anotação salva.")
        st.rerun()

    if anot:
        st.markdown("#### Juiz LLM")
        if not anot.get("revelado"):
            if st.button("👁 Revelar rótulo e raciocínio do juiz", key=f"rev_{chave}"):
                anot["revelado"] = True
                salvar_anotacao({k: anot.get(k) for k in COLUNAS_ANOTACAO})
                st.rerun()
        else:
            if eh_recusa:
                v = int(row["veredicto_recusa"])
                st.markdown(f"**Auditor de recusas:** {ROTULOS_RECUSA.get(v, v)}  \n**Juiz binário:** z = {int(row['z'])} (Inadequada)")
                st.write(row["raciocinio_auditor"])
            else:
                st.markdown(f"**Juiz binário:** {ROTULOS_SUBSTANTIVA[int(row['z'])]}")
                st.write(row["raciocinio_juiz"])
            concorda = _concorda(row, anot)
            if concorda is not None:
                st.markdown("✅ **Concordância**" if concorda else "❌ **Desacordo**")
            pos = st.text_area("Comentário pós-revelação (por que discordou / o que o juiz viu que você não viu)", value=anot.get("comentario_pos_revelacao") or "", key=f"pos_{chave}", height=80)
            if st.button("Salvar comentário pós-revelação", key=f"savepos_{chave}"):
                anot["comentario_pos_revelacao"] = pos
                anot["timestamp"] = datetime.now(timezone.utc).isoformat()
                salvar_anotacao({k: anot.get(k) for k in COLUNAS_ANOTACAO})
                st.toast("Comentário salvo.")
                st.rerun()


def _concorda(row: pd.Series, anot: dict) -> bool | None:
    h = int(anot["rotulo_humano"])
    if h < 0:
        return None
    if row["estrato"] == "recusa":
        return h == int(row["veredicto_recusa"])
    return h == int(row["z"])


def render_resumo(seq: pd.DataFrame, anots: pd.DataFrame) -> None:
    m = seq.merge(anots, on=["item_id", "agent_id", "estrato"], how="left")
    st.markdown("### Andamento")
    linhas = []
    for estrato, g in m.groupby("estrato"):
        feitos = g["rotulo_humano"].notna()
        julgados = g[feitos & (g["rotulo_humano"] >= 0)]
        if estrato == "recusa":
            conc = (julgados["rotulo_humano"].astype(int) == julgados["veredicto_recusa"].astype(int)).sum()
        else:
            conc = (julgados["rotulo_humano"].astype(int) == julgados["z"].astype(int)).sum()
        linhas.append({
            "estrato": estrato, "na sequência": len(g), "anotados": int(feitos.sum()),
            "julgados": len(julgados), "concordam": int(conc),
            "acordo": f"{conc / len(julgados):.0%}" if len(julgados) else "—",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")


# App.

def main() -> None:
    st.set_page_config(page_title="Validação humana do juiz", layout="wide")
    amostra = carregar_amostra()
    anots = carregar_anotacoes()
    disponiveis = amostra["estrato"].value_counts().to_dict()

    with st.sidebar:
        st.markdown("### Amostra por estrato")
        n = {}
        for estrato, padrao in (("adequada", 20), ("inadequada", 15), ("recusa", disponiveis.get("recusa", 0))):
            n[estrato] = st.number_input(
                f"{estrato} (máx. {disponiveis.get(estrato, 0)})", 0, int(disponiveis.get(estrato, 0)),
                min(padrao, int(disponiveis.get(estrato, 0))), key=f"n_{estrato}",
            )
        modo = st.radio("Ordem de exibição", ["Por estrato", "Intercalada"], key="modo")
        seq = montar_sequencia(amostra, n, modo)
        total = len(seq)
        st.markdown("---")
        render_resumo(seq, anots)
        st.caption(f"anotações em `{ANOTACOES}`")

    if total == 0:
        st.warning("Nenhum item na sequência.")
        return

    if "pos" not in st.session_state:
        st.session_state.pos = 0
    st.session_state.pos = min(st.session_state.pos, total - 1)

    def ir_para(i: int) -> None:
        st.session_state.pos = i
        st.session_state.ir_para = i + 1
        st.rerun()

    feitos = seq.merge(anots[["item_id", "agent_id"]].assign(_ok=True), on=["item_id", "agent_id"], how="left")["_ok"].fillna(False).astype(bool).tolist()

    c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 2, 3])
    with c1:
        if st.button("◀ Anterior", disabled=st.session_state.pos == 0, width="stretch"):
            ir_para(st.session_state.pos - 1)
    with c2:
        if st.button("Próximo ▶", disabled=st.session_state.pos >= total - 1, width="stretch"):
            ir_para(st.session_state.pos + 1)
    with c3:
        pendentes = [i for i, ok in enumerate(feitos) if not ok]
        if st.button(f"⏭ Próximo pendente ({len(pendentes)})", disabled=not pendentes, width="stretch"):
            prox = next((i for i in pendentes if i > st.session_state.pos), pendentes[0])
            ir_para(prox)
    with c4:
        st.session_state.setdefault("ir_para", st.session_state.pos + 1)
        st.session_state.ir_para = min(st.session_state.ir_para, total)
        novo = st.number_input("Ir para #", 1, total, label_visibility="collapsed", key="ir_para")
        if int(novo) - 1 != st.session_state.pos:
            st.session_state.pos = int(novo) - 1
            st.rerun()
    with c5:
        st.progress(sum(feitos) / total, text=f"{sum(feitos)}/{total} anotados · item {st.session_state.pos + 1} de {total}")

    row = seq.iloc[st.session_state.pos]
    anot = anotacao_de(anots, row["item_id"], row["agent_id"])
    render_item(row, anot)
    render_formulario(row, anot)


if __name__ == "__main__":
    main()
