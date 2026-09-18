"""Auditoria dos itens em que o agente se RECUSOU a responder.

Reclassifica cada recusa como JUSTIFICADA (o contexto recuperado não permitia
responder) ou EVASIVA (o contexto permitia e o agente se esquivou), via um juiz
que lê o contexto efetivamente servido ao agente. A saída vai para
`runs/<run>/metrics/auditoria_recusas.parquet`; nenhum artefato do pipeline é
modificado.

Custo: uma chamada de juiz por resposta recusada (~146 no run de 715).

Uso:
    uv run python scripts/auditar_recusas.py --run phase2_local_full_715 --probe
    uv run python scripts/auditar_recusas.py --run phase2_local_full_715
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ensemble_llm.agentes.cliente_openrouter import ClienteOpenRouter
from ensemble_llm.configuracao import Segredos
from ensemble_llm.dados.conjunto_dados import carregar_br_taxqa_r
from ensemble_llm.persistencia.manifesto import ler_manifesto

INDICE_RAG = Path("runs/_shared/rag/br_taxqa_r_bge-m3_cuda1/denso/chunks.parquet")
QUESTOES = Path("data/brtaxq/questions_QA_2024_v1.1.json")

PADRAO_RECUSA = re.compile(
    r"não é possível responder|nao e possivel responder|não posso responder"
    r"|contexto (?:fornecido )?(?:é|e) insuficiente|não há informa|nao ha informa"
    r"|material fornecido não|não contém informa|não permite responder",
    re.IGNORECASE,
)

# Prompt de auditoria.

PROMPT_SISTEMA_AUDITOR = """Você é um auditor de sistemas de recuperação e geração (RAG) \
em legislação tributária brasileira.

Um agente de IA recebeu APENAS o "CONTEXTO RECUPERADO" abaixo e, com base nele, \
recusou-se a responder à pergunta, alegando material insuficiente.

Sua tarefa é decidir se essa recusa foi JUSTIFICADA — isto é, se o CONTEXTO RECUPERADO \
realmente não continha o necessário para responder.

REGRA FUNDAMENTAL: julgue a recusa EXCLUSIVAMENTE contra o CONTEXTO RECUPERADO, que é a \
única coisa que o agente viu. A RESPOSTA DE REFERÊNCIA é fornecida apenas para que você \
saiba o que seria necessário afirmar — NÃO a use como se o agente a tivesse recebido. \
O fato de a resposta de referência conter a informação NÃO torna a recusa evasiva: o que \
importa é se o CONTEXTO RECUPERADO a continha.

VEREDICTOS:
  2 = RECUSA JUSTIFICADA — o contexto recuperado não continha o necessário para responder \
à pergunta. A recusa foi o comportamento correto.
  1 = RECUSA EVASIVA — o contexto recuperado continha, sim, elementos suficientes para \
responder (ainda que parcialmente), e o agente se esquivou sem razão.
  0 = NÃO É RECUSA — o texto do agente não é de fato uma recusa; ele respondeu à pergunta.

Em caso de dúvida entre 2 e 1, prefira 2: só marque EVASIVA quando o contexto claramente \
sustentasse uma resposta.

INSTRUÇÕES DE FORMATO:
Responda EXCLUSIVAMENTE em JSON válido com o formato exato abaixo. Não inclua texto antes \
ou depois do JSON. Não use blocos de código markdown.

FORMATO:
{
  "raciocinio": "<2-4 frases: o contexto recuperado continha o necessário? o que faltava?>",
  "veredicto": <0, 1 ou 2>
}
"""

TEMPLATE_USUARIO_AUDITOR = """PERGUNTA:
{pergunta}

CONTEXTO RECUPERADO (tudo o que o agente viu):
{contexto}

RESPOSTA DE REFERÊNCIA (o que seria necessário afirmar; o agente NÃO a recebeu):
{ground_truth}

TEXTO PRODUZIDO PELO AGENTE (a recusa a auditar):
{resposta_agente}

A recusa foi justificada pelo contexto recuperado?
"""


def _contexto(chunk_ids: list[str], mapa: dict[str, dict[str, str]]) -> str:
    if not chunk_ids:
        return "(Nenhum documento recuperado para esta pergunta.)"
    return "\n\n".join(
        f"[DOCUMENTO {rank} — {mapa[cid]['source_doc']}]\n{mapa[cid]['text'].strip()}"
        for rank, cid in enumerate(chunk_ids, 1)
    )


def _parse(bruto: str) -> tuple[str, int]:
    """Extrai (raciocinio, veredicto) do JSON do auditor.

    Tolerante a truncamento: se o JSON não fechar (saída cortada no meio do
    raciocinio), tenta recuperar o veredicto por regex. Devolve veredicto = -1
    quando nem isso é possível, para que um item ruim não derrube a run inteira.
    """
    texto = bruto.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        dados = json.loads(texto)
        veredicto = int(dados["veredicto"])
        if veredicto not in (0, 1, 2):
            raise ValueError(f"veredicto fora do domínio: {veredicto}")
        return str(dados["raciocinio"]), veredicto
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        m = re.search(r'"veredicto"\s*:\s*([012])', texto)
        r = re.search(r'"raciocinio"\s*:\s*"(.*?)(?:"|$)', texto, re.DOTALL)
        return (r.group(1) if r else texto[:500]), (int(m.group(1)) if m else -1)


def coletar_recusas(run_dir: Path, agentes: dict[str, str]) -> pd.DataFrame:
    """Todas as respostas que são auto-recusas, por (item, agente)."""
    partes = []
    for agente_id in agentes:
        r = pd.read_parquet(run_dir / "agent_responses" / agente_id / "responses.parquet")
        r = r[r["response_text"].fillna("").str.contains(PADRAO_RECUSA)]
        partes.append(r.assign(agent_id=agente_id))
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--probe", action="store_true", help="audita 1 recusa e sai (sem gravar)")
    p.add_argument("--flush-every", type=int, default=10)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    manifesto = ler_manifesto(run_dir / "manifest.yaml")
    segredos = Segredos()  # type: ignore[call-arg]

    ch = pd.read_parquet(INDICE_RAG)
    mapa = ch.set_index("chunk_id")[["source_doc", "text"]].to_dict("index")
    itens = {i.item_id: i for i in carregar_br_taxqa_r(QUESTOES)}

    recusas = coletar_recusas(run_dir, manifesto.agents)
    print(f"recusas encontradas: {len(recusas)}")
    print(recusas.agent_id.value_counts().to_string())

    # Saída em caminho próprio.
    saida = run_dir / "metrics" / "auditoria_recusas.parquet"
    linhas: list[dict[str, Any]] = []
    feitos: set[tuple[str, str]] = set()
    if saida.exists() and not args.probe:
        anterior = pd.read_parquet(saida)
        linhas = anterior.to_dict("records")
        feitos = set(zip(anterior.item_id, anterior.agent_id, strict=True))
        print(f"retomando: {len(feitos)} já auditadas.")

    cliente = ClienteOpenRouter(
        model=manifesto.judge_model,
        api_key=segredos.openrouter_api_key,
        url=segredos.openrouter_url,
        referer=segredos.openrouter_referer,
        temperature=0.0,
        # 1024 truncava o JSON no meio do raciocinio (o juiz é um modelo de
        # raciocínio: tokens de reasoning consomem o mesmo orçamento).
        max_tokens=4096,
    )

    pendentes = recusas[
        ~recusas.apply(lambda r: (r.item_id, r.agent_id) in feitos, axis=1)
    ]
    if args.probe:
        pendentes = pendentes.head(1)
    print(f"\nauditando {len(pendentes)} recusas com {manifesto.judge_model}...\n", flush=True)

    custo = 0.0
    for n, (_, row) in enumerate(pendentes.iterrows(), 1):
        item = itens[row.item_id]
        usuario = TEMPLATE_USUARIO_AUDITOR.format(
            pergunta=item.pergunta.strip(),
            contexto=_contexto(list(row.retrieved_chunk_ids), mapa),
            ground_truth=item.ground_truth.strip(),
            resposta_agente=str(row.response_text).strip(),
        )
        res = cliente.invoke(usuario, system_prompt=PROMPT_SISTEMA_AUDITOR)
        raciocinio, veredicto = _parse(res.raw_output)
        custo += res.cost_usd
        if veredicto == -1:
            print(f"  ! parse falhou em {row.item_id}/{row.agent_id} — registrado como -1")

        if args.probe:
            print(f"item={row.item_id}  agente={row.agent_id}")
            print(f"veredicto = {veredicto}  (2=justificada, 1=evasiva, 0=nao-e-recusa)")
            print(f"raciocinio: {raciocinio}")
            print(f"\ncusto desta chamada: US$ {res.cost_usd:.4f}")
            print(f"custo estimado do total ({len(recusas)}): US$ {res.cost_usd*len(recusas):.2f}")
            return

        linhas.append(
            {
                "item_id": row.item_id,
                "agent_id": row.agent_id,
                "veredicto_recusa": veredicto,  # 2=justificada, 1=evasiva, 0=nao-e-recusa
                "raciocinio_auditor": raciocinio,
                "cost_usd": res.cost_usd,
            }
        )
        if n % args.flush_every == 0:
            tmp = saida.with_suffix(".parquet.tmp")
            pd.DataFrame(linhas).to_parquet(tmp)
            tmp.rename(saida)
            print(f"  {n}/{len(pendentes)}  (US$ {custo:.2f})", flush=True)

    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(".parquet.tmp")
    pd.DataFrame(linhas).to_parquet(tmp)
    tmp.rename(saida)

    d = pd.DataFrame(linhas)
    print(f"\nsalvo em {saida}  |  custo total: US$ {custo:.2f}\n")
    rotulos = {2: "JUSTIFICADA", 1: "EVASIVA", 0: "nao-e-recusa"}
    print("veredicto por agente:")
    tab = d.pivot_table(
        index="agent_id", columns="veredicto_recusa", values="item_id", aggfunc="count"
    ).fillna(0).astype(int)
    tab.columns = [rotulos.get(c, c) for c in tab.columns]
    print(tab.to_string())


if __name__ == "__main__":
    main()
