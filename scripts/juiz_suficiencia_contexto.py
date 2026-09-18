"""Segundo eixo de juiz: SUFICIÊNCIA DE CONTEXTO (por item, custo ~US$3).

Produz um eixo ortogonal ao juiz de correção: **suficiência de contexto** — "o contexto
recuperado continha o necessário para responder?" — julgado por item pelo juiz frontier
lendo APENAS o contexto servido (mais a pergunta; o ground truth entra só como referência
do que uma resposta correta exigiria, nunca tratado como se o agente o tivesse recebido).

É por item, não por item×agente: o contexto recuperado é idêntico entre os 3 agentes
(recuperação por pergunta, compartilhada), logo a suficiência é propriedade do par
(pergunta, contexto) — ~715 julgamentos. A ação de cada agente (respondeu/absteve) e o
acerto (`z_agent`) são cruzados na análise (`analise_suficiencia_decisao.py`). A saída vai
só para `runs/<run>/metrics/suficiencia_contexto.parquet`.

Serve qualquer dataset do contrato canônico (`format: brtaxqa_compatible` no YAML), bastando
apontar `--questoes` e `--indice-rag` para os do dataset. Os defaults são os do BR-TaxQA-R.

Uso:
    uv run python scripts/juiz_suficiencia_contexto.py --run phase2_local_full_715 --probe
    uv run python scripts/juiz_suficiencia_contexto.py --run phase2_local_full_715 --limit 20
    uv run python scripts/juiz_suficiencia_contexto.py --run phase2_local_full_715
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

# Prompt de suficiência.

PROMPT_SISTEMA_SUFICIENCIA = """Você é um auditor de sistemas de recuperação (RAG) \
em legislação tributária brasileira.

Um agente de IA recebeu APENAS o "CONTEXTO RECUPERADO" abaixo para responder à PERGUNTA. \
Sua tarefa é decidir se esse CONTEXTO continha o necessário para responder à pergunta de \
forma completa e correta — INDEPENDENTEMENTE do que o agente respondeu.

REGRA FUNDAMENTAL: julgue EXCLUSIVAMENTE o CONTEXTO RECUPERADO. A RESPOSTA DE REFERÊNCIA é \
fornecida apenas para que você saiba o que uma resposta correta precisaria afirmar — NÃO a \
trate como se estivesse no contexto. O fato de a referência conter a informação NÃO torna o \
contexto suficiente: o que importa é se o CONTEXTO RECUPERADO a continha.

VEREDICTOS:
  2 = SUFICIENTE — o contexto recuperado contém tudo o que é necessário para responder à \
pergunta de forma completa.
  1 = PARCIAL — o contexto contém parte do necessário, mas não tudo; uma resposta completa \
não é sustentada apenas pelo contexto.
  0 = INSUFICIENTE — o contexto não contém o necessário para responder; abster-se seria o \
comportamento correto.

INSTRUÇÕES DE FORMATO:
Responda EXCLUSIVAMENTE em JSON válido com o formato exato abaixo. Não inclua texto antes \
ou depois do JSON. Não use blocos de código markdown.

FORMATO:
{
  "raciocinio": "<2-4 frases: o contexto continha o necessário? o que faltava?>",
  "veredicto": <0, 1 ou 2>
}
"""

TEMPLATE_USUARIO_SUFICIENCIA = """PERGUNTA:
{pergunta}

CONTEXTO RECUPERADO (tudo o que o agente recebeu):
{contexto}

RESPOSTA DE REFERÊNCIA (o que uma resposta correta precisaria afirmar; NÃO estava no \
contexto do agente):
{ground_truth}

O CONTEXTO RECUPERADO continha o necessário para responder à pergunta?
"""


def _contexto(chunk_ids: list[str], mapa: dict[str, dict[str, str]]) -> str:
    if not chunk_ids:
        return "(Nenhum documento recuperado para esta pergunta.)"
    partes = []
    for rank, cid in enumerate(chunk_ids, 1):
        ch = mapa.get(cid)
        if ch is None:  # chunk_id ausente do índice — pula sem derrubar a run
            continue
        partes.append(f"[DOCUMENTO {rank} — {ch['source_doc']}]\n{ch['text'].strip()}")
    return "\n\n".join(partes) if partes else "(Nenhum documento recuperado para esta pergunta.)"


def _parse(bruto: str) -> tuple[str, int]:
    """Extrai (raciocinio, veredicto ∈ {0,1,2}) do JSON; -1 em falha total.

    Tolerante a truncamento (o juiz é modelo de raciocínio): se o JSON não fechar,
    recupera o veredicto por regex. -1 mantém um item ruim de derrubar a run.
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


def _gravar(linhas: list[dict[str, Any]], saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(".parquet.tmp")
    pd.DataFrame(linhas).to_parquet(tmp)
    tmp.rename(saida)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--indice-rag", type=Path, default=INDICE_RAG)
    p.add_argument("--questoes", type=Path, default=QUESTOES,
                   help="JSON de questões do dataset (contrato canônico brtaxqa_compatible)")
    p.add_argument("--probe", action="store_true", help="julga 1 item e sai (sem gravar)")
    p.add_argument("--limit", type=int, default=None, help="julga só N itens (verificação)")
    p.add_argument("--flush-every", type=int, default=10)
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    manifesto = ler_manifesto(run_dir / "manifest.yaml")
    segredos = Segredos()  # type: ignore[call-arg]

    ch = pd.read_parquet(args.indice_rag)
    mapa = ch.set_index("chunk_id")[["source_doc", "text"]].to_dict("index")
    itens = {i.item_id: i for i in carregar_br_taxqa_r(args.questoes)}

    # Contexto é compartilhado entre os 3 agentes -> ler UM agente basta (per-item).
    agente0 = list(manifesto.agents)[0]
    resp = pd.read_parquet(run_dir / "agent_responses" / agente0 / "responses.parquet")
    ctx_por_item = dict(zip(resp.item_id, resp.retrieved_chunk_ids, strict=True))
    print(f"itens no run: {len(ctx_por_item)}  (contexto lido de '{agente0}')")

    saida = run_dir / "metrics" / "suficiencia_contexto.parquet"
    linhas: list[dict[str, Any]] = []
    feitos: set[str] = set()
    if saida.exists() and not args.probe:
        anterior = pd.read_parquet(saida)
        linhas = anterior.to_dict("records")
        feitos = set(anterior.item_id)
        print(f"retomando: {len(feitos)} itens já julgados.")

    cliente = ClienteOpenRouter(
        model=manifesto.judge_model,
        api_key=segredos.openrouter_api_key,
        url=segredos.openrouter_url,
        referer=segredos.openrouter_referer,
        temperature=0.0,
        max_tokens=4096,  # 1024 trunca o JSON de raciocínio (juiz é modelo de raciocínio)
    )

    pendentes = [iid for iid in ctx_por_item if iid not in feitos]
    if args.probe:
        pendentes = pendentes[:1]
    elif args.limit is not None:
        pendentes = pendentes[: args.limit]
    print(f"\njulgando suficiência de {len(pendentes)} itens com {manifesto.judge_model}...\n",
          flush=True)

    custo = 0.0
    for n, iid in enumerate(pendentes, 1):
        item = itens[iid]
        chunk_ids = list(ctx_por_item[iid])
        contexto_vazio = len(chunk_ids) == 0

        if contexto_vazio:
            # Sem contexto recuperado é insuficiente por definição — sem chamada ao LLM.
            veredicto, custo_item = 0, 0.0
            raciocinio = "(contexto vazio — nenhum documento recuperado)"
        else:
            usuario = TEMPLATE_USUARIO_SUFICIENCIA.format(
                pergunta=item.pergunta.strip(),
                contexto=_contexto(chunk_ids, mapa),
                ground_truth=item.ground_truth.strip(),
            )
            res = cliente.invoke(usuario, system_prompt=PROMPT_SISTEMA_SUFICIENCIA)
            raciocinio, veredicto = _parse(res.raw_output)
            custo_item = res.cost_usd
            custo += custo_item
            if veredicto == -1:
                print(f"  ! parse falhou em {iid} — registrado como -1")

        if args.probe:
            print(f"item={iid}  has_doc_ref={item.has_doc_ref}  n_chunks={len(chunk_ids)}")
            print(f"veredicto = {veredicto}  (2=suficiente, 1=parcial, 0=insuficiente)")
            print(f"raciocinio: {raciocinio}")
            print(f"\ncusto desta chamada: US$ {custo_item:.4f}")
            print(f"custo estimado do total ({len(ctx_por_item)}): "
                  f"US$ {custo_item*len(ctx_por_item):.2f}")
            return

        linhas.append(
            {
                "item_id": iid,
                "has_doc_ref": bool(item.has_doc_ref),
                "n_chunks": len(chunk_ids),
                "contexto_vazio": contexto_vazio,
                "veredicto_suficiencia": veredicto,  # 2=suf, 1=parcial, 0=insuf, -1=falha
                "raciocinio_juiz": raciocinio,
                "cost_usd": custo_item,
                "judge_model": manifesto.judge_model,
            }
        )
        if n % args.flush_every == 0:
            _gravar(linhas, saida)
            print(f"  {n}/{len(pendentes)}  (US$ {custo:.2f})", flush=True)

    _gravar(linhas, saida)

    d = pd.DataFrame(linhas)
    print(f"\nsalvo em {saida}  |  custo total: US$ {custo:.2f}\n")
    rotulos = {2: "SUFICIENTE", 1: "PARCIAL", 0: "INSUFICIENTE", -1: "falha-parse"}
    vc = d.veredicto_suficiencia.value_counts().sort_index()
    print("distribuição da suficiência:")
    for v, cont in vc.items():
        print(f"  {rotulos.get(v, v):14s} {cont:4d}  ({cont / len(d):.1%})")
    print(f"\ncontexto_vazio (auto-rotulado 0, sem LLM): {int(d.contexto_vazio.sum())}")


if __name__ == "__main__":
    main()
