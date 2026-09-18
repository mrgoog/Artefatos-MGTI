"""CLI `rag2`: subcomandos do protótipo. M0.1 entrega `ambiente`; milestones seguintes acrescentam os seus."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from ensemble_llm.recuperacao.construtor_corpus import carregar_metadados_indice_rag

from rag2.artefatos import gravar_artefatos
from rag2.config import RAIZ, ConfigRag2
from rag2.constantes import AGENTE_REFERENCIA, B_BOOTSTRAP, CANONICO_R0, TOLERANCIA_CANONICO
from rag2.dados import carregar_itens, ids_amostra_200, itens_com_dispositivo
from rag2.escada import DIR_ESCADA, FABRICAS, checar_canonico, executar_degrau, fabrica_r1, tabela
from rag2.indice import carregar_indice
from rag2.indices import (
    IndiceJaExiste,
    comparar_atingibilidade,
    construir_indice_reparado,
    resolver_indice,
)
from rag2.metricas import Casador, avaliar_itens, resumo, teto_oracular
from rag2.reparo import (
    ALVOS,
    DIR_REPARADO,
    DIR_REPARO,
    FALHA,
    REPARADO,
    SEM_FONTE,
    SEM_GANHO,
    executar_reparo,
)


def cmd_ambiente(args: argparse.Namespace) -> int:
    """Imprime caminhos, confere a presença dos insumos externos e as contagens de referência."""
    cfg = ConfigRag2()
    print(f"origem: {cfg.origem}")
    print(f"dados: {cfg.dir_dados()}")
    print(f"indice_producao: {cfg.dir_indice_producao()}")
    print(f"device: {cfg.device}")
    faltantes = cfg.faltantes()
    if faltantes:
        print("FALTANTES (ajuste .env — ver .env.example):")
        for p in faltantes:
            print(f"  - {p}")
        return 1
    itens = carregar_itens(cfg)
    avaliados = itens_com_dispositivo(itens)
    n_disp = sum(len(a.esperados) for a in avaliados)
    arquivos = {arq for a in avaliados for arq, _ in a.esperados}
    ids200 = ids_amostra_200(cfg)
    com_disp_200 = len(set(ids200) & {a.item_id for a in avaliados})
    print(
        f"itens: {len(itens)} | com dispositivo: {len(avaliados)} | "
        f"dispositivos esperados: {n_disp} | arquivos esperados: {len(arquivos)}"
    )
    print(f"amostra 200 (piloto): {len(ids200)} itens | com dispositivo: {com_disp_200}")
    if not args.sem_indice:
        dir_indice = cfg.dir_indice_producao() if args.indice == "producao" else cfg.dir_indice_reparado()
        if not (dir_indice / "bm25" / "bm25.pkl").exists():
            print(f"índice {args.indice} ausente em {dir_indice}: rode `rag2 construir-indice reparado`")
            return 1
        print(f"indice: {dir_indice}")
        idx = carregar_indice(dir_indice)
        comp = idx.composicao()
        print(
            f"chunks no índice: {len(idx)} | documentos-fonte: {idx.chunks['source_doc'].nunique()} "
            f"| dim: {idx.vetores.shape[1]}"
        )
        print("documentos por tipo: " + ", ".join(f"{k}={v}" for k, v in sorted(comp["documentos"].items())))
        print("chunks por tipo: " + ", ".join(f"{k}={v}" for k, v in sorted(comp["chunks"].items())))
    print("ambiente OK")
    return 0


def cmd_metricas_run(args: argparse.Namespace) -> int:
    """Avalia o contexto servido persistido em runs/<run>/agent_responses/<agente>/responses.parquet."""
    cfg = ConfigRag2()
    itens = itens_com_dispositivo(carregar_itens(cfg))
    idx = carregar_indice(cfg.dir_indice_producao())
    casador = Casador(idx.texto, idx.doc)
    caminho = cfg.dir_run(args.run) / "agent_responses" / args.agente / "responses.parquet"
    resp = pd.read_parquet(caminho, columns=["item_id", "retrieved_chunk_ids"])
    servidos = {r.item_id: [str(c) for c in r.retrieved_chunk_ids] for r in resp.itertuples()}
    df = avaliar_itens(itens, servidos, casador)
    res = resumo(df, b=args.b)
    res["teto_oracular"] = teto_oracular(itens, casador)
    res["fonte"] = str(caminho)
    nome = f"producao_{args.run}"
    saida = gravar_artefatos(Path(args.saida) if args.saida else RAIZ / "runs" / "escada" / nome, df, res, nome)
    print(f"run={args.run}  itens={res['n_itens']}  dispositivos={res['n_dispositivos']}")
    print(
        f"recall_passagem={res['recall_passagem']:.4f}  recall_arquivo={res['recall_arquivo']:.4f}  "
        f"hit_passagem={res['hit_passagem']:.4f}  frac_zero={res['frac_zero']:.4f}  "
        f"chunks/item={res['n_chunks_medio']:.2f}  palavras/item={res['palavras_medias']:.1f}"
    )
    lo, hi = res["ic_recall_passagem"]
    print(f"IC95 recall_passagem=[{lo:.4f}, {hi:.4f}]  (percentil, B={args.b})")
    print(f"teto_oracular={res['teto_oracular']:.4f}")
    canon = cfg.dir_run(args.run) / "metrics" / "recall_passagem.parquet"
    if canon.exists():
        m = df.merge(pd.read_parquet(canon), on="item_id", suffixes=("", "_canon"))
        d_pas = float((m["recall_passagem"] - m["recall_passagem_canon"]).abs().max())
        d_arq = float((m["recall_arquivo"] - m["recall_arquivo_canon"]).abs().max())
        print(
            f"comparação com metrics/recall_passagem.parquet: {len(m)} itens, "
            f"max|Δ recall_passagem|={d_pas:.4f}, max|Δ recall_arquivo|={d_arq:.4f}"
        )
    print(f"artefatos em {saida}")
    return 0


def _faltam_subindices(dir_indice: Path) -> list[Path]:
    """Subíndices normas/CARF ausentes (R3 e R5 precisam dos dois); vazio = ok."""
    from rag2.indices import dir_subindice

    return [dir_subindice(dir_indice, t) for t in ("norma", "acordao_carf")
            if not (dir_subindice(dir_indice, t) / "bm25" / "bm25.pkl").exists()]


def _faltam_respostas(saida: Path, ids: set[str]) -> bool:
    """True se algum agente ainda não tem resposta congelada para todos os ids — só então o llama-swap é exigido (registrar uma vez)."""
    from rag2.constantes import AGENTES

    for aid in AGENTES:
        p = saida / "agent_responses" / aid / "responses.parquet"
        if not p.exists() or len(set(pd.read_parquet(p, columns=["item_id"])["item_id"]) & ids) < len(ids):
            return True
    return False


def _preparar_geracao(cfg: ConfigRag2, degrau: str, saida: Path, itens, comando: str):
    """Guardas e clientes comuns a `variante` e `confirmacao executar`: índice e subíndices do degrau, reformulador
    (só R5, e só se faltam reformulações), OPENROUTER_API_KEY, agentes no llama-swap (só se faltam respostas a gerar).
    Devolve (segredos, clientes_agentes, executor_juiz, openrouter) — `openrouter(max_tokens)` fabrica clientes do
    juiz/auditor/suficiência — ou None após imprimir o motivo (exit 1 no chamador)."""
    from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap, ErroLlamaSwap
    from ensemble_llm.agentes.cliente_openrouter import ClienteOpenRouter
    from ensemble_llm.configuracao import Segredos
    from ensemble_llm.juiz.executor import ExecutorJuiz
    from pydantic import ValidationError

    from rag2.constantes import AGENTES, MAX_TOKENS_AGENTE, MAX_TOKENS_JUIZ, MODELO_JUIZ, SEMENTE
    from rag2.geracao import preparar_cache_juiz

    dir_indice = resolver_indice(cfg, degrau)
    if not (dir_indice / "bm25" / "bm25.pkl").exists():
        print(f"índice ausente em {dir_indice}: rode `rag2 construir-indice reparado`")
        return None
    if degrau in ("R3", "R5") and (faltam := _faltam_subindices(dir_indice)):
        print(f"subíndices ausentes ({', '.join(str(p) for p in faltam)}): rode `rag2 construir-indice reparado --dividir`")
        return None
    if degrau == "R5":                                                # como cmd_degrau: R5 só precisa do reformulador se falta reformular
        from rag2.constantes import MODELO_REFORMULADOR
        from rag2.escada import DIR_ESCADA
        from rag2.reformulacao import CacheReformulacoes, caminho_reformulacoes

        if CacheReformulacoes(caminho_reformulacoes(DIR_ESCADA / "R5")).faltantes([it.pergunta for it in itens]):
            try:
                ClienteLlamaSwap(MODELO_REFORMULADOR).validate_model()
            except ErroLlamaSwap as e:
                print(f"{comando}: degrau R5 precisa do reformulador {MODELO_REFORMULADOR} no llama-swap (http://localhost:8080): {e}")
                return None
    try:
        segredos = Segredos()   # type: ignore[call-arg]
    except ValidationError:
        print(f"{comando} precisa de OPENROUTER_API_KEY no .env (juiz e auditor {MODELO_JUIZ}); rode a partir da raiz do repositório")
        return None
    clientes = {
        aid: ClienteLlamaSwap(modelo, host=segredos.llama_swap_host, temperature=0.0, max_tokens=MAX_TOKENS_AGENTE, seed=SEMENTE)
        for aid, modelo in AGENTES.items()
    }
    if _faltam_respostas(saida, {it.item_id for it in itens}):
        for aid, cli in clientes.items():
            try:
                cli.validate_model()
            except ErroLlamaSwap as e:
                print(f"{comando} precisa do agente {aid} ({cli.model}) no llama-swap ({segredos.llama_swap_host}): {e}")
                return None

    def openrouter(max_tokens: int) -> ClienteOpenRouter:
        return ClienteOpenRouter(
            model=MODELO_JUIZ, api_key=segredos.openrouter_api_key, url=segredos.openrouter_url, referer=segredos.openrouter_referer,
            temperature=0.0, max_tokens=max_tokens, seed=SEMENTE,
        )

    return segredos, clientes, ExecutorJuiz(client=openrouter(MAX_TOKENS_JUIZ), cache=preparar_cache_juiz(cfg)), openrouter


def _nome_gpu(device: str) -> str:
    """Nome da GPU do device (torch) ou '' — procedência do custo.json, nunca uma exigência."""
    try:
        import torch

        return torch.cuda.get_device_name(device) if device.startswith("cuda") and torch.cuda.is_available() else ""
    except Exception:   # noqa: BLE001 — procedência, não guarda
        return ""


def cmd_degrau(args: argparse.Namespace) -> int:
    """Executa um degrau registrado em FABRICAS sobre o índice que lhe cabe e imprime o resumo."""
    cfg = ConfigRag2()
    dir_indice = resolver_indice(cfg, args.nome)
    if not (dir_indice / "bm25" / "bm25.pkl").exists():
        print(f"índice ausente em {dir_indice}: rode `rag2 construir-indice reparado`")
        return 1
    if args.max_por_doc is not None and args.nome != "R1":
        print("--max-por-doc é a variante declarada de R1 (DESIGN §4.1); não se aplica a outro degrau")
        return 2
    if args.nome in ("R3", "R5") and (faltam := _faltam_subindices(dir_indice)):
        print(f"subíndices ausentes ({', '.join(str(p) for p in faltam)}): rode `rag2 construir-indice reparado --dividir`")
        return 1
    if args.nome == "R5":
        from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap, ErroLlamaSwap

        from rag2.constantes import MODELO_REFORMULADOR
        from rag2.reformulacao import CacheReformulacoes, caminho_reformulacoes
        perguntas = [it.pergunta for it in itens_com_dispositivo(carregar_itens(cfg))]
        if args.limite:
            perguntas = perguntas[: args.limite]
        if CacheReformulacoes(caminho_reformulacoes(DIR_ESCADA / "R5")).faltantes(perguntas):
            try:                                                          # só precisa do LLM se falta reformular (registrar uma vez)
                ClienteLlamaSwap(MODELO_REFORMULADOR).validate_model()
            except ErroLlamaSwap as e:
                print(f"R5 precisa do reformulador {MODELO_REFORMULADOR} no llama-swap (http://localhost:8080): {e}")
                return 1
    nome = args.nome if args.max_por_doc is None else f"R1_maxdoc{args.max_por_doc}"
    recuperador = FABRICAS[args.nome](cfg) if args.max_por_doc is None else fabrica_r1(cfg, max_por_doc=args.max_por_doc)
    itens = itens_com_dispositivo(carregar_itens(cfg))
    idx = carregar_indice(dir_indice)
    casador = Casador(idx.texto, idx.doc)
    saida = (RAIZ / "debug-temp" / "escada" / nome) if args.limite else None
    res = executar_degrau(
        nome, recuperador, itens, casador,
        saida=saida, b=args.b, limite=args.limite,
        sobrescrever=args.sobrescrever or bool(args.limite),  # execuções parciais nunca são golden
    )
    print(f"degrau={nome}  itens={res['n_itens']}  dispositivos={res['n_dispositivos']}")
    lo, hi = res["ic_recall_passagem"]
    print(
        f"recall_passagem={res['recall_passagem']:.4f} [{lo:.4f}, {hi:.4f}]  recall_arquivo={res['recall_arquivo']:.4f}  "
        f"hit_passagem={res['hit_passagem']:.4f}  frac_zero={res['frac_zero']:.4f}  "
        f"chunks/item={res['n_chunks_medio']:.2f}  palavras/item={res['palavras_medias']:.1f}"
    )
    print("composição do servido: " + " ".join(f"{k}={v:.3f}" for k, v in res["composicao_servido"].items()))
    print(f"teto_oracular={res['teto_oracular']:.4f}")
    if "delta_vs_golden" in res:
        print(f"golden já registrado; Δ={res['delta_vs_golden']:+.4f} (nada gravado)")
    codigo = 0
    if args.checar_canonico:
        ok, delta = checar_canonico(res)
        print(f"R0 reproduz o canônico {CANONICO_R0}: Δ={delta:+.4f} (tolerância {TOLERANCIA_CANONICO}) — {'OK' if ok else 'FALHOU'}")
        codigo = 0 if ok else 1
    destino = saida or DIR_ESCADA / nome
    metadados = dir_indice / "index_metadata.json"
    if metadados.exists() and not (destino / "indice_metadata.json").exists():
        shutil.copyfile(metadados, destino / "indice_metadata.json")   # procedência do índice ao lado do golden (arch §5)
    print(f"índice: {dir_indice}")
    print(f"artefatos em {destino}")
    return codigo


def cmd_variante(args: argparse.Namespace) -> int:
    """Smoke test de uma variante de prompt (P0…) em 20 itens sobre um degrau da escada; imprime o resumo."""
    from rag2.constantes import MAX_TOKENS_AUDITOR
    from rag2.geracao import DIR_PROMPTS, amostra_smoke, executar_variante
    from rag2.prompts import VARIANTES

    cfg = ConfigRag2()
    itens = amostra_smoke(carregar_itens(cfg))   # 20 itens (mecanismo); a confirmação usa itens_confirmacao
    if args.limite:
        itens = itens[: args.limite]
    saida = (RAIZ / "debug-temp" / "prompts" / args.nome) if args.limite else DIR_PROMPTS / args.nome
    prep = _preparar_geracao(cfg, args.degrau, saida, itens, "variante")
    if prep is None:
        return 1
    _, clientes, executor, openrouter = prep
    res = executar_variante(
        args.nome, args.degrau, FABRICAS[args.degrau](cfg), itens, VARIANTES[args.nome](), clientes, executor, openrouter(MAX_TOKENS_AUDITOR),
        saida=saida, b=args.b, sobrescrever=args.sobrescrever or bool(args.limite),   # execuções parciais nunca são golden
    )
    print(f"variante={args.nome}  degrau={args.degrau}  itens={res['n_itens']}  respostas={res['n_respostas']}")
    _imprimir_geracao(res)
    if "delta_vs_golden" in res:
        print(f"golden já registrado; Δ={res['delta_vs_golden']:+.4f} (nada gravado)")
    print(f"artefatos em {saida}")
    return 0


def _imprimir_geracao(res: dict) -> None:
    """As duas linhas de métricas de geração (adequação por agente; recusa/evasiva/parse/contexto) — comuns a variante e confirmação."""
    lo, hi = res["ic_adequacao_media"]
    print(
        f"adequacao_media={res['adequacao_media']:.4f} [{lo:.4f}, {hi:.4f}]  por agente: "
        + " ".join(f"{aid}={v['adequacao']:.4f}" for aid, v in res["por_agente"].items())
    )
    lr, hr = res["ic_taxa_recusa"]
    le, he = res["ic_taxa_evasiva"]
    print(
        f"recusa={res['taxa_recusa']:.4f} [{lr:.4f}, {hr:.4f}]  evasiva={res['taxa_evasiva']:.4f} [{le:.4f}, {he:.4f}]  "
        f"falha_parse={res['taxa_falha_parse']:.4f}  chunks/item={res['n_chunks_medio']:.2f}  palavras/item={res['palavras_medias']:.1f}"
    )


def cmd_tabela_prompts(args: argparse.Namespace) -> int:
    """Imprime a tabela P0…P3 a partir de runs/prompts/*/resumo.json."""
    from rag2.geracao import tabela_prompts

    print(tabela_prompts())
    return 0


def cmd_confirmacao_executar(args: argparse.Namespace) -> int:
    """`rag2 confirmacao executar`: pipeline novo (variante sobre degrau; padrão P0 sobre R3) nos 200 do piloto + juiz de suficiência + custo; grava runs/confirmacao/novo/."""
    from rag2.confirmacao import DIR_CONFIRMACAO, NOME_NOVO, executar_confirmacao, itens_confirmacao
    from rag2.constantes import MAX_TOKENS_AUDITOR, MAX_TOKENS_SUFICIENCIA
    from rag2.prompts import VARIANTES

    cfg = ConfigRag2()
    itens = itens_confirmacao(cfg, carregar_itens(cfg))
    if args.limite:
        itens = itens[: args.limite]
    saida = (RAIZ / "debug-temp" / "confirmacao" / NOME_NOVO) if args.limite else DIR_CONFIRMACAO / NOME_NOVO
    prep = _preparar_geracao(cfg, args.degrau, saida, itens, "confirmacao")
    if prep is None:
        return 1
    segredos, clientes, executor, openrouter = prep
    ambiente = {
        "device": cfg.device, "gpu": _nome_gpu(cfg.device), "llama_swap_host": segredos.llama_swap_host,
        "indice": str(resolver_indice(cfg, args.degrau)),
    }
    res = executar_confirmacao(
        args.degrau, FABRICAS[args.degrau](cfg), itens, VARIANTES[args.variante](), clientes, executor,
        openrouter(MAX_TOKENS_AUDITOR), openrouter(MAX_TOKENS_SUFICIENCIA),
        saida=saida, b=args.b, sobrescrever=args.sobrescrever or bool(args.limite), ambiente=ambiente,   # execuções parciais nunca são golden
    )
    print(f"confirmacao={NOME_NOVO}  variante={args.variante}  degrau={args.degrau}  itens={res['n_itens']}  respostas={res['n_respostas']}")
    _imprimir_geracao(res)
    s = res["suficiencia"]
    lo, hi = s["ic_frac_suficiente"]
    print(f"suficiencia: suficiente={s['frac_suficiente']:.4f} [{lo:.4f}, {hi:.4f}]  parcial={s['n_parcial']}  insuficiente={s['n_insuficiente']}  falha_parse={s['n_falha_parse']}")
    if "delta_vs_golden" in res:
        print(f"golden já registrado; Δ={res['delta_vs_golden']:+.4f} (suficiência Δ={res['delta_vs_golden_suficiencia']:+.4f}; nada gravado)")
    c = res["custo"]
    print(f"custo desta execução: US$ {c['custo_usd_total']:.2f} | segundos: " + " ".join(f"{k}={v['segundos']:.0f}" for k, v in c["etapas"].items()))
    print(f"artefatos em {saida}")
    return 0


def cmd_confirmacao_comparar(args: argparse.Namespace) -> int:
    """`rag2 confirmacao comparar`: comparação pareada atual (produção) × novo nos 200 + custo; CPU, zero chamadas de modelo; grava runs/confirmacao/{pareado,comparacao}.parquet, comparacao.{json,md}, custo.md, atual/rotulos.parquet."""
    from rag2.comparacao import (
        CABECALHO_COMPARACAO,
        artefatos_producao,
        executar_comparacao,
        linha_comparacao,
    )
    from rag2.confirmacao import DIR_CONFIRMACAO, NOME_NOVO, itens_confirmacao

    cfg = ConfigRag2()
    if not (DIR_CONFIRMACAO / NOME_NOVO / "resumo.json").exists():
        print(f"confirmação '{NOME_NOVO}' não registrada em {DIR_CONFIRMACAO / NOME_NOVO}: rode `rag2 confirmacao executar`")
        return 1
    if faltam := [p for p in artefatos_producao(cfg).values() if not p.exists()]:
        print("artefatos da produção ausentes (RAG2_ORIGEM / RAG2_INDICE_PRODUCAO no .env): " + ", ".join(str(p) for p in faltam))
        return 1
    itens = itens_confirmacao(cfg, carregar_itens(cfg))
    saida = Path(args.saida) if args.saida else DIR_CONFIRMACAO
    res = executar_comparacao(cfg, itens, saida=saida, b=args.b, sobrescrever=args.sobrescrever)
    print(f"comparacao: {res['n_itens']} itens (has_doc_ref={res['n_has_doc_ref']}, com dispositivo={res['n_com_dispositivo']}) | atual={res['atual']} novo={res['novo']} | b0={res['b0']}")
    print(CABECALHO_COMPARACAO)
    for r in res["metricas"]:
        print(linha_comparacao(r))
    a, n = res["custo"]["atual"], res["custo"]["novo"]
    print(
        f"custo: atual US$ {a['custo_usd_total']:.2f} (juiz {a['juiz']['chamadas']} rótulos/US$ {a['juiz']['custo_usd']:.2f}, auditor {a['auditor']['chamadas']}/US$ {a['auditor']['custo_usd']:.2f}, suficiência {a['suficiencia']['chamadas']}/US$ {a['suficiencia']['custo_usd']:.2f})"
        f" | novo US$ {n['custo_usd_total']:.2f} (juiz {n['juiz']['chamadas']}/US$ {n['juiz']['custo_usd']:.2f}, auditor {n['auditor']['chamadas']}/US$ {n['auditor']['custo_usd']:.2f}, suficiência {n['suficiencia']['chamadas']}/US$ {n['suficiencia']['custo_usd']:.2f})"
    )
    rb = res["revisao_b3"]
    lo, hi = rb["ic"]
    veredito = "IC exclui 0 por cima: significativamente melhor — REVISAR a decisão B3" if rb["significativamente_melhor"] else "IC inclui 0: não significativamente melhor; B3 permanece"
    print(f"condição de revisão (B3 waived): Δ adequação média = {rb['delta_adequacao_media']:+.4f} [{lo:+.4f}, {hi:+.4f}] — {veredito}")
    if "delta_vs_golden" in res:
        g = res["delta_vs_golden"]
        print(f"golden já registrado; Δ={g['z_media']:+.4f} (recall Δ={g['recall_passagem']:+.4f}; nada gravado)")
    print(f"artefatos em {saida}")
    return 0


def cmd_confirmacao_braco_api(args: argparse.Namespace) -> int:
    """`rag2 confirmacao braco-api`: contrastes pareados, corte por suficiência e custo do braço de API sobre R3 (exploratório); CPU, zero chamadas; grava runs/confirmacao/novo_api/contrastes.{json,md}."""
    from rag2.braco_api import NOME_API, gravar_contrastes
    from rag2.confirmacao import DIR_CONFIRMACAO

    if faltam := [p for p in (DIR_CONFIRMACAO / "pareado.parquet", DIR_CONFIRMACAO / NOME_API / "por_item.parquet") if not p.exists()]:
        print("artefatos ausentes: " + ", ".join(str(p) for p in faltam))
        return 1
    saida = gravar_contrastes(DIR_CONFIRMACAO, b=args.b)
    print((saida / "contrastes.md").read_text(encoding="utf-8"))
    print(f"artefatos em {saida}")
    return 0


def cmd_construir_indice(args: argparse.Namespace) -> int:
    """Constrói o índice reparado (registrar uma vez); se já existe e não há --sobrescrever, imprime os metadados e sai 0."""
    cfg = ConfigRag2()
    saida = Path(args.saida) if args.saida else cfg.dir_indice_reparado()
    if args.dividir:
        from rag2.indices import construir_subindices

        try:
            metas = construir_subindices(cfg, saida_base=saida, sobrescrever=args.sobrescrever)
        except IndiceJaExiste as e:
            print(f"subíndice já registrado em {e} (--sobrescrever exige decisão registrada)")
            return 0
        for tipo, m in metas.items():
            print(f"subíndice {tipo}: {saida.name}_{m['indice_id'].split('_')[-1]} | chunks={m['n_chunks']} id={m['indice_id']}")
        return 0
    try:
        meta = construir_indice_reparado(cfg, saida=saida, sobrescrever=args.sobrescrever)
    except IndiceJaExiste:
        meta = carregar_metadados_indice_rag(saida)
        print(f"índice já registrado em {saida} (--sobrescrever exige decisão registrada)")
    print(f"índice: {saida}")
    print(
        "chunks: " + " ".join(f"{k}={v}" for k, v in meta["chunks"].items())
        + f" | total={meta['n_chunks']} | dim={meta['dim']}"
    )
    print("sha256 " + " ".join(f"{k}={v}" for k, v in meta["sha256_corpus"].items()))
    print(
        f"embedder={meta['embedder_model']} batch={meta['embedder_batch_size']} device={meta['embedder_device']} "
        f"({meta['gpu']}) {meta['embedder_dtype']} | construído em {meta['construido_em']} | {meta['duracao_s']} s"
    )
    return 0


def cmd_comparar_indices(args: argparse.Namespace) -> int:
    """Teto oracular e atingibilidade par a par, produção (antes) × reparado (depois); só CPU. exit 1 se houver perdas."""
    cfg = ConfigRag2()
    itens = itens_com_dispositivo(carregar_itens(cfg))
    casadores: dict[str, Casador] = {}
    for rotulo, d in (("producao", cfg.dir_indice_producao()), ("reparado", cfg.dir_indice_reparado())):
        if not (d / "bm25" / "bm25.pkl").exists():
            print(f"índice {rotulo} ausente em {d}")
            return 1
        idx = carregar_indice(d)
        casadores[rotulo] = Casador(idx.texto, idx.doc)
        print(f"{rotulo}: chunks={len(idx)} teto_oracular={teto_oracular(itens, casadores[rotulo]):.4f}")
    c = comparar_atingibilidade(itens, casadores["producao"], casadores["reparado"])
    print(
        f"pares={c['n_pares']} atingíveis: {c['atingiveis_antes']} → {c['atingiveis_depois']} "
        f"| ganhos={c['ganhos']} perdas={c['perdas']}"
    )
    return 0 if c["perdas"] == 0 else 1


def cmd_tabela(args: argparse.Namespace) -> int:
    """Imprime a tabela consolidada da escada."""
    print(tabela())
    return 0


def cmd_reparar_corpus(args: argparse.Namespace) -> int:
    """Baixa (ou lê de data/reparo/) os alvos, grava o JSON reparado e o relatório; exit 1 se algum alvo FALHA."""
    cfg = ConfigRag2()
    resultados, sha, avisos = executar_reparo(
        cfg, offline=args.offline, so=args.so, sobrescrever=args.sobrescrever,
        dir_reparo=Path(args.dir_reparo), saida=Path(args.saida),
    )
    for a in avisos:
        print(f"AVISO {a}")
    for r in resultados:
        print(r.linha())
    cont = Counter(r.status for r in resultados)
    print(
        f"reparados={cont[REPARADO]} sem_ganho={cont[SEM_GANHO]} sem_fonte={cont[SEM_FONTE]} falhas={cont[FALHA]} "
        f"| pares esperados que passam a casar: {sum(r.pares_ganhos for r in resultados)} "
        f"| registros=478 | sha256={sha}"
    )
    print(f"JSON: {Path(args.saida) / 'referred_legal_documents_QA_2024_v1.1_reparado.json'}")
    print(f"relatório: {Path(args.saida) / 'relatorio_reparo.md'}")
    return 1 if cont[FALHA] else 0


def cmd_numeros(args: argparse.Namespace) -> int:
    """`rag2 numeros`: gera docs/numeros_canonicos_rag2.md dos artefatos versionados; --tabelas imprime as tabelas do apêndice; --conferir <md> confere lastro (exit 1 se faltar)."""
    from rag2.numeros import (
        ARQUIVO_NUMEROS,
        NOMES_TABELAS,
        carregar,
        cifras,
        conferir,
        ledger,
        tabelas_relatorio,
    )

    a = carregar()
    if args.tabelas:
        for nome, t in tabelas_relatorio(a).items():
            print(f"### {nome}\n{t}\n")
        return 0
    texto = ledger(a)
    if args.conferir:
        r = conferir(Path(args.conferir).read_text(encoding="utf-8"), texto, tabelas_relatorio(a))
        print(
            f"conferido: {args.conferir} — {r['cifras']} cifras com 4 casas, {len(r['sem_lastro'])} sem lastro, "
            f"{len(r['formato_invalido'])} com ponto decimal; "
            f"{len(NOMES_TABELAS) - len(r['tabelas_ausentes'])}/{len(NOMES_TABELAS)} tabelas geradas presentes"
        )
        for x in r["sem_lastro"]:
            print(f"  sem lastro: {x}")
        for x in r["formato_invalido"]:
            print(f"  ponto decimal (use vírgula): {x}")
        for k in r["tabelas_ausentes"]:
            print(f"  tabela ausente ou alterada: {k}")
        return 1 if r["sem_lastro"] or r["formato_invalido"] or r["tabelas_ausentes"] else 0
    saida = Path(args.saida) if args.saida else ARQUIVO_NUMEROS
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(texto, encoding="utf-8")
    print(f"numeros: ledger com {texto.count(chr(10))} linhas e {len(cifras(texto))} cifras com 4 casas → {saida}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    """Monta o parser raiz e os subcomandos; cada subcomando define `fn`."""
    p = argparse.ArgumentParser(prog="rag2", description="Protótipo RAG-2 (BR-TaxQA-R)")
    sub = p.add_subparsers(dest="comando", required=True)
    pa = sub.add_parser("ambiente", help="confere caminhos e contagens de referência")
    pa.add_argument("--sem-indice", action="store_true", help="não carrega o índice (mais rápido)")
    pa.add_argument("--indice", choices=("producao", "reparado"), default="producao", help="qual índice inspecionar")
    pa.set_defaults(fn=cmd_ambiente)
    pm = sub.add_parser("metricas-run", help="recomputa as métricas de uma run da origem a partir de retrieved_chunk_ids")
    pm.add_argument("run", help="ex.: phase2_local_full_715")
    pm.add_argument("--agente", default=AGENTE_REFERENCIA)
    pm.add_argument("--b", type=int, default=B_BOOTSTRAP, help="reamostras do bootstrap")
    pm.add_argument("--saida", default=None, help="diretório de saída (padrão runs/escada/producao_<run>; testes usam tmp)")
    pm.set_defaults(fn=cmd_metricas_run)
    pd_ = sub.add_parser("degrau", help="roda um degrau da escada nos 556 itens e grava runs/escada/<degrau>/")
    pd_.add_argument("nome", choices=sorted(FABRICAS))
    pd_.add_argument("--limite", type=int, default=None, help="só os N primeiros itens (saída vai para debug-temp/)")
    pd_.add_argument("--b", type=int, default=B_BOOTSTRAP)
    pd_.add_argument("--checar-canonico", action="store_true", help="exit 1 se |recall_passagem − 0,2351| > 0,001")
    pd_.add_argument("--sobrescrever", action="store_true", help="substitui um golden já registrado (exige decisão registrada)")
    pd_.add_argument("--max-por-doc", type=int, default=None, help="só R1: variante declarada com no máximo N chunks por documento; grava R1_maxdoc<N>")
    pd_.set_defaults(fn=cmd_degrau)
    pt = sub.add_parser("tabela", help="imprime a tabela R0…R5 a partir de runs/escada/*/resumo.json")
    pt.set_defaults(fn=cmd_tabela)
    pv = sub.add_parser("variante", help="smoke test de uma variante de prompt em 20 itens sobre um degrau; grava runs/prompts/<variante>/")
    pv.add_argument("nome", choices=("P0",), help="variante registrada em rag2.prompts.VARIANTES")
    pv.add_argument("--degrau", default="R3", choices=sorted(FABRICAS), help="degrau da escada que serve o contexto (padrão R3, o melhor de A pelo pré-registro §6)")
    pv.add_argument("--limite", type=int, default=None, help="só os N primeiros dos 20 itens (saída vai para debug-temp/)")
    pv.add_argument("--b", type=int, default=B_BOOTSTRAP)
    pv.add_argument("--sobrescrever", action="store_true", help="substitui um golden já registrado (exige decisão registrada)")
    pv.set_defaults(fn=cmd_variante)
    ptp = sub.add_parser("tabela-prompts", help="imprime a tabela P0…P3 a partir de runs/prompts/*/resumo.json")
    ptp.set_defaults(fn=cmd_tabela_prompts)
    pcf = sub.add_parser("confirmacao", help="confirmação em N=200 (DESIGN §2.3): executar o pipeline novo; comparar com a produção")
    scf = pcf.add_subparsers(dest="acao", required=True)
    pce = scf.add_parser("executar", help="P0 sobre R3 nos 200 do piloto + juiz de suficiência; grava runs/confirmacao/novo/")
    pce.add_argument("--degrau", default="R3", choices=sorted(FABRICAS), help="degrau que serve o contexto (padrão R3, o melhor de A pelo pré-registro §6)")
    pce.add_argument("--variante", default="P0", choices=("P0",), help="variante de prompt (P0 = prompt atual; P1–P3 são condicionais)")
    pce.add_argument("--limite", type=int, default=None, help="só os N primeiros dos 200 (saída vai para debug-temp/)")
    pce.add_argument("--b", type=int, default=B_BOOTSTRAP)
    pce.add_argument("--sobrescrever", action="store_true", help="substitui um golden já registrado (exige decisão registrada)")
    pce.set_defaults(fn=cmd_confirmacao_executar)
    pcc = scf.add_parser("comparar", help="comparação pareada atual (produção) × novo nos 200 + tabela de custo; CPU, zero chamadas; grava runs/confirmacao/{pareado,comparacao}.*, custo.md")
    pcc.add_argument("--b", type=int, default=B_BOOTSTRAP)
    pcc.add_argument("--sobrescrever", action="store_true", help="substitui um golden já registrado (exige decisão registrada)")
    pcc.add_argument("--saida", default=None, help="diretório de saída (padrão runs/confirmacao; testes usam tmp)")
    pcc.set_defaults(fn=cmd_confirmacao_comparar)
    pcb = scf.add_parser("braco-api", help="contrastes pareados, corte por suficiência e custo do braço de API sobre R3 (exploratório, fora do pré-registro); CPU; grava runs/confirmacao/novo_api/contrastes.{json,md}")
    pcb.add_argument("--b", type=int, default=B_BOOTSTRAP, help="reamostras do bootstrap")
    pcb.set_defaults(fn=cmd_confirmacao_braco_api)
    pc = sub.add_parser("construir-indice", help="constrói um índice deste repositório em runs/_shared/rag/ (registrar uma vez)")
    pc.add_argument("nome", choices=("reparado",), help="reparado: corpus normativo reparado + CARF, sem pseudo-docs")
    pc.add_argument("--sobrescrever", action="store_true", help="reconstrói um índice já registrado (exige decisão registrada)")
    pc.add_argument("--saida", default=None, help="diretório do índice (padrão runs/_shared/rag/rag2_reparado; testes usam tmp)")
    pc.add_argument("--dividir", action="store_true", help="divide o índice reparado em subíndices normas/CARF (R3); não reconstrói o índice")
    pc.set_defaults(fn=cmd_construir_indice)
    pci = sub.add_parser("comparar-indices", help="teto oracular e pares atingíveis: produção × reparado (CPU)")
    pci.set_defaults(fn=cmd_comparar_indices)
    pr = sub.add_parser("reparar-corpus", help="baixa das fontes oficiais os documentos normativos truncados e grava o JSON reparado")
    pr.add_argument("--offline", action="store_true", help="não acessa a rede; usa os textos de data/reparo/")
    pr.add_argument("--so", default=None, choices=ALVOS, metavar="FILENAME", help="processa só este alvo (os demais ficam como no original)")
    pr.add_argument("--sobrescrever", action="store_true", help="substitui um texto já registrado em data/reparo/ (exige decisão registrada)")
    pr.add_argument("--dir-reparo", default=str(DIR_REPARO), help="diretório dos textos extraídos (testes usam tmp)")
    pr.add_argument("--saida", default=str(DIR_REPARADO), help="diretório do JSON reparado e do relatório (testes usam tmp)")
    pr.set_defaults(fn=cmd_reparar_corpus)
    pn = sub.add_parser("numeros", help="gera docs/numeros_canonicos_rag2.md dos artefatos versionados (arch §5); --tabelas imprime as tabelas do apêndice; --conferir <md> confere lastro")
    pn.add_argument("--tabelas", action="store_true", help="imprime as cinco tabelas do apêndice (para colar em docs/relatorio_apendice.md)")
    pn.add_argument("--conferir", default=None, metavar="RELATORIO", help="confere que toda cifra com 4 casas do relatório está no ledger e que as tabelas aparecem verbatim; exit 1 se faltar")
    pn.add_argument("--saida", default=None, help="caminho do ledger (padrão docs/numeros_canonicos_rag2.md; testes usam tmp)")
    pn.set_defaults(fn=cmd_numeros)
    return p


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script `rag2`."""
    args = construir_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
