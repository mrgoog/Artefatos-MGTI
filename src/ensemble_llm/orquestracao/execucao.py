"""Orquestração end-to-end da Fase 1."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ensemble_llm.agentes.cliente_llama_swap import ClienteLlamaSwap
from ensemble_llm.agentes.cliente_ollama import ClienteOllama
from ensemble_llm.agentes.cliente_openrouter import ClienteOpenRouter
from ensemble_llm.agentes.contratos import ClienteLLM
from ensemble_llm.agentes.estruturado import invocar_estruturado
from ensemble_llm.agentes.prompts_agentes import DICA_SCHEMA_AGENTE, construir_prompt_agente
from ensemble_llm.avaliacao.bootstrap_estratificado import ic_bootstrap_estratificado
from ensemble_llm.avaliacao.diversidade import matriz_concordancia_erros
from ensemble_llm.avaliacao.metricas import (
    cobertura_empirica,
    dados_diagrama_confiabilidade,
    erro_calibracao_esperado,
    estatisticas_estabilidade_limiar,
    estatisticas_estabilidade_pesos,
    risco_seletivo,
)
from ensemble_llm.avaliacao.retrieval_recall import calcular_retrieval_recall
from ensemble_llm.calibracao.isotonica import ajustar_calibrador_isotonico, aplicar_calibracao
from ensemble_llm.calibracao.pesos import calcular_pesos_agentes
from ensemble_llm.configuracao import ConfigExperimento, Segredos, carregar_config_experimento
from ensemble_llm.dados.conjunto_dados import (
    carregar_br_taxqa_r,
    carregar_documentos_legais,
    construir_corpus_de_itens,
)
from ensemble_llm.dados.particionamento import criar_particoes, validar_cobertura_completa
from ensemble_llm.decisao import baselines as bl
from ensemble_llm.decisao.pontuacao import decidir
from ensemble_llm.decisao.risco_seletivo import curva_risco_cobertura, encontrar_tau_estrela
from ensemble_llm.esquemas import ItemDataset, RespostaAgente
from ensemble_llm.juiz.cache_rotulos import CacheRotulosJuiz
from ensemble_llm.juiz.executor import ExecutorJuiz
from ensemble_llm.orquestracao.validacao import validar_config_experimento
from ensemble_llm.persistencia.caminhos import (
    caminho_curva_risco_cobertura,
    caminho_decisoes_teste_fold,
    caminho_limiar_fold,
    caminho_manifesto,
    caminho_pesos_fold,
    caminho_pontos_tau_estrela,
    caminho_respostas_agente,
    caminho_teste_consolidado,
    diretorio_cache_juiz,
    diretorio_execucao,
    diretorio_metricas,
    diretorio_rag,
    diretorio_reliability,
)
from ensemble_llm.persistencia.manifesto import ManifestoExecucao, agora_iso, escrever_manifesto
from ensemble_llm.persistencia.tabelas import escrever_parquet
from ensemble_llm.recuperacao.construtor_corpus import (
    carregar_metadados_indice_rag,
    carregar_recuperador_hibrido,
    construir_recuperador_hibrido,
    salvar_metadados_indice_rag,
)
from ensemble_llm.recuperacao.vetorizacao import VetorizadorSentenceTransformer

logger = logging.getLogger(__name__)
K_RELIABILITY = 1


def _adequacao_util(dados: pd.DataFrame) -> float:
    """Adequação útil: soma dos acertos respondidos dividida por N total."""
    if len(dados) == 0:
        return float("nan")
    respondidos = dados.loc[~dados["abstained"], "z_system"]
    return float(pd.to_numeric(respondidos, errors="coerce").fillna(0.0).sum() / len(dados))


def _precisao_abstencao_bsolo(dados: pd.DataFrame) -> float:
    """Precisão da abstenção no B_solo: fração de abstidos com `z_agent=0`."""
    abstidos = dados[dados["abstained"]]
    if len(abstidos) == 0:
        return float("nan")
    return float((1 - abstidos["z_agent"].astype(int)).mean())


def _taxa_base_inadequacao_bsolo(dados: pd.DataFrame) -> float:
    """Taxa-base de inadequação do agente solo no conjunto avaliado."""
    if len(dados) == 0:
        return float("nan")
    return float((1 - dados["z_agent"].astype(int)).mean())


def _lift_abstencao_pp_bsolo(dados: pd.DataFrame) -> float:
    """Lift em pontos percentuais da precisão da abstenção vs taxa-base."""
    precisao = _precisao_abstencao_bsolo(dados)
    if not np.isfinite(precisao):
        return float("nan")
    return float(precisao - _taxa_base_inadequacao_bsolo(dados))


def _n_abstencoes(dados: pd.DataFrame) -> float:
    """Número de abstenções como escalar para resumo/IC bootstrap."""
    return float(dados["abstained"].sum())


def _auroc_bsolo(dados: pd.DataFrame) -> float:
    """AUROC de `eta_solo` como preditor de inadequação do agente."""
    y_true = 1 - dados["z_agent"].astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, dados["eta_solo"].to_numpy()))


def _bootstrap_metric_or_none(
    dados: pd.DataFrame,
    fn_estatistica: Any,
    *,
    config: ConfigExperimento,
) -> dict[str, Any] | None:
    """Executa bootstrap estratificado e retorna `None` quando a métrica é inviável."""
    try:
        return ic_bootstrap_estratificado(
            dados,
            fn_estatistica,
            coluna_estratificacao="has_doc_ref",
            n_reamostras=config.bootstrap.n_resamples,
            nivel_confianca=config.bootstrap.confidence_level,
            metodo=config.bootstrap.method,
            seed=config.bootstrap.seed,
        ).model_dump()
    except (RuntimeError, ValueError):
        return None


@dataclass(frozen=True)
class ArtefatosExecucao:
    """Caminhos dos artefatos principais produzidos por executar_experimento."""

    diretorio_execucao: Path
    consolidated_tests: dict[float, Path]
    metrics_json: Path
    manifest: Path


def _resolver_diretorio_rag_compartilhado(config: ConfigExperimento) -> Path | None:
    """Retorna o diretório compartilhado do RAG quando configurado."""
    if config.retrieval.persisted_index_dir is None:
        return None
    return Path(config.retrieval.persisted_index_dir)


def _resolver_arquivos_dataset(
    config: ConfigExperimento,
    *,
    questions_json_path: str | Path | None,
    corpus_extra_paths: list[str | Path] | None,
) -> tuple[Path, list[Path] | None]:
    """Resolve arquivos físicos do dataset com precedência para overrides da CLI."""
    caminho_questoes = (
        Path(questions_json_path)
        if questions_json_path is not None
        else config.dataset.resolver_questions_path()
    )

    caminhos_corpus = (
        [Path(path) for path in corpus_extra_paths]
        if corpus_extra_paths is not None
        else config.dataset.resolver_corpus_paths()
    )
    return caminho_questoes, caminhos_corpus or None


def _metadados_indice_rag(
    config: ConfigExperimento,
    questions_json: Path,
    corpus_paths: list[Path] | None,
) -> dict[str, Any]:
    """Monta metadados descritivos do índice RAG para auditoria e validação."""
    return {
        "dataset_id": config.dataset.dataset_id,
        "dataset_format": config.dataset.format,
        "questions_file": str(questions_json),
        "corpus_files": [str(path) for path in (corpus_paths or [])],
        "embedder_model": config.retrieval.embedder_model,
        "chunk_max_tokens": config.retrieval.chunk_max_tokens,
        "chunk_overlap_tokens": config.retrieval.chunk_overlap_tokens,
    }


def _validar_dataset_do_indice_rag(config: ConfigExperimento, rag_root: Path) -> None:
    """Valida que o índice RAG carregado pertence ao dataset esperado pelo YAML."""
    metadados = carregar_metadados_indice_rag(rag_root)
    if metadados is None:
        logger.warning(
            "Índice RAG em %s sem metadados. Seguindo por compatibilidade; "
            "rebuild recomendado para registrar dataset_id.",
            rag_root,
        )
        return

    dataset_id_indice = metadados.get("dataset_id")
    if dataset_id_indice != config.dataset.dataset_id:
        raise ValueError(
            "Índice RAG incompatível com o dataset configurado: "
            f"dataset_id do índice={dataset_id_indice!r}, "
            f"dataset_id do YAML={config.dataset.dataset_id!r}, "
            f"diretório={rag_root}."
        )


def _git_sha() -> str:
    """Obtém git SHA curto quando disponível."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _construir_clientes(
    config: ConfigExperimento,
    secrets: Segredos,
) -> tuple[dict[str, ClienteLLM], ClienteOpenRouter]:
    """Cria clientes dos agentes e do juiz.

    Agentes podem usar backend ``"openrouter"``, ``"ollama"`` ou ``"llama-swap"``.
    O juiz permanece sempre OpenRouter (modelo frontier).
    """
    agents: dict[str, ClienteLLM] = {}
    for agent in config.agents:
        if agent.backend == "openrouter":
            agents[agent.agent_id] = ClienteOpenRouter(
                model=agent.model,
                api_key=secrets.openrouter_api_key,
                url=secrets.openrouter_url,
                referer=secrets.openrouter_referer,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                seed=agent.seed,
                reasoning_effort=agent.reasoning_effort,
            )
        elif agent.backend == "ollama":
            host = agent.host or secrets.ollama_host_gpu0
            agents[agent.agent_id] = ClienteOllama(
                model=agent.model,
                host=host,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                seed=agent.seed,
            )
        elif agent.backend == "llama-swap":
            host = agent.host or secrets.llama_swap_host
            agents[agent.agent_id] = ClienteLlamaSwap(
                model=agent.model,
                host=host,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                seed=agent.seed,
            )
        else:
            raise NotImplementedError(f"Backend {agent.backend!r} não suportado.")
    if config.judge.backend != "openrouter":
        raise NotImplementedError("Juiz precisa ser openrouter.")
    judge = ClienteOpenRouter(
        model=config.judge.model,
        api_key=secrets.openrouter_api_key,
        url=secrets.openrouter_url,
        referer=secrets.openrouter_referer,
        temperature=config.judge.temperature,
        max_tokens=config.judge.max_tokens,
        seed=config.judge.seed,
    )
    return agents, judge


def _escrever_parquet_atomico(df: pd.DataFrame, path: Path) -> None:
    """Escreve parquet via tmp + rename para atomicidade no POSIX.

    Se o processo for interrompido durante o write, o arquivo final permanece
    intacto (com o conteúdo do flush anterior). Sem isso, um SIGTERM no meio
    de `df.to_parquet` poderia deixar o parquet truncado, inutilizando todo o
    cache do agente.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _executar_agentes(
    items: list[ItemDataset],
    agent_clients: dict[str, ClienteLLM],
    retriever: Any,
    base_dir: Path,
    experiment_id: str,
    *,
    flush_every: int = 10,
) -> dict[str, pd.DataFrame]:
    """Executa todos os agentes em todos os itens e persiste respostas.

    Resumability em dois níveis:

    1. **Skip de agente completo.** Se `responses.parquet` do agente já cobre
       todos os `item_ids` desta run, pula a re-execução desse agente inteiro.
    2. **Retomada de agente parcial.** Se o parquet existe mas só cobre parte
       dos itens, processa apenas os faltantes e anexa ao parquet existente.

    Persistência incremental: a cada `flush_every` itens NOVOS processados,
    grava o parquet atomicamente (tmp + rename). Em crash, o pior caso é
    perder os últimos < flush_every itens, não todo o agente.

    Args:
        items: itens a processar (item_id estável é a chave de retomada).
        agent_clients: mapping agent_id → cliente LLM.
        retriever: módulo RAG.
        base_dir: diretório raiz das runs.
        experiment_id: id do experimento.
        flush_every: persiste parquet a cada N itens NOVOS (default 10).
            Para Ollama local lento, 5–10 é razoável: overhead I/O desprezível
            comparado à inferência. Para API rápida, 20–50 reduz I/O sem
            arriscar muito.
    """
    out: dict[str, pd.DataFrame] = {}
    item_ids_alvo_set = {it.item_id for it in items}
    logger.info(
        "Iniciando execução dos agentes: %d agente(s), %d item(ns), flush_every=%d.",
        len(agent_clients),
        len(items),
        flush_every,
    )

    for agent_id, client in agent_clients.items():
        t_agente = time.perf_counter()
        path = caminho_respostas_agente(base_dir, experiment_id, agent_id)
        logger.info("Agente %s: iniciando processamento (cache=%s).", agent_id, path)

        if path.exists():
            try:
                df_existente = pd.read_parquet(path)
            except Exception as e:
                raise RuntimeError(
                    f"Falha ao ler parquet existente do agente {agent_id!r} em {path}. "
                    f"Inspecione ou remova manualmente antes de re-executar."
                ) from e
            cobertos = set(df_existente["item_id"])
            pendentes_ids = item_ids_alvo_set - cobertos
            if not pendentes_ids:
                logger.info(
                    "Agente %s: parquet completo (%d itens cobrem %d alvos), pulando.",
                    agent_id,
                    len(cobertos & item_ids_alvo_set),
                    len(item_ids_alvo_set),
                )
                out[agent_id] = df_existente[df_existente["item_id"].isin(item_ids_alvo_set)]
                continue
            logger.info(
                "Agente %s: retomando — %d/%d cobertos, %d pendentes.",
                agent_id,
                len(cobertos & item_ids_alvo_set),
                len(item_ids_alvo_set),
                len(pendentes_ids),
            )
            rows: list[dict[str, Any]] = df_existente.to_dict("records")
        else:
            pendentes_ids = item_ids_alvo_set
            rows = []

        pendentes = [it for it in items if it.item_id in pendentes_ids]
        logger.info(
            "Agente %s: %d item(ns) pendente(s) nesta execução.",
            agent_id,
            len(pendentes),
        )
        for i, item in enumerate(pendentes, start=1):
            t_item = time.perf_counter()
            logger.info(
                "Agente %s: item %d/%d (%s) — recuperando contexto RAG.",
                agent_id,
                i,
                len(pendentes),
                item.item_id,
            )
            t_rag = time.perf_counter()
            chunks = retriever.recuperar(item.pergunta)
            duracao_rag = time.perf_counter() - t_rag
            logger.info(
                "Agente %s: item %s — RAG concluído em %.2fs com %d chunk(s).",
                agent_id,
                item.item_id,
                duracao_rag,
                len(chunks),
            )
            system_prompt, user_prompt = construir_prompt_agente(item.pergunta, chunks)
            t_llm = time.perf_counter()
            result = invocar_estruturado(
                client,
                user_prompt=user_prompt,
                schema=RespostaAgente,
                system_prompt=system_prompt,
                schema_hint=DICA_SCHEMA_AGENTE,
                k_max=3,
            )
            duracao_llm = time.perf_counter() - t_llm
            parsed = result.parsed if isinstance(result.parsed, RespostaAgente) else None
            rows.append(
                {
                    "item_id": item.item_id,
                    "agent_id": agent_id,
                    "prompt_hash": result.prompt_hash,
                    "response_text": parsed.resposta if parsed else None,
                    "confidence_raw": parsed.confianca if parsed else None,
                    "valid_json": result.valid_json,
                    "n_retries": result.n_retries,
                    "raw_output": result.raw_output,
                    "cost_usd": result.cost_usd,
                    "retrieved_chunk_ids": [c.chunk_id for c in chunks],
                    "timestamp": agora_iso(),
                }
            )
            logger.info(
                "Agente %s: item %d/%d (%s) concluído em %.2fs total; LLM=%.2fs; json_ok=%s; retries=%d; cost=$%.6f",
                agent_id,
                i,
                len(pendentes),
                item.item_id,
                time.perf_counter() - t_item,
                duracao_llm,
                result.valid_json,
                result.n_retries,
                result.cost_usd,
            )
            if i % flush_every == 0:
                _escrever_parquet_atomico(pd.DataFrame(rows), path)
                logger.info(
                    "Agente %s: flush em %d/%d pendentes (parquet: %d linhas).",
                    agent_id,
                    i,
                    len(pendentes),
                    len(rows),
                )

        df_final = pd.DataFrame(rows)
        _escrever_parquet_atomico(df_final, path)
        logger.info(
            "Agente %s: finalizado em %.2fs; parquet final com %d linha(s).",
            agent_id,
            time.perf_counter() - t_agente,
            len(df_final),
        )
        out[agent_id] = df_final[df_final["item_id"].isin(item_ids_alvo_set)]
    return out


def _executar_rotulos_juiz(
    items: list[ItemDataset],
    agent_tables: dict[str, pd.DataFrame],
    judge_runner: ExecutorJuiz,
) -> pd.DataFrame:
    """Rotula adequação de cada resposta de agente (ou z=0 em JSON inválido)."""
    item_map = {it.item_id: it for it in items}
    rows: list[dict[str, Any]] = []
    total = sum(len(df) for df in agent_tables.values())
    logger.info("Iniciando rotulagem do juiz para %d resposta(s) de agentes.", total)
    idx = 0
    for agent_id, df in agent_tables.items():
        for row in df.itertuples(index=False):
            idx += 1
            t0 = time.perf_counter()
            if not bool(row.valid_json) or not row.response_text:
                logger.info(
                    "Juiz: %d/%d item=%s agente=%s — pulado, JSON inválido do agente.",
                    idx,
                    total,
                    row.item_id,
                    agent_id,
                )
                rows.append({"item_id": row.item_id, "agent_id": agent_id, "z_agent": 0})
                continue
            logger.info(
                "Juiz: %d/%d item=%s agente=%s — iniciando avaliação.",
                idx,
                total,
                row.item_id,
                agent_id,
            )
            label = judge_runner.julgar(
                item=item_map[row.item_id],
                resposta_candidata=str(row.response_text),
                fonte_resposta=agent_id,
            )
            logger.info(
                "Juiz: %d/%d item=%s agente=%s concluído em %.2fs; z=%d; cost=$%.6f",
                idx,
                total,
                row.item_id,
                agent_id,
                time.perf_counter() - t0,
                int(label.z),
                label.cost_usd,
            )
            rows.append({"item_id": row.item_id, "agent_id": agent_id, "z_agent": int(label.z)})
    judge_runner.cache.flush()
    logger.info("Juiz: cache flush concluído; %d rótulo(s) materializado(s).", len(rows))
    return pd.DataFrame(rows)


def executar_experimento(
    config: ConfigExperimento,
    secrets: Segredos,
    *,
    items: list[ItemDataset] | None = None,
    questions_json_path: str | Path | None = None,
    corpus_extra_paths: list[str | Path] | None = None,
    embedder: Any | None = None,
    validate_models: bool = True,
) -> ArtefatosExecucao:
    """Executa pipeline completo e persiste artefatos.

    corpus_extra_paths: caminhos adicionais de JSON para o índice RAG
    (ex: acordaos_CARF_2023.json, referred_legal_documents_QA_2024_v1.1.json).
    Quando None, o corpus é composto apenas dos snippets embutidos nas questões.
    """
    logging.basicConfig(level=logging.INFO)
    t_execucao = time.perf_counter()
    logger.info("Execução iniciada: experiment_id=%s", config.experiment_id)

    if validate_models:
        t_validacao = time.perf_counter()
        validar_config_experimento(config, secrets)
        logger.info(
            "Validação de modelos concluída em %.2fs.",
            time.perf_counter() - t_validacao,
        )

    questions_json_resolvido, corpus_extra_paths_resolvidos = _resolver_arquivos_dataset(
        config,
        questions_json_path=questions_json_path,
        corpus_extra_paths=corpus_extra_paths,
    )

    if items is None:
        t_dataset = time.perf_counter()
        items = carregar_br_taxqa_r(
            questions_json_resolvido,
            tamanho_amostra=config.sample_size,
            semente_amostra=secrets.experiment_seed,
        )
        logger.info(
            "Dataset carregado em %.2fs: %d item(ns).",
            time.perf_counter() - t_dataset,
            len(items),
        )
    if not items:
        raise ValueError("Nenhum item no dataset.")

    base_dir = secrets.runs_base_dir
    root = diretorio_execucao(base_dir, config.experiment_id)
    root.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        embedder = VetorizadorSentenceTransformer(
            nome_modelo=config.retrieval.embedder_model,
            tamanho_lote=config.retrieval.embedder_batch_size,
            dispositivo=config.retrieval.embedder_device,
        )

    rag_shared_root = _resolver_diretorio_rag_compartilhado(config)
    rag_root = rag_shared_root or diretorio_rag(base_dir, config.experiment_id)
    if rag_shared_root is not None and not rag_root.exists():
        raise ValueError(
            f"Índice RAG compartilhado não encontrado em {rag_root}. "
            "Gere-o antes com o script de build offline."
        )

    if rag_root.exists():
        _validar_dataset_do_indice_rag(config, rag_root)
        t_rag = time.perf_counter()
        retriever = carregar_recuperador_hibrido(
            rag_root,
            vetorizador=embedder,
            top_k_denso=config.retrieval.dense_top_k,
            top_k_bm25=config.retrieval.bm25_top_k,
        )
        logger.info(
            "RAG carregado de disco em %.2fs: %s%s",
            time.perf_counter() - t_rag,
            rag_root,
            " (compartilhado)" if rag_shared_root is not None else "",
        )
    else:
        t_rag = time.perf_counter()
        documents = construir_corpus_de_itens(items)
        if corpus_extra_paths_resolvidos:
            extra = carregar_documentos_legais(corpus_extra_paths_resolvidos)
            logger.info(
                "Corpus extra: %d documentos de %d arquivo(s) adicionados ao índice RAG.",
                len(extra),
                len(corpus_extra_paths_resolvidos),
            )
            documents.update(extra)
        logger.info("Corpus RAG total: %d documentos.", len(documents))
        retriever = construir_recuperador_hibrido(
            documents,
            vetorizador=embedder,
            top_k_denso=config.retrieval.dense_top_k,
            top_k_bm25=config.retrieval.bm25_top_k,
            diretorio_persistencia=rag_root,
        )
        salvar_metadados_indice_rag(
            rag_root,
            _metadados_indice_rag(
                config,
                questions_json_resolvido,
                corpus_extra_paths_resolvidos,
            ),
        )
        logger.info(
            "RAG construído e persistido em %.2fs: %s",
            time.perf_counter() - t_rag,
            rag_root,
        )

    agent_clients, judge_client = _construir_clientes(config, secrets)
    t_agentes = time.perf_counter()
    agent_tables = _executar_agentes(items, agent_clients, retriever, base_dir, config.experiment_id)
    logger.info("Fase de agentes concluída em %.2fs.", time.perf_counter() - t_agentes)

    judge_cache = CacheRotulosJuiz(
        cache_dir=diretorio_cache_juiz(base_dir, config.experiment_id),
        judge_model=judge_client.model,
    )
    judge_runner = ExecutorJuiz(client=judge_client, cache=judge_cache)
    t_juiz = time.perf_counter()
    z_agent_df = _executar_rotulos_juiz(items, agent_tables, judge_runner)
    logger.info("Fase do juiz concluída em %.2fs.", time.perf_counter() - t_juiz)

    item_ids = [it.item_id for it in items]
    strat_labels = [1 if it.has_doc_ref else 0 for it in items]
    partitions, fold_perm = criar_particoes(
        item_ids=item_ids,
        rotulos_estratificacao=strat_labels,
        n_splits=config.partition.n_splits,
        n_folds_val=config.partition.n_val_folds,
        seed=secrets.experiment_seed,
    )
    validar_cobertura_completa(partitions)
    logger.info("Particionamento concluído: %d fold(s).", len(partitions))

    escrever_manifesto(
        caminho_manifesto(base_dir, config.experiment_id),
        ManifestoExecucao(
            experiment_id=config.experiment_id,
            created_at=agora_iso(),
            git_sha=_git_sha(),
            root_seed=secrets.experiment_seed,
            fold_permutation=fold_perm,
            agents={a.agent_id: a.model for a in config.agents},
            judge_model=config.judge.model,
        ),
    )

    by_agent = {aid: df.set_index("item_id") for aid, df in agent_tables.items()}
    z_lookup = {(r.item_id, r.agent_id): int(r.z_agent) for r in z_agent_df.itertuples(index=False)}
    has_doc_ref_lookup = {it.item_id: it.has_doc_ref for it in items}
    
    # retrieval_recall_lookup = _calcular_retrieval_recall(items, agent_tables)

    retrieval_recall_lookup = calcular_retrieval_recall(
        items,
        agent_tables,
        rag_root=Path(config.retrieval.persisted_index_dir) if config.retrieval.persisted_index_dir else None,
        questions_json=questions_json_resolvido,
    )

    artefatos_full: dict[float, list[dict[str, Any]]] = {c: [] for c in config.decision.c_min_grid}
    artefatos_bsolo: dict[tuple[str, float], list[dict[str, Any]]] = {
        (aid, c): [] for aid in by_agent for c in config.decision.c_min_grid
    }
    consolidados_full: dict[float, pd.DataFrame] = {}
    rows_b1: list[dict[str, Any]] = []
    rows_b2: list[dict[str, Any]] = []
    rows_b3: list[dict[str, Any]] = []
    for p in partitions:
        t_fold = time.perf_counter()
        logger.info(
            "Fold k=%02d: fit=%d, val=%d, test=%d.",
            p.k,
            len(p.fit_ids),
            len(p.val_ids),
            len(p.test_ids),
        )
        z_fit: dict[str, list[int]] = {}
        for aid in by_agent:
            z_fit[aid] = [z_lookup[(iid, aid)] for iid in p.fit_ids]
        weights_art = calcular_pesos_agentes(z_fit)

        calibrators: dict[str, Any] = {}
        for aid, table in by_agent.items():
            fit_conf = []
            fit_z = []
            for iid in p.fit_ids:
                raw_conf = table.loc[iid, "confidence_raw"]
                conf = float(raw_conf) if pd.notna(raw_conf) else 0.0
                fit_conf.append(conf)
                fit_z.append(z_lookup[(iid, aid)])
            calibrators[aid] = ajustar_calibrador_isotonico(fit_conf, fit_z)
        logger.info("Fold k=%02d: pesos e calibradores ajustados.", p.k)

        rows_b1.extend(
            bl.b1_agente_unico(
                pesos_art=weights_art,
                by_agent=by_agent,
                test_ids=list(p.test_ids),
                z_lookup=z_lookup,
                has_doc_ref_lookup=has_doc_ref_lookup,
                retrieval_recall_lookup=retrieval_recall_lookup,
                fold_k=p.k,
            )
        )
        rows_b2.extend(
            bl.b2_ponderado_sem_calibracao(
                pesos_art=weights_art,
                by_agent=by_agent,
                test_ids=list(p.test_ids),
                z_lookup=z_lookup,
                has_doc_ref_lookup=has_doc_ref_lookup,
                retrieval_recall_lookup=retrieval_recall_lookup,
                fold_k=p.k,
                tiebreak=config.decision.tiebreak_strategy,
            )
        )
        rows_b3.extend(
            bl.b3_ponderado_calibrado(
                pesos_art=weights_art,
                calibradores=calibrators,
                by_agent=by_agent,
                test_ids=list(p.test_ids),
                z_lookup=z_lookup,
                has_doc_ref_lookup=has_doc_ref_lookup,
                retrieval_recall_lookup=retrieval_recall_lookup,
                fold_k=p.k,
                tiebreak=config.decision.tiebreak_strategy,
            )
        )
        for aid in by_agent:
            rows_bsolo_por_cmin = bl.bsolo_agente_calibrado_com_abstencao(
                agent_id=aid,
                calibrador=calibrators[aid],
                by_agent=by_agent,
                val_ids=list(p.val_ids),
                test_ids=list(p.test_ids),
                z_lookup=z_lookup,
                has_doc_ref_lookup=has_doc_ref_lookup,
                retrieval_recall_lookup=retrieval_recall_lookup,
                fold_k=p.k,
                c_min_grid=list(config.decision.c_min_grid),
            )
            for c_min, rows_bsolo in rows_bsolo_por_cmin.items():
                artefatos_bsolo[(aid, c_min)].extend(rows_bsolo)

        val_rows: list[dict[str, Any]] = []
        for iid in p.val_ids:
            conf_raw_v = {}
            calibs = {}
            for aid, table in by_agent.items():
                raw_conf = table.loc[iid, "confidence_raw"]
                conf = float(raw_conf) if pd.notna(raw_conf) else 0.0
                conf_raw_v[aid] = conf
                calibs[aid] = aplicar_calibracao(calibrators[aid], conf)
            winner, eta = decidir(
                conf_raw_v,
                calibs,
                weights_art.weights,
                metodo=config.decision.method,
                desempate=config.decision.tiebreak_strategy,
            )
            val_rows.append({"eta": eta, "z_system": z_lookup[(iid, winner)]})
        val_df = pd.DataFrame(val_rows)

        test_rows_base: list[dict[str, Any]] = []
        for iid in p.test_ids:
            conf_raw = {}
            conf_calib = {}
            z_per_agent = {}
            for aid, table in by_agent.items():
                raw_conf = table.loc[iid, "confidence_raw"]
                conf = float(raw_conf) if pd.notna(raw_conf) else 0.0
                conf_raw[aid] = conf
                conf_calib[aid] = aplicar_calibracao(calibrators[aid], conf)
                z_per_agent[aid] = z_lookup[(iid, aid)]
            winner, eta = decidir(
                conf_raw,
                conf_calib,
                weights_art.weights,
                metodo=config.decision.method,
                desempate=config.decision.tiebreak_strategy,
            )
            test_rows_base.append(
                {
                    "item_id": iid,
                    "fold_k": p.k,
                    "has_doc_ref": has_doc_ref_lookup[iid],
                    "agent_chosen": winner,
                    "eta": eta,
                    "z_winner": z_lookup[(iid, winner)],
                    "_conf_raw": conf_raw,
                    "_conf_calib": conf_calib,
                    "_z_per_agent": z_per_agent,
                }
            )

        # weights_art é persistido em cada subdiretório cmin=X.X/ embora não dependa de C_min.
        pesos_json = json.dumps(weights_art.model_dump(), ensure_ascii=False, indent=2)

        for c_min in config.decision.c_min_grid:
            t_cmin = time.perf_counter()
            wp = caminho_pesos_fold(base_dir, config.experiment_id, p.k, c_min)
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_text(pesos_json, encoding="utf-8")

            threshold = encontrar_tau_estrela(
                etas=val_df["eta"].to_numpy(),
                z=val_df["z_system"].to_numpy(),
                c_min=c_min,
            )
            tp = caminho_limiar_fold(base_dir, config.experiment_id, p.k, c_min)
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(
                json.dumps(threshold.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            test_rows: list[dict[str, Any]] = []
            for base in test_rows_base:
                abstained = bool(base["eta"] > threshold.tau_star)
                test_rows.append(
                    {
                        "item_id": base["item_id"],
                    "fold_k": base["fold_k"],
                    "has_doc_ref": base["has_doc_ref"],
                    "retrieval_recall": retrieval_recall_lookup.get(base["item_id"], float("nan")),
                    "agent_chosen": base["agent_chosen"],
                        "eta": base["eta"],
                        "abstained": abstained,
                        "z_winner": base["z_winner"],
                        "z_system": None if abstained else base["z_winner"],
                        **{f"confidence_raw_{aid}": base["_conf_raw"][aid] for aid in by_agent},
                        **{f"confidence_calib_{aid}": base["_conf_calib"][aid] for aid in by_agent},
                        **{f"z_agent_{aid}": base["_z_per_agent"][aid] for aid in by_agent},
                    }
                )
            artefatos_full[c_min].extend(test_rows)
            escrever_parquet(
                pd.DataFrame(test_rows),
                caminho_decisoes_teste_fold(base_dir, config.experiment_id, p.k, c_min),
            )
            logger.info(
                "Fold k=%02d c_min=%.1f: tau*=%.4f, constraint_ok=%s, %d decisão(ões) de teste gravadas em %.2fs.",
                p.k,
                c_min,
                threshold.tau_star,
                threshold.constraint_satisfied,
                len(test_rows),
                time.perf_counter() - t_cmin,
            )
        logger.info("Fold k=%02d concluído em %.2fs.", p.k, time.perf_counter() - t_fold)

    for c_min, rows in artefatos_full.items():
        df = pd.DataFrame(rows)
        escrever_parquet(
            df,
            caminho_teste_consolidado(base_dir, config.experiment_id, c_min=c_min, baseline="full"),
        )
        consolidados_full[c_min] = df
        logger.info(
            "Consolidado full c_min=%.1f gravado com %d linha(s).",
            c_min,
            len(df),
        )

    for baseline_id, rows in [("B1", rows_b1), ("B2", rows_b2), ("B3", rows_b3)]:
        df = pd.DataFrame(rows)
        escrever_parquet(
            df,
            caminho_teste_consolidado(base_dir, config.experiment_id, baseline=baseline_id),
        )
        logger.info("Baseline %s gravado com %d linha(s).", baseline_id, len(df))

    for (agent_id, c_min), rows in artefatos_bsolo.items():
        df = pd.DataFrame(rows)
        escrever_parquet(
            df,
            caminho_teste_consolidado(
                base_dir,
                config.experiment_id,
                c_min=c_min,
                baseline="Bsolo",
                agent_id=agent_id,
            ),
        )
        logger.info(
            "Baseline Bsolo agente=%s c_min=%.1f gravado com %d linha(s).",
            agent_id,
            c_min,
            len(df),
        )

    consolidated = consolidados_full[config.decision.c_min_grid[0]]
    fold_ks = [p.k for p in partitions]

    def _risk_stat(df: pd.DataFrame) -> float:
        return risco_seletivo(df)

    def _cov_stat(df: pd.DataFrame) -> float:
        return cobertura_empirica(df)

    risk_ci = ic_bootstrap_estratificado(
        consolidated,
        _risk_stat,
        coluna_estratificacao="has_doc_ref",
        n_reamostras=config.bootstrap.n_resamples,
        nivel_confianca=config.bootstrap.confidence_level,
        metodo=config.bootstrap.method,
        seed=config.bootstrap.seed,
    )
    cov_ci = ic_bootstrap_estratificado(
        consolidated,
        _cov_stat,
        coluna_estratificacao="has_doc_ref",
        n_reamostras=config.bootstrap.n_resamples,
        nivel_confianca=config.bootstrap.confidence_level,
        metodo=config.bootstrap.method,
        seed=config.bootstrap.seed,
    )
    logger.info(
        "Bootstrap concluído: risco=%.4f [%.4f, %.4f], cobertura=%.4f [%.4f, %.4f].",
        risk_ci.point,
        risk_ci.ci_lo,
        risk_ci.ci_hi,
        cov_ci.point,
        cov_ci.ci_lo,
        cov_ci.ci_hi,
    )

    ece_raw: dict[str, float] = {}
    ece_cal: dict[str, float] = {}
    for aid in by_agent:
        ece_raw[aid] = erro_calibracao_esperado(
            consolidated[f"confidence_raw_{aid}"].to_numpy(),
            consolidated[f"z_agent_{aid}"].to_numpy(),
        )
        ece_cal[aid] = erro_calibracao_esperado(
            consolidated[f"confidence_calib_{aid}"].to_numpy(),
            consolidated[f"z_agent_{aid}"].to_numpy(),
        )

    z_for_div = {
        aid: consolidated[f"z_agent_{aid}"].astype(int).tolist()
        for aid in by_agent
    }
    diversity = matriz_concordancia_erros(z_for_div)

    estabilidade: dict[str, dict[str, Any]] = {}
    for c_min in config.decision.c_min_grid:
        estabilidade[f"cmin={c_min:.1f}"] = {
            "weights": estatisticas_estabilidade_pesos(
                base_dir, config.experiment_id, fold_ks, c_min
            ),
            "tau_star": estatisticas_estabilidade_limiar(
                base_dir, config.experiment_id, fold_ks, c_min
            ),
        }

    mdir = diretorio_metricas(base_dir, config.experiment_id)
    mdir.mkdir(parents=True, exist_ok=True)

    if K_RELIABILITY in fold_ks:
        fold_diag = consolidated[consolidated["fold_k"] == K_RELIABILITY]
        rdir = diretorio_reliability(base_dir, config.experiment_id, K_RELIABILITY)
        rdir.mkdir(parents=True, exist_ok=True)
        for aid in by_agent:
            for tipo, col_conf in (
                ("raw", f"confidence_raw_{aid}"),
                ("calibrated", f"confidence_calib_{aid}"),
            ):
                df_diag = dados_diagrama_confiabilidade(
                    fold_diag[col_conf].to_numpy(),
                    fold_diag[f"z_agent_{aid}"].to_numpy(),
                    n_bins=15,
                )
                escrever_parquet(df_diag, rdir / f"{aid}_{tipo}.parquet")
                logger.info(
                    "Reliability fold k=%02d: %s/%s com %d bin(s).",
                    K_RELIABILITY,
                    aid,
                    tipo,
                    len(df_diag),
                )
    else:
        logger.warning(
            "Reliability diagram solicitado para k=%d, mas esse fold não existe nas partições. Pulando.",
            K_RELIABILITY,
        )

    pontos_curva = curva_risco_cobertura(
        etas=consolidated["eta"].to_numpy(),
        z=consolidated["z_winner"].astype(int).to_numpy(),
        n_pontos=200,
    )
    df_curva = pd.DataFrame(
        [
            {
                "tau": p.tau,
                "cobertura": p.cobertura,
                "risco_seletivo": p.risco_seletivo,
                "n_respondidos": p.n_respondidos,
                "estrato": "all",
            }
            for p in pontos_curva
        ]
    )
    curvas_estratificadas: list[pd.DataFrame] = [df_curva]
    for nome_estrato, mask in [
        ("with_doc_ref", consolidated["has_doc_ref"]),
        ("without_doc_ref", ~consolidated["has_doc_ref"]),
    ]:
        sub = consolidated[mask]
        if len(sub) == 0:
            continue
        pontos = curva_risco_cobertura(
            etas=sub["eta"].to_numpy(),
            z=sub["z_winner"].astype(int).to_numpy(),
            n_pontos=200,
        )
        curvas_estratificadas.append(
            pd.DataFrame(
                [
                    {
                        "tau": p.tau,
                        "cobertura": p.cobertura,
                        "risco_seletivo": p.risco_seletivo,
                        "n_respondidos": p.n_respondidos,
                        "estrato": nome_estrato,
                    }
                    for p in pontos
                ]
            )
        )
    escrever_parquet(
        pd.concat(curvas_estratificadas, ignore_index=True),
        caminho_curva_risco_cobertura(base_dir, config.experiment_id),
    )
    logger.info(
        "Curva risco-cobertura gravada com %d estrato(s).",
        len(curvas_estratificadas),
    )

    pontos_tau = []
    for c_min in config.decision.c_min_grid:
        stats = estabilidade[f"cmin={c_min:.1f}"]["tau_star"]
        pontos_tau.append(
            {
                "c_min": c_min,
                "tau_star_mean": stats["mean"],
                "tau_star_std": stats["std"],
                "n_folds": stats["n_folds"],
            }
        )
    escrever_parquet(
        pd.DataFrame(pontos_tau),
        caminho_pontos_tau_estrela(base_dir, config.experiment_id),
    )
    logger.info("Pontos de estabilidade de tau* gravados para %d valor(es) de c_min.", len(pontos_tau))

    metrics_json = mdir / "summary.json"
    bsolo_summary: dict[str, dict[str, Any]] = {}
    for (agent_id, c_min), rows in artefatos_bsolo.items():
        df = pd.DataFrame(rows)
        bsolo_summary[f"agent={agent_id}|cmin={c_min:.1f}"] = {
            "agent": agent_id,
            "c_min": c_min,
            "coverage": _bootstrap_metric_or_none(df, cobertura_empirica, config=config),
            "risk_selective": _bootstrap_metric_or_none(df, risco_seletivo, config=config),
            "adequacy_useful": _bootstrap_metric_or_none(df, _adequacao_util, config=config),
            "precision_abstention": _bootstrap_metric_or_none(
                df, _precisao_abstencao_bsolo, config=config
            ),
            "base_rate_inadequacy": _bootstrap_metric_or_none(
                df, _taxa_base_inadequacao_bsolo, config=config
            ),
            "lift_pp": _bootstrap_metric_or_none(df, _lift_abstencao_pp_bsolo, config=config),
            "n_abstentions": _bootstrap_metric_or_none(df, _n_abstencoes, config=config),
            "auroc": _bootstrap_metric_or_none(df, _auroc_bsolo, config=config),
        }
    metrics_json.write_text(
        json.dumps(
            {
                "risk_selective": risk_ci.model_dump(),
                "coverage": cov_ci.model_dump(),
                "ece_raw": ece_raw,
                "ece_calibrated": ece_cal,
                "diversity_matrix": diversity.to_dict(),
                "stability": estabilidade,
                "bsolo": bsolo_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Execução concluída em %.2fs. Artefatos principais: manifest=%s metrics=%s",
        time.perf_counter() - t_execucao,
        caminho_manifesto(base_dir, config.experiment_id),
        metrics_json,
    )

    return ArtefatosExecucao(
        diretorio_execucao=root,
        consolidated_tests={
            c_min: caminho_teste_consolidado(
                base_dir, config.experiment_id, c_min=c_min, baseline="full"
            )
            for c_min in config.decision.c_min_grid
        },
        metrics_json=metrics_json,
        manifest=caminho_manifesto(base_dir, config.experiment_id),
    )


def _parse_args() -> argparse.Namespace:
    """Define e parseia argumentos de linha de comando do pipeline end-to-end."""
    parser = argparse.ArgumentParser(description="Executa pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="Caminho do YAML de experimento.")
    parser.add_argument(
        "--questions-json",
        default=None,
        help=(
            "Override opcional do arquivo principal de questões "
            "definido em dataset.questions_file."
        ),
    )
    parser.add_argument(
        "--corpus-json",
        nargs="*",
        metavar="PATH",
        default=None,
        help=(
            "Caminhos de JSONs adicionais para o corpus RAG "
            "que sobrescrevem dataset.corpus_files do YAML."
        ),
    )
    parser.add_argument(
        "--no-validate-models",
        action="store_true",
        help="Pula validação prévia de slugs no OpenRouter.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrada CLI."""
    args = _parse_args()
    config = carregar_config_experimento(args.config)
    secrets = Segredos()

    artifacts = executar_experimento(
        config,
        secrets,
        questions_json_path=args.questions_json,
        corpus_extra_paths=args.corpus_json,
        validate_models=not args.no_validate_models,
    )
    logger.info("Run finalizada em %s", artifacts.diretorio_execucao)


if __name__ == "__main__":
    main()
