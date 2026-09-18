"""
Validação prévia dos slugs dos modelos (OpenRouter e/ou Ollama) — gratuita.

Antes de gastar qualquer token de API ou hora de GPU, este script:
1. Carrega config YAML.
2. Para cada agente OpenRouter: GET /models contra a API.
3. Para cada agente Ollama: GET /api/tags contra o host configurado.
4. Para cada agente llama-swap: GET /v1/models contra o host configurado.
5. Para o juiz (sempre OpenRouter): GET /models.
6. Imprime modelos similares se algum slug não existir.

Uso:
    python scripts/validate_config.py configs/phase1_api_pilot.yaml
    python scripts/validate_config.py configs/phase2_local_full.yaml
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from ensemble_llm.configuracao import Segredos, carregar_config_experimento
from ensemble_llm.orquestracao.validacao import ErroValidacao, validar_config_experimento


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config não encontrado: {config_path}", file=sys.stderr)
        return 1

    print(f"Carregando config: {config_path}")
    config = carregar_config_experimento(config_path)
    print(f"  experiment_id: {config.experiment_id}")
    print(f"  phase:         {config.phase}")
    print(f"  dataset_id:    {config.dataset.dataset_id}")
    print(f"  dataset_fmt:   {config.dataset.format}")
    print(f"  questions:     {config.dataset.resolver_questions_path()}")
    if config.dataset.corpus_files:
        print("  corpus_files:")
        for path in config.dataset.resolver_corpus_paths():
            print(f"    - {path}")
    else:
        print("  corpus_files:  []")
    print()

    # Resumo por backend
    backends = Counter(a.backend for a in config.agents)
    print("Agentes a validar:")
    for agent in config.agents:
        host_info = f"  host={agent.host}" if agent.host else ""
        print(f"  [{agent.backend:>10s}] {agent.agent_id:<15s} → {agent.model}{host_info}")
    print(f"\nResumo: {dict(backends)}")
    print(f"Juiz [{config.judge.backend}]: {config.judge.judge_id} → {config.judge.model}")
    print()

    print("Carregando secrets de .env...")
    try:
        secrets = Segredos()  # type: ignore[call-arg]
    except Exception as e:
        print(f"Erro ao carregar .env: {e}", file=sys.stderr)
        print("Verifique que .env existe; OPENROUTER_API_KEY é obrigatório para o juiz.", file=sys.stderr)
        return 1

    # Avisos prévios sobre backends locais
    if "ollama" in backends:
        print()
        print("NOTA: para agentes Ollama, verifique antes que:")
        print("  1. Servidor Ollama está rodando (geralmente `ollama serve`).")
        print("  2. Modelos foram baixados (`ollama pull <slug>`).")
        print(f"  3. Host acessível (default {secrets.ollama_host_gpu0}).")
    if "llama-swap" in backends:
        print()
        print("NOTA: para agentes llama-swap, verifique antes que:")
        print("  1. Servidor llama-swap está rodando.")
        print("  2. Modelo alvo está configurado/carregado no swap.")
        print(f"  3. Host acessível (default {secrets.llama_swap_host}).")

    print()
    print("Validando contra os backends configurados...")
    print("  - OpenRouter: GET /models (gratuito)")
    if "ollama" in backends:
        print("  - Ollama:     GET /api/tags por host (gratuito)")
    if "llama-swap" in backends:
        print("  - llama-swap: GET /v1/models por host (gratuito)")
    print()

    try:
        validar_config_experimento(config, secrets)
    except ErroValidacao as e:
        print(f"FALHA DE VALIDAÇÃO:\n{e}", file=sys.stderr)
        return 1

    print()
    print("✓ Todos os modelos disponíveis. Pronto para rodar.")
    if "ollama" in backends or "llama-swap" in backends:
        print()
        print("Próximo passo sugerido:")
        print(f"  uv run python scripts/smoke_e2e.py {config_path} --n 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
