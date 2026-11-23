"""Confirmação N=200 com agentes de API (phase1_api_pilot_200_1) sobre R3.
Off-plan: reusa executar_confirmacao (frozen); não toca src/ nem runs/confirmacao/novo.
Uso: uv run python run_api.py <saida> [limite]"""
import sys
from pathlib import Path
from rag2.config import ConfigRag2
from rag2.confirmacao import executar_confirmacao, itens_confirmacao
from rag2.dados import carregar_itens
from rag2.escada import FABRICAS
from rag2.geracao import preparar_cache_juiz
from rag2.prompts import VARIANTES
from rag2.constantes import MODELO_JUIZ, MAX_TOKENS_JUIZ, MAX_TOKENS_AUDITOR, MAX_TOKENS_SUFICIENCIA, SEMENTE
from ensemble_llm.configuracao import Segredos
from ensemble_llm.agentes.cliente_openrouter import ClienteOpenRouter
from ensemble_llm.juiz.executor import ExecutorJuiz

saida = Path(sys.argv[1]); saida.mkdir(parents=True, exist_ok=True)
limite = int(sys.argv[2]) if len(sys.argv) > 2 else None
cfg = ConfigRag2(); s = Segredos()
MAX_AG = 4096  # phase1_api_pilot_200_1: max_tokens 4096, temp 0, seed 42

def orouter(model, max_tokens, reasoning_effort=None):
    return ClienteOpenRouter(model=model, api_key=s.openrouter_api_key, url=s.openrouter_url,
        referer=s.openrouter_referer, temperature=0.0, max_tokens=max_tokens, seed=SEMENTE,
        reasoning_effort=reasoning_effort)

clientes_agentes = {
    "claude-haiku-4.5":      orouter("anthropic/claude-haiku-4.5", MAX_AG),
    "gemini-3.1-flash-lite": orouter("google/gemini-3.1-flash-lite", MAX_AG, reasoning_effort="none"),
    "gpt-5.4-nano":          orouter("openai/gpt-5.4-nano", MAX_AG),
}
executor = ExecutorJuiz(client=orouter(MODELO_JUIZ, MAX_TOKENS_JUIZ), cache=preparar_cache_juiz(cfg))
itens = itens_confirmacao(cfg, carregar_itens(cfg))
res = executar_confirmacao(
    "R3", FABRICAS["R3"](cfg), itens, VARIANTES["P0"](), clientes_agentes, executor,
    orouter(MODELO_JUIZ, MAX_TOKENS_AUDITOR), orouter(MODELO_JUIZ, MAX_TOKENS_SUFICIENCIA),
    saida=saida, limite=limite,
    ambiente={"experimento": "api_agents_R3", "agentes": "phase1_api_pilot_200_1", "juiz": MODELO_JUIZ},
)
print("OK adequacao_media=%.4f suf=%.4f custo_usd=%.4f" % (
    res["adequacao_media"], res["suficiencia"]["frac_suficiente"], res["custo"]["custo_usd_total"]))
pa=res.get("por_agente", {})
for a in clientes_agentes:
    v=pa.get(a); 
    print("  ", a, (round(v,4) if isinstance(v,(int,float)) else v))
