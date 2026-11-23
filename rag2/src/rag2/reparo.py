"""Reparo do corpus normativo: baixa das fontes oficiais o texto integral dos documentos truncados e grava o JSON reparado."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

from rag2 import canonico
from rag2.config import RAIZ, ConfigRag2
from rag2.dados import carregar_itens, contar_palavras, itens_com_dispositivo

DIR_REPARO = RAIZ / "data" / "reparo"
"""Textos extraídos, um arquivo por alvo baixado (versionados); brutos/ guarda o HTML (gitignored)."""
DIR_REPARADO = RAIZ / "data" / "reparado"
NOME_JSON_REPARADO = "referred_legal_documents_QA_2024_v1.1_reparado.json"
NOME_RELATORIO = "relatorio_reparo.md"
JANELA_TRECHO = 15
"""Palavras consecutivas do original que devem reaparecer no texto novo para valer como o mesmo documento."""
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) rag2-reparo/0.1"
CAMPOS_PROIBIDOS = frozenset(
    {"id", "numero_acordao", "number", "doc_id", "key", "acórdão",
     "text", "texto", "ementa", "content", "conteudo", "body", "decision"}
)
"""Nomes que _carregar_json_corpus da origem leria como id ou texto; os campos novos nunca os usam."""

REPARADO = "REPARADO"      # texto novo tem mais palavras e compartilha trecho com o original
SEM_GANHO = "SEM_GANHO"    # a fonte oficial não tem mais texto que o dataset
SEM_FONTE = "SEM_FONTE"    # não há HTML estático oficial nesta data (declarado em FONTES)
FALHA = "FALHA"            # download, codificação ou documento errado — STOP, não improviso


@dataclass(frozen=True)
class Fonte:
    """Página oficial de um alvo; `url=None` declara que não há fonte estática (motivo em `nota`)."""

    url: str | None
    nota: str = ""
    codificacao: str = "windows-1252"   # usada só quando o servidor não declara charset (Planalto: nunca declara)


NOTA_SIJUT = "SIJUT/RFB sem HTML estático nesta data (SPA desde 2024-09; ver docs/decisions.md)"
NOTA_ANEXO_PDF = "página idêntica ao dataset; o dispositivo esperado está em anexo PDF digitalizado (sem camada de texto)"

ALVOS: tuple[str, ...] = (
    "Medida Provisória nº 252.txt",                                      # 27 palavras
    "Lei nº 10.406.txt",                                                 # 62
    "Decreto nº 3.000.txt",                                              # 90
    "Instrução Normativa RFB nº 118.txt",                                # 109
    "Decreto nº 93.153.txt",                                             # 171
    "Decreto nº 85.801.txt",                                             # 183
    "Decreto nº 27.784.txt",                                             # 184
    "Instrução Normativa SRF nº 4, de 13 de janeiro de 1999.txt",        # 232
    "Decreto nº 361.txt",                                                # 245
    "Lei nº 8.971.txt",                                                  # 285
    "Constituição Federal de 1988.txt",                                  # 8106, truncada no art. 24
)
"""Os dez arquivos esperados com < 300 palavras (ordem crescente) mais a Constituição — fixos (DESIGN §4.3)."""

FONTES: dict[str, Fonte] = {
    "Medida Provisória nº 252.txt": Fonte("https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2005/mpv/252.htm"),
    "Lei nº 10.406.txt": Fonte("https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm"),
    "Decreto nº 3.000.txt": Fonte("https://www.planalto.gov.br/ccivil_03/decreto/d3000.htm"),
    "Instrução Normativa RFB nº 118.txt": Fonte(None, NOTA_SIJUT),
    "Decreto nº 93.153.txt": Fonte("https://www.planalto.gov.br/ccivil_03/decreto/1980-1989/1985-1987/d93153.htm", NOTA_ANEXO_PDF),
    "Decreto nº 85.801.txt": Fonte("https://www.planalto.gov.br/ccivil_03/atos/decretos/1981/d85801.html", NOTA_ANEXO_PDF),
    "Decreto nº 27.784.txt": Fonte("https://www.planalto.gov.br/ccivil_03/decreto/antigos/d27784.htm", NOTA_ANEXO_PDF),
    "Instrução Normativa SRF nº 4, de 13 de janeiro de 1999.txt": Fonte(None, NOTA_SIJUT),
    "Decreto nº 361.txt": Fonte("https://www.planalto.gov.br/ccivil_03/decreto/1990-1994/d0361.htm", NOTA_ANEXO_PDF),
    "Lei nº 8.971.txt": Fonte("https://www.planalto.gov.br/ccivil_03/leis/l8971.htm", "lei curta (5 artigos); a fonte tem as mesmas 285 palavras"),
    "Constituição Federal de 1988.txt": Fonte("https://www.planalto.gov.br/ccivil_03/constituicao/constituicaocompilado.htm"),
}
"""filename → página oficial (Planalto, versão compilada quando existe). Sondadas em 2026-09-12."""


@dataclass(frozen=True)
class ResultadoReparo:
    """O que aconteceu com um alvo; `linha()` é a linha impressa pela CLI e a do relatório."""

    filename: str
    status: str
    palavras_antes: int
    palavras_depois: int
    trecho_comum: bool
    artigos_total: int          # artigos esperados distintos neste arquivo
    artigos_antes: int          # quantos casam a regex canônica no texto original
    artigos_depois: int         # idem no texto que vai para o JSON
    pares_ganhos: int           # pares (com repetição entre itens) que não casavam e passam a casar
    fonte: str                  # URL, ou "" quando não há
    nota: str

    def linha(self) -> str:
        trecho = "—" if self.status == SEM_FONTE else ("sim" if self.trecho_comum else "não")
        return (
            f"{self.status:<9} {self.filename} | {self.palavras_antes} → {self.palavras_depois} palavras "
            f"| trecho={trecho} | artigos {self.artigos_antes}/{self.artigos_total} → "
            f"{self.artigos_depois}/{self.artigos_total} | {self.fonte or self.nota}"
        )


class _ExtratorTexto(HTMLParser):
    """Coleta o texto visível; ignora script, style, head e title (o título poluía o início do texto)."""

    IGNORAR = frozenset({"script", "style", "head", "title"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[str] = []
        self._ignorando = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.IGNORAR:
            self._ignorando += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORAR and self._ignorando:
            self._ignorando -= 1

    def handle_data(self, dados: str) -> None:
        if not self._ignorando:
            self.partes.append(dados)


def extrair_texto(html: str) -> str:
    """Texto visível do HTML: nós unidos por espaço, espaços em branco colapsados — a forma do filedata do dataset."""
    p = _ExtratorTexto()
    p.feed(html)
    p.close()
    return re.sub(r"\s+", " ", " ".join(p.partes)).strip()


def compartilha_trecho(original: str, novo: str, janela: int = JANELA_TRECHO) -> bool:
    """Verdadeiro se alguma sequência de `janela` palavras do original aparece no novo (mesmo documento).

    Original menor que a janela: exige o original inteiro. Original vazio: falso.
    """
    o, n = original.split(), novo.split()
    janela = min(janela, len(o))
    if janela == 0:
        return False
    janelas_novo = {tuple(n[i : i + janela]) for i in range(len(n) - janela + 1)}
    return any(tuple(o[i : i + janela]) in janelas_novo for i in range(len(o) - janela + 1))


def artigos_casaveis(texto: str, artigos: Iterable[str]) -> int:
    """Quantos artigos têm cabeçalho casável pela regex canônica no texto integral (limite superior do que um chunk casa)."""
    return sum(bool(canonico.regex_artigo(a).search(texto)) for a in artigos)


def baixar(fonte: Fonte, cliente: httpx.Client) -> tuple[bytes, str]:
    """GET da página oficial → (bytes brutos, texto decodificado). Status ≠ 200 é erro; não há fonte alternativa."""
    if fonte.url is None:
        raise ValueError("fonte sem URL")
    r = cliente.get(fonte.url)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} em {fonte.url}")
    return r.content, r.text


def cliente_http(codificacao_padrao: str = "windows-1252") -> httpx.Client:
    """Cliente com redirects, 60 s e `default_encoding`: o charset do cabeçalho vence; sem ele, o padrão da fonte."""
    return httpx.Client(
        timeout=60.0, follow_redirects=True, default_encoding=codificacao_padrao, headers={"User-Agent": USER_AGENT}
    )


def avaliar_reparo(
    filename: str, original: str, novo: str | None, esperados: Sequence[str], fonte: Fonte
) -> ResultadoReparo:
    """Regras de status (nesta ordem): sem URL → SEM_FONTE; sem texto ou U+FFFD ou sem trecho comum → FALHA;
    mais palavras → REPARADO; senão → SEM_GANHO. `esperados` traz repetição entre itens (para pares_ganhos)."""
    distintos = set(esperados)
    antes = contar_palavras(original)
    a_antes = artigos_casaveis(original, distintos)
    base = dict(filename=filename, palavras_antes=antes, artigos_total=len(distintos), artigos_antes=a_antes,
                fonte=fonte.url or "", nota=fonte.nota)
    if fonte.url is None:
        return ResultadoReparo(status=SEM_FONTE, palavras_depois=antes, trecho_comum=False,
                               artigos_depois=a_antes, pares_ganhos=0, **base)
    if novo is None or chr(0xFFFD) in novo or not compartilha_trecho(original, novo):
        motivo = "sem texto" if novo is None else ("codificação (U+FFFD)" if chr(0xFFFD) in novo else "documento errado")
        base["nota"] = f"{motivo}; {fonte.nota}".strip("; ")
        return ResultadoReparo(status=FALHA, palavras_depois=contar_palavras(novo or ""), trecho_comum=False,
                               artigos_depois=a_antes, pares_ganhos=0, **base)
    depois = contar_palavras(novo)
    if depois <= antes:
        return ResultadoReparo(status=SEM_GANHO, palavras_depois=depois, trecho_comum=True,
                               artigos_depois=a_antes, pares_ganhos=0, **base)
    ganhos = sum(
        1 for a in esperados
        if not canonico.regex_artigo(a).search(original) and canonico.regex_artigo(a).search(novo)
    )
    return ResultadoReparo(status=REPARADO, palavras_depois=depois, trecho_comum=True,
                           artigos_depois=artigos_casaveis(novo, distintos), pares_ganhos=ganhos, **base)


def reparar_registros(
    registros: Sequence[Mapping],
    textos: Mapping[str, str | None],
    alvos: Sequence[str],
    fontes: Mapping[str, Fonte],
    esperados: Mapping[str, Sequence[str]],
) -> tuple[list[dict], list[ResultadoReparo]]:
    """Registros novos (mesma ordem, mesmos filename, entrada não mutada) e um resultado por alvo.

    Todo registro ganha `reparado`, `palavras_antes`, `palavras_depois`, `fonte`; só REPARADO troca `filedata`.
    """
    saida: list[dict] = []
    resultados: dict[str, ResultadoReparo] = {}
    for reg in registros:
        novo_reg = dict(reg)
        fn, original = reg["filename"], reg["filedata"]
        n = contar_palavras(original)
        campos = {"reparado": False, "palavras_antes": n, "palavras_depois": n, "fonte": ""}
        if fn in alvos:
            r = avaliar_reparo(fn, original, textos.get(fn), list(esperados.get(fn, [])), fontes[fn])
            resultados[fn] = r
            campos.update(reparado=r.status == REPARADO, palavras_depois=r.palavras_depois, fonte=r.fonte)
            if r.status == REPARADO:
                novo_reg["filedata"] = textos[fn]
        novo_reg.update(campos)
        saida.append(novo_reg)
    faltantes = [a for a in alvos if a not in resultados]
    if faltantes:
        raise KeyError(f"alvos ausentes do JSON: {faltantes}")
    return saida, [resultados[a] for a in alvos]


def escrever_json(registros: Sequence[Mapping], caminho: Path) -> str:
    """Grava o JSON (UTF-8, indent=2, sem escapar acentos, newline final) e devolve o sha256 hex dos bytes."""
    dados = (json.dumps(list(registros), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(dados)
    return hashlib.sha256(dados).hexdigest()


def esperados_por_arquivo(cfg: ConfigRag2) -> dict[str, list[str]]:
    """arquivo → artigos esperados dos 556 itens, com repetição entre itens (uso diagnóstico, declarado)."""
    saida: dict[str, list[str]] = {}
    for it in itens_com_dispositivo(carregar_itens(cfg)):
        for arquivo, artigo in it.esperados:
            saida.setdefault(arquivo, []).append(artigo)
    return saida


def relatorio_markdown(resultados: Sequence[ResultadoReparo], sha256: str, n_registros: int) -> str:
    """Tabela dos alvos + resumo + sha256. Sem data (a data de acesso fica em docs/decisions.md): reprodutível byte a byte."""
    cont = Counter(r.status for r in resultados)
    linhas = [
        "# Reparo do corpus normativo",
        "",
        "Fontes: `FONTES` em `src/rag2/reparo.py`; textos extraídos em `data/reparo/`; "
        "data de acesso e não reparados em `docs/decisions.md`. Artigos casáveis medidos no texto integral "
        "(limite superior do casamento por chunk).",
        "",
        "| status | arquivo | palavras antes → depois | trecho comum | artigos casáveis antes → depois | pares que passam a casar | fonte / nota |",
        "|---|---|---:|:-:|---:|---:|---|",
    ]
    for r in resultados:
        trecho = "—" if r.status == SEM_FONTE else ("sim" if r.trecho_comum else "não")
        linhas.append(
            f"| {r.status} | {r.filename} | {r.palavras_antes} → {r.palavras_depois} | {trecho} "
            f"| {r.artigos_antes}/{r.artigos_total} → {r.artigos_depois}/{r.artigos_total} | {r.pares_ganhos} "
            f"| {r.fonte} {r.nota}".rstrip() + " |"
        )
    linhas += [
        "",
        f"**Resumo:** reparados={cont[REPARADO]} sem_ganho={cont[SEM_GANHO]} sem_fonte={cont[SEM_FONTE]} "
        f"falhas={cont[FALHA]} · pares que passam a casar: {sum(r.pares_ganhos for r in resultados)} "
        f"· registros: {n_registros}",
        "",
        f"sha256 do JSON reparado: `{sha256}`",
        "",
    ]
    return "\n".join(linhas)


def executar_reparo(
    cfg: ConfigRag2,
    *,
    offline: bool = False,
    so: str | None = None,
    sobrescrever: bool = False,
    dir_reparo: Path = DIR_REPARO,
    saida: Path = DIR_REPARADO,
) -> tuple[list[ResultadoReparo], str, list[str]]:
    """Obtém os textos (rede ou data/reparo/), monta o JSON reparado e o relatório; devolve (resultados, sha256, avisos).

    Registrar uma vez: online, um texto já existente em `dir_reparo` não é sobrescrito sem `sobrescrever`;
    se o download diferir dele, entra um aviso e vale o registrado. `so` restringe os alvos a um só.
    """
    with Path(cfg.arquivo_normas()).open(encoding="utf-8") as f:
        registros = json.load(f)
    alvos = [so] if so else list(ALVOS)
    dir_reparo, saida = Path(dir_reparo), Path(saida)
    textos: dict[str, str | None] = {}
    avisos: list[str] = []
    cliente = None if offline else cliente_http()
    try:
        for fn in alvos:
            fonte = FONTES[fn]
            if fonte.url is None:
                textos[fn] = None
                continue
            guardado = dir_reparo / fn
            if offline:
                textos[fn] = guardado.read_text(encoding="utf-8") if guardado.exists() else None
                if textos[fn] is None:
                    avisos.append(f"{fn}: ausente em {dir_reparo} (rode sem --offline)")
                continue
            try:
                bruto, html = baixar(fonte, cliente)
            except (httpx.HTTPError, RuntimeError) as e:
                avisos.append(f"{fn}: download falhou — {e}")
                textos[fn] = None
                continue
            (dir_reparo / "brutos").mkdir(parents=True, exist_ok=True)
            (dir_reparo / "brutos" / f"{fn}.html").write_bytes(bruto)
            novo = extrair_texto(html)
            if guardado.exists() and not sobrescrever:
                registrado = guardado.read_text(encoding="utf-8")
                if registrado != novo:
                    avisos.append(
                        f"{fn}: fonte mudou desde o registro ({contar_palavras(registrado)} → {contar_palavras(novo)} "
                        "palavras); mantido o registrado (--sobrescrever para atualizar)"
                    )
                novo = registrado
            else:
                guardado.write_text(novo, encoding="utf-8")
            textos[fn] = novo
    finally:
        if cliente is not None:
            cliente.close()
    novos, resultados = reparar_registros(registros, textos, alvos, FONTES, esperados_por_arquivo(cfg))
    sha = escrever_json(novos, saida / NOME_JSON_REPARADO)
    (saida / NOME_RELATORIO).write_text(relatorio_markdown(resultados, sha, len(novos)), encoding="utf-8")
    return resultados, sha, avisos
