"""
Tools del agente SoporteGit (Avance 3).

Estilo de la práctica de clase:
  - @tool de LangChain
  - Límites escritos en el código (no solo en el prompt)
  - Cada fragmento incluye documento y página/sección para citar
  - Tool de tarea de dominio que el LLM no debe hacer de memoria
  - Búsqueda web controlada (Tavily) con dominios fijos

Uso de prueba sin LLM:
    python -m src.tools
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Límites en código (requisito del avance)
# ---------------------------------------------------------------------------
MAX_FRAGMENTS = 4          # máximo de fragmentos por llamada a buscar_en_base
MAX_CHARS_PER_FRAGMENT = 600  # longitud máxima de cada fragmento
MAX_WEB_RESULTS = 3        # máximo de resultados web

# Dominios permitidos (el LLM no puede cambiar esta lista)
ALLOWED_DOMAINS = [
    "git-scm.com",
    "docs.github.com",
    "stackoverflow.com",
    "atlassian.com",
    "learngitbranching.js.org",
]


# ---------------------------------------------------------------------------
# Tool 1: Búsqueda en la base de conocimientos
# ---------------------------------------------------------------------------
def _get_vectorstore():
    """Importación diferida para evitar ciclos y cargar solo cuando se use."""
    from src.rag_chain import get_vectorstore
    return get_vectorstore()


@tool
def buscar_en_base(consulta: str) -> str:
    """Busca en los manuales de Git de la base de conocimientos interna.

    Úsala SIEMPRE primero cuando la pregunta trate de comandos Git, ramas,
    rebase, merge, tags, descartar cambios, sincronización con remoto,
    historial o cualquier tema técnico de Git cubierto en los manuales.

    NO la uses para:
    - Saludos o conversación casual
    - Información pública actual (versiones recientes, noticias)
    - Cálculos o tareas exactas (usa las tools de dominio)

    Cada fragmento devuelto incluye el nombre del documento y la sección
    para que puedas citar la fuente.
    """
    vs = _get_vectorstore()
    docs = vs.similarity_search(consulta, k=MAX_FRAGMENTS)

    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimientos."

    parts: List[str] = []
    for d in docs:
        source = d.metadata.get("source_file", d.metadata.get("source", "desconocido"))
        # page o chunk_id como identificador de sección
        page = d.metadata.get("page", d.metadata.get("chunk_id", "?"))
        content = d.page_content[:MAX_CHARS_PER_FRAGMENT]
        if len(d.page_content) > MAX_CHARS_PER_FRAGMENT:
            content += "..."
        parts.append(f"[Fuente: {source} | sección: {page}]\n{content}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 2: Tarea propia del dominio (Git)
# El LLM no debe hacer esto "de memoria".
# ---------------------------------------------------------------------------
@tool
def verificar_comando_destructivo(comando: str) -> str:
    """Verifica si un comando de Git es potencialmente destructivo
    (puede causar pérdida irreversible de datos) y devuelve la advertencia
    oficial correspondiente.

    Úsala cuando el usuario mencione o pida un comando que pueda borrar
    trabajo (reset --hard, clean -fd, push --force, rebase de ramas
    públicas, etc.).

    Args:
        comando: el comando Git a verificar (ej: "git reset --hard HEAD").
    """
    cmd = comando.strip().lower()

    # Reglas exactas (no inventadas por el LLM)
    reglas = [
        (
            ["reset --hard", "reset -hard"],
            "DESTRUCTIVO: 'git reset --hard' descarta de forma irreversible "
            "todos los cambios no confirmados en el directorio de trabajo y "
            "en el área de staging. No hay forma sencilla de recuperarlos.",
        ),
        (
            ["clean -fd", "clean -f -d", "clean -ffd"],
            "DESTRUCTIVO: 'git clean -fd' elimina archivos y directorios "
            "no rastreados de forma permanente.",
        ),
        (
            ["push --force", "push -f", "push --force-with-lease"],
            "PELIGROSO: 'git push --force' reescribe el historial remoto. "
            "Puede sobrescribir commits de otros colaboradores. "
            "Prefiere --force-with-lease y coordina con el equipo.",
        ),
        (
            ["rebase", "rebase -i"],
            "PRECAUCIÓN: rebase reescribe el historial. No lo uses en ramas "
            "públicas/compartidas que otros ya hayan descargado.",
        ),
        (
            ["branch -d", "branch -D"],
            "ATENCIÓN: eliminar una rama borra la referencia. "
            "Si la rama no está mergeada, usa -d (seguro) en lugar de -D.",
        ),
    ]

    for patrones, mensaje in reglas:
        if any(p in cmd for p in patrones):
            return f"Resultado de verificación: {mensaje}"

    return (
        f"Resultado de verificación: el comando '{comando}' no se identifica "
        "como destructivo según las reglas internas. Aun así, revisa la "
        "documentación si tienes dudas."
    )


@tool
def generar_referencia_apa(documento: str, seccion: str = "") -> str:
    """Genera la referencia bibliográfica en formato APA 7 de un documento
    de la base de conocimientos de Git.

    Úsala cuando el usuario pida citar, referenciar o generar la bibliografía
    de un manual de la base.

    Args:
        documento: nombre del archivo (ej: "01_comandos_basicos.md").
        seccion: opcional, número de sección o página.
    """
    # Mapeo exacto de los documentos del proyecto
    catalogo = {
        "00_introduccion_git.md": (
            "Equipo SoporteGit. (2024). Introducción a Git. "
            "Manual técnico interno de SoporteGit."
        ),
        "01_comandos_basicos.md": (
            "Equipo SoporteGit. (2024). Comandos básicos de Git. "
            "Manual técnico interno de SoporteGit."
        ),
        "02_sincronizacion_ramas.md": (
            "Equipo SoporteGit. (2024). Sincronización de ramas. "
            "Manual técnico interno de SoporteGit."
        ),
        "03_descartar_cambios.md": (
            "Equipo SoporteGit. (2024). Descartar cambios en Git. "
            "Manual técnico interno de SoporteGit."
        ),
        "04_rebase_y_historial.md": (
            "Equipo SoporteGit. (2024). Rebase e historial. "
            "Manual técnico interno de SoporteGit."
        ),
        "05_tags_y_releases.md": (
            "Equipo SoporteGit. (2024). Tags y releases. "
            "Manual técnico interno de SoporteGit."
        ),
        "progit.pdf": (
            "Chacon, S., & Straub, B. (2014). Pro Git (2.ª ed.). Apress. "
            "https://git-scm.com/book/en/v2"
        ),
    }

    # Normalizar nombre
    nombre = Path(documento).name
    ref = catalogo.get(nombre) or catalogo.get(documento)

    if not ref:
        return (
            f"No se encontró el documento '{documento}' en el catálogo de la base. "
            "Documentos disponibles: " + ", ".join(catalogo.keys())
        )

    if seccion:
        ref += f" (sección {seccion})."
    else:
        ref += "."

    return ref


# ---------------------------------------------------------------------------
# Tool 3: Búsqueda web controlada (Tavily)
# ---------------------------------------------------------------------------
@tool
def buscar_en_web(consulta: str) -> str:
    """Busca información pública y actual sobre Git en sitios oficiales y confiables.

    SOLO úsala cuando se cumplan TODAS estas condiciones:
    1. La tool buscar_en_base NO devolvió información suficiente, Y
    2. La pregunta trata de información pública o actual (documentación oficial
       actualizada, versión reciente de Git, definición general, best practices
       actuales).

    NUNCA la uses como primera opción ni para saludos o charla casual.
    La lista de sitios permitidos está fija en el código y no puede cambiarse.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: falta TAVILY_API_KEY. Configúrala en .env o en los secrets "
            "de la plataforma de despliegue."
        )

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=consulta,
            max_results=MAX_WEB_RESULTS,
            include_domains=ALLOWED_DOMAINS,
            search_depth="basic",
        )

        results = response.get("results", [])[:MAX_WEB_RESULTS]
        if not results:
            return "No se encontraron resultados web relevantes en los sitios permitidos."

        parts: List[str] = []
        for r in results:
            title = r.get("title", "Sin título")
            url = r.get("url", "")
            content = (r.get("content") or "")[:300]
            parts.append(f"[web: {url}]\nTítulo: {title}\nExtracto: {content}")

        return "\n\n".join(parts)

    except Exception as e:
        return f"Error al realizar la búsqueda web: {e}"


# ---------------------------------------------------------------------------
# Lista de tools listas para el agente
# ---------------------------------------------------------------------------
def get_tools() -> list:
    """Devuelve la lista de tools que el agente puede usar."""
    return [
        buscar_en_base,
        verificar_comando_destructivo,
        generar_referencia_apa,
        buscar_en_web,
    ]


# ---------------------------------------------------------------------------
# Pruebas sin LLM (requisito del avance)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de tools SIN LLM")
    print("=" * 60)

    print("\n1) buscar_en_base.invoke(...)")
    print("-" * 40)
    try:
        out = buscar_en_base.invoke({"consulta": "cómo crear una nueva rama"})
        print(out[:800] + ("..." if len(out) > 800 else ""))
    except Exception as e:
        print(f"Error (¿índice creado?): {e}")

    print("\n2) verificar_comando_destructivo.invoke(...)")
    print("-" * 40)
    out = verificar_comando_destructivo.invoke({"comando": "git reset --hard HEAD"})
    print(out)

    print("\n3) generar_referencia_apa.invoke(...)")
    print("-" * 40)
    out = generar_referencia_apa.invoke(
        {"documento": "03_descartar_cambios.md", "seccion": "2"}
    )
    print(out)

    print("\n4) buscar_en_web (requiere TAVILY_API_KEY)")
    print("-" * 40)
    if os.getenv("TAVILY_API_KEY"):
        out = buscar_en_web.invoke({"consulta": "git rebase best practices 2024"})
        print(out[:600] + ("..." if len(out) > 600 else ""))
    else:
        print("TAVILY_API_KEY no configurada → se omite la prueba de web.")