"""
Interfaz de chat — SoporteGit Agente con tools (Avance 3).

Requisitos del avance:
  - Usa el agente (no el pipeline fijo)
  - Historial de conversación
  - Muestra tools usadas y argumentos (sección desplegable)
  - Fuentes separadas por origen: base (documento/sección) y web (título/URL)
  - Comportamiento correcto cuando no hay info en base ni en web
  - Claves solo por variables de entorno / secrets

Ejecución local:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag_chain import ensure_index, _index_is_ready
from src.agent import responder, get_llm_con_tools

st.set_page_config(
    page_title="SoporteGit — Agente RAG",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🔧 SoporteGit")
st.caption(
    "Agente con tools · Manuales de Git  \n"
    "Embeddings locales · ChromaDB · Groq · Búsqueda web controlada (Tavily)"
)

# ---------------------------------------------------------------------------
# Índice vectorial (Streamlit Cloud / primera ejecución)
# ---------------------------------------------------------------------------
if "index_ready" not in st.session_state:
    if not _index_is_ready():
        with st.spinner(
            "⏳ Primera ejecución: construyendo el índice vectorial "
            "(modelo de embeddings + indexado). Puede tardar 1-2 minutos..."
        ):
            try:
                ensure_index()
                st.session_state.index_ready = True
                st.success("✅ Índice listo. Ya puedes hacer preguntas.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al construir el índice: {e}")
                st.stop()
    else:
        st.session_state.index_ready = True

# ---------------------------------------------------------------------------
# Cargar LLM + tools una sola vez por sesión
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo y tools...")
def _cargar_agente():
    return get_llm_con_tools()


try:
    llm_con_tools, tools_por_nombre = _cargar_agente()
except Exception as e:
    st.error(f"No se pudo cargar el agente. Revisa GROQ_API_KEY en secrets/.env\n\n{e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Acerca de")
    st.markdown(
        """
        **SoporteGit (Avance 3)** es un agente que decide cuándo consultar
        la base de conocimientos y qué herramienta usar.

        **Tools disponibles:**
        - `buscar_en_base` — manuales Git internos
        - `verificar_comando_destructivo` — reglas de seguridad
        - `generar_referencia_apa` — citas APA
        - `buscar_en_web` — solo sitios oficiales (Tavily)

        **Política:** primero la base; web solo si la base no responde
        y la pregunta es pública/actual.
        """
    )
    st.divider()
    st.markdown(
        f"**Tools cargadas:** `{', '.join(tools_por_nombre.keys())}`"
    )
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------
def _separar_fuentes(tools_usadas: list) -> tuple[list, list]:
    """Separa resultados de base y de web a partir de las tools usadas."""
    fuentes_base = []
    fuentes_web = []

    for t in tools_usadas:
        nombre = t.get("name", "")
        texto = t.get("resultado") or t.get("resultado_preview") or ""
        args = t.get("args") or {}

        if nombre == "buscar_en_base" and texto:
            bloques = re.split(r"\n\n---\n\n", texto)
            for b in bloques:
                m = re.search(
                    r"\[Fuente:\s*([^|\]]+?)(?:\s*\|\s*sección:\s*([^\]]+))?\]",
                    b,
                )
                doc = m.group(1).strip() if m else "desconocido"
                seccion = m.group(2).strip() if m and m.group(2) else "?"
                contenido = re.sub(
                    r"^\[Fuente:[^\]]+\]\s*", "", b, count=1
                ).strip()
                fuentes_base.append(
                    {
                        "documento": doc,
                        "seccion": seccion,
                        "contenido": contenido[:500],
                        "consulta": args.get("consulta", ""),
                    }
                )

        elif nombre == "buscar_en_web" and texto:
            bloques = texto.split("\n\n")
            for b in bloques:
                m_url = re.search(r"\[web:\s*([^\]]+)\]", b)
                m_tit = re.search(r"Título:\s*(.+)", b)
                m_ext = re.search(r"Extracto:\s*(.+)", b, re.DOTALL)
                if m_url:
                    fuentes_web.append(
                        {
                            "url": m_url.group(1).strip(),
                            "titulo": m_tit.group(1).strip() if m_tit else "Sin título",
                            "extracto": (m_ext.group(1).strip()[:300] if m_ext else b[:300]),
                            "consulta": args.get("consulta", ""),
                        }
                    )

    return fuentes_base, fuentes_web


def _render_respuesta_asistente(msg: dict):
    """Pinta la respuesta del asistente con tools y fuentes."""
    st.markdown(msg.get("content", ""))

    tools_usadas = msg.get("tools_usadas") or []
    n_llm = msg.get("num_llamadas_llm")
    tiempo = msg.get("tiempo_segundos")

    meta_parts = []
    if n_llm is not None:
        meta_parts.append(f"Llamadas LLM: {n_llm}")
    if tiempo is not None:
        meta_parts.append(f"Tiempo: {tiempo}s")
    if tools_usadas:
        meta_parts.append(f"Tools: {len(tools_usadas)}")
    else:
        meta_parts.append("Sin tools")
    if meta_parts:
        st.caption(" · ".join(meta_parts))

    with st.expander("🛠️ Tools usadas y argumentos", expanded=bool(tools_usadas)):
        if not tools_usadas:
            st.info("El agente no llamó ninguna tool (saludo o respuesta directa).")
        else:
            for i, t in enumerate(tools_usadas, 1):
                st.markdown(f"**{i}. `{t.get('name', '?')}`**")
                st.json(t.get("args") or {})
                preview = t.get("resultado_preview") or ""
                if preview:
                    st.text(preview[:400] + ("..." if len(preview) > 400 else ""))

    fuentes_base = msg.get("fuentes_base") or []
    fuentes_web = msg.get("fuentes_web") or []

    with st.expander("📄 Fuentes (base y web)", expanded=bool(fuentes_base or fuentes_web)):
        if not fuentes_base and not fuentes_web:
            st.info("No se recuperaron fragmentos de la base ni resultados web.")
        if fuentes_base:
            st.markdown("#### Base de conocimientos")
            for i, f in enumerate(fuentes_base, 1):
                st.markdown(
                    f"**{i}.** `{f['documento']}` · sección `{f['seccion']}`"
                )
                if f.get("contenido"):
                    st.text(
                        f["contenido"][:400]
                        + ("..." if len(f.get("contenido", "")) > 400 else "")
                    )
        if fuentes_web:
            st.markdown("#### Web (sitios permitidos)")
            for i, f in enumerate(fuentes_web, 1):
                st.markdown(f"**{i}.** [{f['titulo']}]({f['url']})")
                st.caption(f"[web: {f['url']}]")
                if f.get("extracto"):
                    st.text(f["extracto"][:300])


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_respuesta_asistente(msg)
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Input del usuario
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Escribe tu pregunta sobre Git..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("El agente está decidiendo qué tools usar..."):
            try:
                out = responder(
                    prompt,
                    llm_con_tools=llm_con_tools,
                    tools_por_nombre=tools_por_nombre,
                    verbose=False,
                )
                respuesta = out.get("respuesta") or ""
                tools_usadas = out.get("tools_usadas") or []
                fuentes_base, fuentes_web = _separar_fuentes(tools_usadas)

                if "No encuentro esa información" in respuesta:
                    st.warning(respuesta)
                else:
                    st.markdown(respuesta)

                n_llm = out.get("num_llamadas_llm")
                tiempo = out.get("tiempo_segundos")
                meta_parts = []
                if n_llm is not None:
                    meta_parts.append(f"Llamadas LLM: {n_llm}")
                if tiempo is not None:
                    meta_parts.append(f"Tiempo: {tiempo}s")
                meta_parts.append(
                    f"Tools: {len(tools_usadas)}" if tools_usadas else "Sin tools"
                )
                st.caption(" · ".join(meta_parts))

                with st.expander(
                    "🛠️ Tools usadas y argumentos", expanded=bool(tools_usadas)
                ):
                    if not tools_usadas:
                        st.info(
                            "El agente no llamó ninguna tool (saludo o respuesta directa)."
                        )
                    else:
                        for i, t in enumerate(tools_usadas, 1):
                            st.markdown(f"**{i}. `{t.get('name', '?')}`**")
                            st.json(t.get("args") or {})
                            preview = t.get("resultado_preview") or ""
                            if preview:
                                st.text(
                                    preview[:400]
                                    + ("..." if len(preview) > 400 else "")
                                )

                with st.expander(
                    "📄 Fuentes (base y web)",
                    expanded=bool(fuentes_base or fuentes_web),
                ):
                    if not fuentes_base and not fuentes_web:
                        st.info(
                            "No se recuperaron fragmentos de la base ni resultados web."
                        )
                    if fuentes_base:
                        st.markdown("#### Base de conocimientos")
                        for i, f in enumerate(fuentes_base, 1):
                            st.markdown(
                                f"**{i}.** `{f['documento']}` · sección `{f['seccion']}`"
                            )
                            if f.get("contenido"):
                                st.text(
                                    f["contenido"][:400]
                                    + (
                                        "..."
                                        if len(f.get("contenido", "")) > 400
                                        else ""
                                    )
                                )
                    if fuentes_web:
                        st.markdown("#### Web (sitios permitidos)")
                        for i, f in enumerate(fuentes_web, 1):
                            st.markdown(
                                f"**{i}.** [{f['titulo']}]({f['url']})"
                            )
                            st.caption(f"[web: {f['url']}]")
                            if f.get("extracto"):
                                st.text(f["extracto"][:300])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": respuesta,
                        "tools_usadas": [
                            {
                                "name": t.get("name"),
                                "args": t.get("args"),
                                "resultado_preview": t.get("resultado_preview"),
                                "resultado": t.get("resultado"),
                            }
                            for t in tools_usadas
                        ],
                        "fuentes_base": fuentes_base,
                        "fuentes_web": fuentes_web,
                        "num_llamadas_llm": n_llm,
                        "tiempo_segundos": tiempo,
                    }
                )
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Error: {e}",
                        "tools_usadas": [],
                        "fuentes_base": [],
                        "fuentes_web": [],
                    }
                )