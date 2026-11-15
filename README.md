# SoporteGit — Agente RAG con herramientas (Avance 3)

**Autores:** Leisson Arley Murillo Alvarez · Milen Dayana Herrera Delgado  
**Institución:** Fundación Universitaria Konrad Lorenz  

---

## Descripción

Sistema de IA tipo **Analista de Soporte Técnico** que responde preguntas sobre manuales de Git. En el Avance 3 el pipeline RAG fijo se convierte en un **agente con tools**: el LLM decide cuándo consultar la base de conocimientos, cuándo ejecutar una tarea de dominio y cuándo buscar información pública en la web.

Los documentos fuente **nunca** se envían completos a servicios externos: solo se recuperan y envían al LLM los fragmentos necesarios para cada consulta. La base no se expone al buscador web; solo sale la consulta del usuario.

**Stack:**

| Componente | Tecnología |
|---|---|
| Embeddings | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) — local, CPU |
| Vector store | ChromaDB (persistente, similitud coseno) |
| LLM | Groq API (tool calling) |
| Framework | LangChain (`bind_tools` + bucle manual) + Ragas |
| Búsqueda web | Tavily (dominios fijos) |
| Interfaz | Streamlit |

---

## Arquitectura del agente

<img width="2297" height="1489" alt="mermaid-diagram-2026-11-14-184812" src="https://github.com/user-attachments/assets/4448bd60-cbab-4c40-a409-c391263ace31" />

**Qué sale del sistema**

| Información | Destino |
|-------------|---------|
| Fragmentos recuperados de la base | Solo al LLM (Groq), nunca al buscador web |
| Consulta del usuario (si aplica web) | Tavily |
| Corpus completo | No sale: permanece en Chroma local |

---

## Tools del agente

| Tool | Tipo | Qué hace y cuándo se usa | Argumentos | Límites en el código |
|------|------|--------------------------|------------|----------------------|
| `buscar_en_base` | Búsqueda | Manuales Git internos. **Siempre primero** en preguntas técnicas. | `consulta: str` | `MAX_FRAGMENTS=4`, `MAX_CHARS_PER_FRAGMENT=600` |
| `verificar_comando_destructivo` | Tarea dominio | Regla exacta: si un comando es destructivo (reset --hard, clean -fd, push --force, etc.). | `comando: str` | Reglas fijas en código |
| `generar_referencia_apa` | Tarea dominio | Cita APA 7 de un documento de la base. | `documento: str`, `seccion: str=""` | Catálogo fijo de documentos |
| `buscar_en_web` | Web | Info pública/actual **solo si la base no responde**. | `consulta: str` | `MAX_WEB_RESULTS=3`, `ALLOWED_DOMAINS` fijo |

**Bucle:** implementación manual con `bind_tools` + `ToolMessage`.  
**Límite de iteraciones:** `MAX_ITERACIONES = 5` por pregunta.  
**Registro de uso:** `logs/tool_usage.jsonl` (tools, argumentos, nº de llamadas LLM, tiempo).

Prueba de tools sin LLM:

```bash
python -m src.tools
```

Prueba del agente:

```bash
python -m src.agent
```

---

## Política de búsqueda web

| Aspecto | Detalle |
|---------|---------|
| **Cuándo se usa** | Solo si `buscar_en_base` no devolvió información suficiente **y** la pregunta trata de información pública o actual (versión reciente, documentación oficial, best practices actuales). |
| **Servicio** | Tavily |
| **Máximo de resultados** | 3 |
| **Campos devueltos** | Título, URL, extracto corto, marcados como `[web: URL]` |
| **Quién controla los dominios** | El código (`ALLOWED_DOMAINS`). El LLM no puede cambiar la lista. |

**Sitios permitidos y por qué:**

| Sitio | Por qué es confiable para el dominio |
|-------|--------------------------------------|
| `git-scm.com` | Documentación oficial de Git |
| `docs.github.com` | Documentación oficial de GitHub |
| `stackoverflow.com` | Comunidad técnica de referencia |
| `atlassian.com` | Guías de Git / Bitbucket bien mantenidas |
| `learngitbranching.js.org` | Recurso educativo de flujos de ramas |

---

## System Prompt (cambios frente al Avance 2)

- **Ya no incluye el contexto** dentro del prompt: el contexto llega solo vía tools.
- Indica **cuándo usar cada tool** y la política “primero la base”.
- Exige responder **solo con lo que devuelvan las tools** y citar fuentes (base o `[web: URL]`).
- Mantiene la frase exacta cuando no hay información:  
  `"No encuentro esa información en la base de conocimiento proporcionada."`
- Saludos y charla casual **sin tools**.
- Rechazo de instrucciones maliciosas (no volcar el corpus).

---

## Estructura del repositorio

```
Entrega_Final_Chat_Git/
├── app/
│   └── streamlit_app.py          # Interfaz del agente (tools + fuentes)
├── data/docs/                    # Corpus (manuales Git)
├── eval/
│   ├── eval_dataset.json         # 18 preguntas + ground truth
│   ├── run_ragas.py              # Ragas pipeline fijo (Avance 2)
│   ├── run_ragas_agente.py       # Ragas fijo vs agente + costo
│   ├── casos_comportamiento.py   # 7 casos de comportamiento
│   └── results/                  # Resultados JSON/CSV
├── src/
│   ├── ingest.py                 # Ingesta + chunking + Chroma
│   ├── rag_chain.py              # Pipeline fijo (reutilizado)
│   ├── tools.py                  # Tools del agente
│   └── agent.py                  # Bucle del agente + registro
├── logs/
│   └── tool_usage.jsonl          # Registro de uso de tools
├── chroma_db/                    # Índice vectorial
├── requirements.txt
├── .env                          # No versionar (GROQ_API_KEY, TAVILY_API_KEY)
└── README.md
```

---

## Instalación y ejecución

```bash
# 1. Clonar
git clone https://github.com/ProyectoDesarrollo-1/Entrega_Final_Chat_Git.git
cd Entrega_Final_Chat_Git

# 2. Entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Dependencias
pip install -r requirements.txt

# 4. Variables de entorno
# Crear .env con:
#   GROQ_API_KEY=gsk_...
#   TAVILY_API_KEY=tvly_...
#   GROQ_MODEL=...   (opcional)

# 5. Índice vectorial (solo la primera vez o si cambia el corpus)
python -m src.ingest

# 6. Probar tools sin LLM
python -m src.tools

# 7. Probar el agente en consola
python -m src.agent

# 8. Casos de comportamiento
python -m eval.casos_comportamiento

# 9. Ragas: pipeline fijo vs agente
python -m eval.run_ragas_agente --max_samples 5
python -m eval.run_ragas_agente

# 10. Interfaz de chat
streamlit run app/streamlit_app.py
```

---

## Despliegue (Streamlit Community Cloud)

1. Sube el repo a GitHub **sin** `.env`.
2. En [share.streamlit.io](https://share.streamlit.io) selecciona el repo y `app/streamlit_app.py`.
3. En **Secrets** añade:

```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly_..."
```

4. Si el índice Chroma no está en el repo, la app lo regenera al arranque (`ensure_index`).

**URL pública:**  
https://entregafinalchatgit-bqbdvtfsvb5pxlnrotkhdq.streamlit.app/

**Repositorio:**  
https://github.com/ProyectoDesarrollo-1/Entrega_Final_Chat_Git.git

---

## Evaluación

### Casos de comportamiento (7)

| # | Tipo | Comportamiento esperado |
|---|------|-------------------------|
| 1 | Saludo | Sin tools |
| 2 | Pregunta compuesta | `buscar_en_base` (una o más veces), citas |
| 3 | Tarea dominio | `verificar_comando_destructivo` |
| 4 | Tarea dominio | `generar_referencia_apa` |
| 5 | Info pública/actual | Base primero; luego `buscar_en_web` en dominios permitidos |
| 6 | Pregunta de la base | Solo base; **no** web |
| 7 | Instrucción maliciosa | No vuelca el corpus ni rompe reglas |

Resultados en `eval/results/casos_comportamiento_*.json`.

### Ragas (pipeline fijo vs agente)

Mismo `eval_dataset.json` (≥15 preguntas). Métricas: faithfulness, answer_relevancy, context_precision, context_recall.  
Los `retrieved_contexts` del agente son los fragmentos que devolvieron sus tools.

```bash
python -m eval.run_ragas_agente
```

Comparación de costo (llamadas LLM, tokens aprox., tiempo) en la salida del script y en `eval/results/comparacion_avance2_vs_agente.json`.

---

## Interfaz desplegada

La UI usa el **agente** (no el pipeline fijo):

- Historial de conversación
- Expander **Tools usadas y argumentos**
- Expander **Fuentes** separadas: base (documento + sección) y web (título + URL)
- Mensaje explícito cuando no hay información en base ni en sitios permitidos

---

*Avance 3 — Agente RAG con herramientas*  
*Desarrollo de Aplicaciones con IA · Fundación Universitaria Konrad Lorenz*
