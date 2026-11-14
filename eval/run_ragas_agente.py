"""
Evaluación Ragas del AGENTE (Avance 3) + comparación con el pipeline fijo (Avance 2).

- Usa el mismo eval_dataset.json del Avance 2 (mínimo 15 preguntas).
- Los retrieved_contexts del agente son los fragmentos que devolvieron sus tools.
- Reporta faithfulness, answer_relevancy, context_precision, context_recall.
- Compara métricas y costo (llamadas LLM, tokens aprox, tiempo) vs pipeline fijo.

Uso:
    python -m eval.run_ragas_agente
    python -m eval.run_ragas_agente --max_samples 5   # prueba rápida
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest.mock as mock
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# Evitar errores de import de vertexai en algunos entornos
sys.modules["langchain_community.chat_models.vertexai"] = mock.MagicMock()
sys.modules["langchain_community.llms.vertexai"] = mock.MagicMock()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.rag_chain import get_llm, get_embeddings, rag_pipeline, parse_json_safe, TOP_K
from src.agent import responder, get_llm_con_tools

EVAL_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_eval_set():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pipeline fijo (Avance 2) — para comparación
# ---------------------------------------------------------------------------
def run_pipeline_fijo(items, top_k: int = TOP_K):
    questions, answers, contexts, ground_truths = [], [], [], []
    tiempos, n_llm = [], []

    for i, item in enumerate(items):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [fijo {i+1}/{len(items)}] {q[:55]}...")

        t0 = time.time()
        try:
            result = rag_pipeline(q, k=top_k)
            raw = result["respuesta"]
            parsed = parse_json_safe(raw)

            if parsed.get("diagnostico", "").startswith("No encuentro"):
                answer_text = parsed["diagnostico"]
            else:
                parts = [parsed.get("diagnostico", "")]
                if parsed.get("pasos_solucion"):
                    parts.append("Pasos: " + "; ".join(parsed["pasos_solucion"]))
                if parsed.get("comandos"):
                    parts.append("Comandos: " + "; ".join(parsed["comandos"]))
                answer_text = " | ".join(p for p in parts if p)

            ctx_list = [d.page_content for d in result["fragmentos"]]
            # Pipeline fijo: 1 llamada de recuperación + 1 llamada LLM de generación
            n_llm.append(1)
        except Exception as e:
            print(f"    ERROR: {e}")
            answer_text = "Error"
            ctx_list = []
            n_llm.append(0)

        tiempos.append(time.time() - t0)
        questions.append(q)
        answers.append(answer_text)
        contexts.append(ctx_list)
        ground_truths.append(gt)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "tiempos": tiempos,
        "n_llm": n_llm,
    }


# ---------------------------------------------------------------------------
# Agente (Avance 3)
# ---------------------------------------------------------------------------
def run_agente(items):
    questions, answers, contexts, ground_truths = [], [], [], []
    tiempos, n_llm, tools_por_pregunta = [], [], []

    llm_con_tools, tools_por_nombre = get_llm_con_tools()

    for i, item in enumerate(items):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [agente {i+1}/{len(items)}] {q[:55]}...")

        try:
            out = responder(
                q,
                llm_con_tools=llm_con_tools,
                tools_por_nombre=tools_por_nombre,
                verbose=False,
            )
            answer_text = out["respuesta"]
            tools_usadas = out["tools_usadas"]

            # Contextos = resultados de las tools de búsqueda
            ctx_list = []
            for t in tools_usadas:
                if t["name"] in ("buscar_en_base", "buscar_en_web"):
                    # Preferir resultado completo si existe; si no, el preview
                    texto = t.get("resultado") or t.get("resultado_preview", "")
                    if texto:
                        ctx_list.append(texto)
            if not ctx_list:
                ctx_list = [""]

            n_llm.append(out["num_llamadas_llm"])
            tiempos.append(out["tiempo_segundos"])
            tools_por_pregunta.append([t["name"] for t in tools_usadas])
        except Exception as e:
            print(f"    ERROR: {e}")
            answer_text = "Error"
            ctx_list = [""]
            n_llm.append(0)
            tiempos.append(0.0)
            tools_por_pregunta.append([])

        questions.append(q)
        answers.append(answer_text)
        contexts.append(ctx_list)
        ground_truths.append(gt)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "tiempos": tiempos,
        "n_llm": n_llm,
        "tools": tools_por_pregunta,
    }


# ---------------------------------------------------------------------------
# Ragas
# ---------------------------------------------------------------------------
def evaluar_ragas(dataset_dict, run_name: str):
    """Ejecuta Ragas y guarda resultados. Solo usa columnas estándar."""
    ds_dict = {
        "question": dataset_dict["question"],
        "answer": dataset_dict["answer"],
        "contexts": dataset_dict["contexts"],
        "ground_truth": dataset_dict["ground_truth"],
    }
    ds = Dataset.from_dict(ds_dict)

    llm = get_llm()
    embeddings = get_embeddings()
    llm_juez = LangchainLLMWrapper(llm)
    emb_juez = LangchainEmbeddingsWrapper(embeddings)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"\n  Ejecutando Ragas ({run_name})...")
    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm_juez,
        embeddings=emb_juez,
    )

    df = result.to_pandas()
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(ds_dict["question"]),
        "faithfulness": _safe_mean(df, "faithfulness"),
        "answer_relevancy": _safe_mean(df, "answer_relevancy"),
        "context_precision": _safe_mean(df, "context_precision"),
        "context_recall": _safe_mean(df, "context_recall"),
    }

    out_csv = RESULTS_DIR / f"ragas_{run_name}.csv"
    out_json = RESULTS_DIR / f"ragas_{run_name}_summary.json"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, df


def _safe_mean(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) == 0:
        return None
    return float(s.mean())


def _promedio(lista):
    return round(sum(lista) / len(lista), 3) if lista else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Limitar nº de preguntas (prueba rápida)")
    parser.add_argument("--skip_fijo", action="store_true", help="Solo evaluar el agente")
    parser.add_argument("--skip_agente", action="store_true", help="Solo evaluar el pipeline fijo")
    args = parser.parse_args()

    items = load_eval_set()
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"Preguntas de evaluación: {len(items)}")

    summary_fijo = None
    summary_agente = None
    costo_fijo = {}
    costo_agente = {}

    # --- Pipeline fijo ---
    if not args.skip_fijo:
        print("\n" + "=" * 60)
        print("1) Pipeline fijo (Avance 2)")
        print("=" * 60)
        data_fijo = run_pipeline_fijo(items)
        summary_fijo, _ = evaluar_ragas(data_fijo, run_name="avance2_fijo")
        costo_fijo = {
            "llamadas_llm_promedio": _promedio(data_fijo["n_llm"]),
            "tiempo_promedio_s": _promedio(data_fijo["tiempos"]),
            # Tokens: aproximación gruesa (chars/4) de pregunta+respuesta
            "tokens_aprox_promedio": _promedio([
                (len(q) + len(a)) / 4
                for q, a in zip(data_fijo["question"], data_fijo["answer"])
            ]),
        }
        print("\n  Métricas fijo:", {k: v for k, v in summary_fijo.items() if k not in ("run_name", "timestamp")})

    # --- Agente ---
    if not args.skip_agente:
        print("\n" + "=" * 60)
        print("2) Agente (Avance 3)")
        print("=" * 60)
        data_agente = run_agente(items)
        summary_agente, _ = evaluar_ragas(data_agente, run_name="avance3_agente")
        costo_agente = {
            "llamadas_llm_promedio": _promedio(data_agente["n_llm"]),
            "tiempo_promedio_s": _promedio(data_agente["tiempos"]),
            "tokens_aprox_promedio": _promedio([
                (len(q) + len(a)) / 4
                for q, a in zip(data_agente["question"], data_agente["answer"])
            ]),
        }
        print("\n  Métricas agente:", {k: v for k, v in summary_agente.items() if k not in ("run_name", "timestamp")})

    # --- Tabla comparativa ---
    print("\n" + "=" * 70)
    print("TABLA RAGAS: Avance 2 (pipeline fijo) vs Avance 3 (agente)")
    print("=" * 70)

    metricas = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"{'Métrica':<22} {'Avance 2 (fijo)':>16} {'Avance 3 (agente)':>18} {'Diferencia':>12}")
    print("-" * 70)

    if summary_fijo and summary_agente:
        for m in metricas:
            v2 = summary_fijo.get(m)
            v3 = summary_agente.get(m)
            if v2 is not None and v3 is not None:
                diff = v3 - v2
                print(f"{m:<22} {v2:>16.4f} {v3:>18.4f} {diff:>+12.4f}")
            else:
                print(f"{m:<22} {str(v2):>16} {str(v3):>18} {'—':>12}")
        print(f"\n{'Nº preguntas':<22} {summary_fijo['n_samples']:>16} {summary_agente['n_samples']:>18}")

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE COSTO (promedio por pregunta)")
    print("=" * 70)
    print(f"{'Medida':<28} {'Pipeline fijo':>14} {'Agente':>14}")
    print("-" * 58)
    if costo_fijo and costo_agente:
        print(f"{'Llamadas al LLM':<28} {costo_fijo['llamadas_llm_promedio']:>14.2f} {costo_agente['llamadas_llm_promedio']:>14.2f}")
        print(f"{'Tokens (aprox)':<28} {costo_fijo['tokens_aprox_promedio']:>14.1f} {costo_agente['tokens_aprox_promedio']:>14.1f}")
        print(f"{'Tiempo de respuesta (s)':<28} {costo_fijo['tiempo_promedio_s']:>14.2f} {costo_agente['tiempo_promedio_s']:>14.2f}")

    # Guardar comparación
    comparacion = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(items),
        "ragas_fijo": summary_fijo,
        "ragas_agente": summary_agente,
        "costo_fijo": costo_fijo,
        "costo_agente": costo_agente,
    }
    out_comp = RESULTS_DIR / "comparacion_avance2_vs_agente.json"
    with open(out_comp, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, indent=2, ensure_ascii=False)
    print(f"\nComparación guardada en: {out_comp}")
    print("\nAnálisis sugerido para el PDF:")
    print("  - Qué mejoró / empeoró y a qué componente se atribuye")
    print("  - El agente gana flexibilidad (saludos, tareas, web) a costa de más llamadas LLM y tiempo")


if __name__ == "__main__":
    main()