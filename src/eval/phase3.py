"""
phase3.py — Fully local Phoenix-style evaluation using Ollama directly.

Calls judge_llm (OllamaLLM) in a thread pool — no arize-phoenix-evals,
no network calls. Safe for air-gapped / company-GPU environments.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PHASE3_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall,correctness"

# Backwards compatibility flags expected by generate_evals.py
RAGAS_AVAILABLE = False
RAGAS_IMPORT_ERROR = "Replaced by local Phoenix-style evaluator"

# Context recall variants (unused in this flow)
NonLLMContextRecall = None
context_recall = None


# ---------------------------------------------------------------------------
# Metric definitions
#
# Each metric has:
#   template  – prompt string with {column} placeholders filled per-row.
#               Available columns: question, response, contexts, expected_output
#   rails     – ordered list of accepted single-word responses from the LLM.
#   scores    – label → float mapping written to the output JSONL.
# ---------------------------------------------------------------------------
_METRIC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "faithfulness": {
        "template": (
            "You are evaluating whether the response is faithful to the provided context.\n"
            "The response is faithful if it uses only information from the context and does not invent facts.\n\n"
            "Question: {question}\n"
            "Context: {contexts}\n"
            "Response: {response}\n\n"
            "Answer with one word only: faithful or hallucinated."
        ),
        "rails": ["faithful", "hallucinated"],
        "scores": {"faithful": 1.0, "hallucinated": 0.0},
    },
    "answer_relevancy": {
        "template": (
            "You are evaluating whether the response is relevant to the question.\n\n"
            "Question: {question}\n"
            "Response: {response}\n\n"
            "Answer with one word only: relevant or irrelevant."
        ),
        "rails": ["relevant", "irrelevant"],
        "scores": {"relevant": 1.0, "irrelevant": 0.0},
    },
    "context_precision": {
        "template": (
            "You are evaluating whether the response uses only relevant information from the provided context.\n\n"
            "Question: {question}\n"
            "Context: {contexts}\n"
            "Response: {response}\n\n"
            "Answer with one word only: precise or imprecise."
        ),
        "rails": ["precise", "imprecise"],
        "scores": {"precise": 1.0, "imprecise": 0.0},
    },
    "context_recall": {
        "template": (
            "You are evaluating whether the provided context contains sufficient information to answer the question.\n\n"
            "Question: {question}\n"
            "Context: {contexts}\n"
            "Response: {response}\n\n"
            "Answer with one word only: yes or no."
        ),
        "rails": ["yes", "no"],
        "scores": {"yes": 1.0, "no": 0.0},
    },
    "correctness": {
        "template": (
            "You are evaluating whether the response is correct compared to the expected answer.\n\n"
            "Question: {question}\n"
            "Expected answer: {expected_output}\n"
            "Response: {response}\n\n"
            "Answer with one word only: correct or incorrect."
        ),
        "rails": ["correct", "incorrect"],
        "scores": {"correct": 1.0, "incorrect": 0.0},
    },
}


def _normalize_metric_field(metric_name: str) -> str:
    """Map a metric name to its phoenix_* output field name."""
    name = metric_name.strip().lower()
    if name == "correctness":
        return "phoenix_answer_accuracy"
    if name in {"contextprecision", "context_precision"}:
        return "phoenix_context_precision"
    if name in {"contextrecall", "context_recall"}:
        return "phoenix_context_recall"
    if name in {"answerrelevancy", "answer_relevance", "answer_relevancy"}:
        return "phoenix_answer_relevancy"
    if name == "faithfulness":
        return "phoenix_faithfulness"
    return f"phoenix_{name}"


def _load_phase2_items(rag_answers: str) -> List[Dict[str, Any]]:
    items = []
    with open(rag_answers, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                snippet = line.replace("\\n", " ")[:200]
                logger.warning("⚠️ Skipping malformed line %s: %s", line_no, snippet)
    return items


def _build_row(item: Dict[str, Any]) -> Dict[str, str]:
    """Flatten one Phase 2 item into the template placeholder dict."""
    return {
        "question": item.get("question", ""),
        "response": item.get("rag_answer", ""),
        "contexts": "\n\n".join(item.get("contexts", [])),
        "expected_output": item.get("answer", ""),
    }


def _match_label(raw_output: str, rails: List[str]) -> Optional[str]:
    """
    Try to find a rail label in the LLM's raw output.
    Strips whitespace/punctuation and matches case-insensitively.
    Returns the matched rail string, or None if nothing matched.
    """
    cleaned = raw_output.strip().lower().rstrip(".,!?;:")
    # Exact match first
    if cleaned in rails:
        return cleaned
    # Prefix match (model may output "faithful." or "faithful\n...")
    for rail in rails:
        if cleaned.startswith(rail):
            return rail
    # Substring match as last resort
    for rail in rails:
        if rail in cleaned:
            return rail
    return None


def _call_judge(judge_llm: Any, prompt: str) -> str:
    """Invoke judge_llm and return the raw string output."""
    result = judge_llm.invoke(prompt)
    if isinstance(result, str):
        return result
    # LangChain AIMessage / BaseMessage fallback
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _run_metric(
    judge_llm: Any,
    metric_name: str,
    items: List[Dict[str, Any]],
    concurrency: int = 5,
) -> List[Optional[float]]:
    """
    Run one metric against all items using a thread pool.
    Returns a list of float scores (or None for unresolved) aligned to items.
    """
    config = _METRIC_CONFIGS[metric_name]
    rails = config["rails"]
    scores_map = config["scores"]
    template = config["template"]

    results: List[Optional[float]] = [None] * len(items)

    def _grade_one(idx_item):
        idx, item = idx_item
        row = _build_row(item)
        try:
            prompt = template.format(**row)
        except KeyError as exc:
            logger.warning("  ⚠️ Template key error for item %d metric '%s': %s", idx, metric_name, exc)
            return idx, None

        try:
            raw = _call_judge(judge_llm, prompt)
        except Exception as exc:
            logger.warning("  ⚠️ Judge call failed for item %d metric '%s': %s", idx, metric_name, exc)
            return idx, None

        label = _match_label(raw, rails)
        if label is None:
            logger.debug(
                "  ⚠️ No rail matched for item %d metric '%s'. Raw output: %r",
                idx, metric_name, raw[:120],
            )
            return idx, None
        return idx, scores_map[label]

    logger.info("  📐 Running metric: %s (%d items, concurrency=%d)", metric_name, len(items), concurrency)
    resolved = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_grade_one, (i, item)): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            try:
                idx, score = future.result()
                results[idx] = score
                if score is not None:
                    resolved += 1
            except Exception as exc:
                logger.warning("  ⚠️ Unexpected error in metric '%s': %s", metric_name, exc)

    unresolved = len(items) - resolved
    logger.info("  ✅ %s done — %d scored, %d unresolved", metric_name, resolved, unresolved)
    if unresolved:
        logger.warning(
            "  ⚠️  %d item(s) returned an unexpected label for '%s'. Expected one of: %s",
            unresolved, metric_name, rails,
        )
    return results


def run_judge_and_report(
    rag_answers,
    final_report,
    judge_llm,
    phase3_timeout=120,
    selected_metrics=DEFAULT_PHASE3_METRICS,
    stub_judge=False,
    per_question_timing=False,
    translator_llm=None,
    debug_phase3=False,
):
    logger.info("⚖️ Phase 3: Grading locally via Ollama judge...")

    if debug_phase3:
        logger.info("🧪 Phase 3 debug enabled")

    if per_question_timing:
        logger.warning(
            "⚠️ Per-question timing is not supported in batch evaluation; "
            "this flag will be ignored."
        )

    # Parse and validate requested metrics up front (needed for stub too)
    requested = [m.strip().lower() for m in selected_metrics.split(",") if m.strip()]
    if not requested:
        requested = DEFAULT_PHASE3_METRICS.split(",")

    valid_metrics = [m for m in requested if m in _METRIC_CONFIGS]
    unknown = set(requested) - set(valid_metrics)
    for u in unknown:
        logger.warning("⚠️ Unknown metric '%s'. Skipping.", u)

    if not valid_metrics:
        logger.error("❌ No valid metrics to evaluate. Check --metrics.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load Phase 2 output
    # ------------------------------------------------------------------
    items = _load_phase2_items(rag_answers)
    if not items:
        logger.error("❌ No valid Phase 2 items found for evaluation")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stub mode — synthetic scores, no LLM calls
    # ------------------------------------------------------------------
    if stub_judge:
        logger.warning("⚠️ Running stub judge: synthetic scores only")
        with open(final_report, "w", encoding="utf-8") as fout:
            for item in items:
                item_copy = item.copy()
                for metric_name in valid_metrics:
                    item_copy[_normalize_metric_field(metric_name)] = 0.75
                fout.write(json.dumps(item_copy, ensure_ascii=False) + "\n")
        logger.info("✅ Stub Phase 3 complete: %s", final_report)
        return

    if judge_llm is None:
        logger.error(
            "❌ Judge LLM is required for Phase 3 evaluation when not using --stub-judge"
        )
        sys.exit(1)

    logger.info("📊 Metrics to evaluate: %s", ", ".join(valid_metrics))
    logger.info("📊 Items to evaluate: %d", len(items))

    # ------------------------------------------------------------------
    # Run each metric via direct Ollama calls (no internet)
    # ------------------------------------------------------------------
    score_columns: Dict[str, List[Optional[float]]] = {}
    for metric_name in valid_metrics:
        score_columns[metric_name] = _run_metric(judge_llm, metric_name, items)

    # ------------------------------------------------------------------
    # Merge scores back onto original items and write output
    # ------------------------------------------------------------------
    output_rows = []
    for i, item in enumerate(items):
        row = item.copy()
        for metric_name, scores in score_columns.items():
            field = _normalize_metric_field(metric_name)
            raw = scores[i] if i < len(scores) else None
            row[field] = raw  # already float or None
        output_rows.append(row)

    with open(final_report, "w", encoding="utf-8") as fout:
        for row in output_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("✅ Phase 3 complete: %s", final_report)

    if output_rows:
        sample = output_rows[0]
        logger.info("Sample metrics from first result:")
        for metric_name in valid_metrics:
            field = _normalize_metric_field(metric_name)
            logger.info("  %s: %s", field, sample.get(field))
