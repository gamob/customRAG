"""
generate_evals.py — Phase 2 (RAG inference) + Phase 3 (judge & report) + Faithfulness Check.

Questions are loaded from evaluation/eval_test.jsonl, eval_train.jsonl, or eval_valid.jsonl
(created by merge_eval_datasets.py).

This file is now a thin orchestrator that imports Phase 2 and Phase 3 logic from
separate modules under src/eval.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time

from . import phase2, phase3, utils
from .phase3 import run_judge_and_report
from .utils import configure_logging, get_data_file, EVAL_DIR, build_ollama_llm
from ..core.brain_service import Brain
from langchain_ollama import OllamaLLM

PHASE3_TIMEOUT = 120

logger = logging.getLogger(__name__)


def _stop_ollama_model(model_name=None):
    try:
        if model_name:
            logger.info(f"🛑 Stopping Ollama model: {model_name}")
            subprocess.run(["ollama", "stop", model_name], timeout=10, capture_output=True)
        else:
            logger.info(f"🛑 Stopping all Ollama models")
            subprocess.run(["ollama", "stop"], timeout=10, capture_output=True)
        time.sleep(1)
        logger.info("   Model stopped successfully")
    except Exception as e:
        logger.warning(f"   Warning: Could not stop model: {e}")


def _switch_ollama_model(from_model, to_model):
    logger.info(f"🔄 Switching models: {from_model} → {to_model}")
    _stop_ollama_model(from_model)
    logger.info(f"✅ Ready to load: {to_model}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results using RAG inference and judging"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "train", "valid"],
        default="test",
        help="Which evaluation dataset to use (default: test)"
    )
    configure_logging()

    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Randomly sample N questions (default: 5 questions; set 0 for all questions)"
    )
    parser.add_argument(
        "--phase2-workers",
        type=int,
        default=phase2.PHASE2_WORKERS,
        help=f"Number of worker threads for phase 2 search/answering (default: {phase2.PHASE2_WORKERS})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more verbose status output"
    )
    parser.add_argument(
        "--bug-only",
        action="store_true",
        help="Only print bugged/failed question lines; suppress normal progress output."
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default="qwen_35b_a3b_MoE",
        help="Answer generation model name to use with Ollama (default: qwen_35b_a3b_MoE). Used in Phase 2."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="local-llama3.1",
        help="Judge LLM model name to use with Ollama (default: local-llama3.1). Used in Phase 3. Script will auto-switch from answer-model."
    )
    parser.add_argument(
        "--judge-num-predict",
        type=int,
        default=128,
        help="Max tokens to predict for judge LLM (lower = faster; default: 128)"
    )
    parser.add_argument(
        "--judge-num-ctx",
        type=int,
        default=1024,
        help="Context window for judge LLM (default: 1024)"
    )
    parser.add_argument(
        "--phase3-timeout",
        type=int,
        default=PHASE3_TIMEOUT,
        help=f"Phase 3 timeout in seconds (default: {PHASE3_TIMEOUT})"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=phase3.DEFAULT_PHASE3_METRICS,
        help="Comma-separated list of evaluation metrics to run."
    )
    parser.add_argument(
        "--stub-judge",
        action="store_true",
        help="Use a fast stub judge that synthesizes metric scores (no LLM calls)."
    )
    parser.add_argument(
        "--per-question-timing",
        action="store_true",
        help="Measure and log per-question evaluation time (may slow overall run)."
    )
    parser.add_argument(
        "--test-phase-3",
        action="store_true",
        dest="test_phase_3",
        help="Generate or reuse a single Phase 2 sample and run only Phase 3 for debug testing."
    )
    parser.add_argument(
        "--require-local-judge",
        action="store_true",
        help="If set, error out if the configured judge LLM cannot be constructed."
    )
    args = parser.parse_args()

    if args.bug_only:
        configure_logging(logging.ERROR)
        utils.BUG_ONLY = True
        logger.error("🔎 BUG-ONLY mode active: showing only error diagnostics and critical failure lines")
    elif args.debug:
        configure_logging(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if args.phase2_workers < 1:
        args.phase2_workers = 1

    if args.test_phase_3:
        args.sample = 1

    phase2.PHASE2_WORKERS = args.phase2_workers

    data_file = get_data_file(args.dataset)
    default_rag_answers = os.path.join(EVAL_DIR, f"rag_responses_{args.dataset}.jsonl")
    phase3_sample_file = os.path.join(EVAL_DIR, f"rag_responses_{args.dataset}_phase3_sample.jsonl")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phase3_debug_log = os.path.join(script_dir, f"phase3_debug_{args.dataset}.log")
    final_report = os.path.join(EVAL_DIR, f"final_evaluation_{args.dataset}.jsonl")
    os.makedirs(EVAL_DIR, exist_ok=True)

    if not os.path.exists(data_file):
        logger.error(f"❌ Dataset file not found: {data_file}")
        logger.error("   Make sure you've run merge_eval_datasets.py first to create the split datasets.")
        sys.exit(1)

    if args.test_phase_3:
        if os.path.exists(phase3_sample_file) and os.path.getsize(phase3_sample_file) > 0:
            rag_answers = phase3_sample_file
            phase2_required = False
            logger.info(f"🔁 Reusing saved Phase 3 sample: {phase3_sample_file}")
        elif os.path.exists(default_rag_answers) and os.path.getsize(default_rag_answers) > 0:
            logger.info(f"🔁 Found existing Phase 2 output; extracting one sample for Phase 3 debug.")
            with open(default_rag_answers, "r", encoding="utf-8") as fin:
                sample_item = None
                for line in fin:
                    if line.strip():
                        sample_item = line
                        break
            if sample_item is None:
                logger.error(f"❌ No valid entries found in {default_rag_answers} to sample from.")
                sys.exit(1)
            with open(phase3_sample_file, "w", encoding="utf-8") as fout:
                fout.write(sample_item)
            rag_answers = phase3_sample_file
            phase2_required = False
            logger.info(f"✅ Saved Phase 3 sample file: {phase3_sample_file}")
        else:
            rag_answers = phase3_sample_file
            phase2_required = True
            logger.info(f"🔧 No saved sample found; Phase 2 will generate one sample and save it to {phase3_sample_file}")

        with open(phase3_debug_log, "w", encoding="utf-8"):
            pass
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(phase3_debug_log, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        root_logger.addHandler(file_handler)
        logger.info(f"📄 Phase 3 debug output will be written to: {phase3_debug_log}")
        debug_phase3 = True
    else:
        rag_answers = default_rag_answers
        phase2_required = True
        debug_phase3 = False

    logger.info(f"📁 Dataset file: {data_file}")
    logger.info(f"🐘 Process ID: {os.getpid()}")
    logger.info(f"🧮 CPU count: {os.cpu_count() or 1}; Phase 2 workers: {args.phase2_workers}")
    logger.info("⏱️  Counting questions in dataset file...")
    q_count = sum(1 for l in open(data_file, encoding="utf-8") if l.strip())
    logger.info(f"📋 Loaded {q_count} questions from '{data_file}' (dataset: {args.dataset})")

    brain = Brain()
    logger.info(f"🧠 Brain indices present? {brain.is_built()}")
    if brain.is_built():
        logger.info("✅ Existing indices detected; loading brain directly.")
        brain.load()
    else:
        logger.info("🔄 Syncing indices...")
        brain.sync_indices()

    if phase2_required:
        logger.info("🚀 Starting Phase 2 (RAG inference). This may take a while, especially on the first question.")
        phase2.run_rag_inference(brain, args, data_file, rag_answers)
    else:
        logger.info("🔁 Skipping Phase 2; using saved Phase 3 sample for Phase 3 debug.")

    if args.answer_model != args.judge_model:
        logger.info("\n" + "=" * 60)
        _switch_ollama_model(args.answer_model, args.judge_model)
        logger.info("=" * 60 + "\n")

    logger.info("🚀 Starting Phase 3 (Judge & report).")
    logger.info(f"📋 Using judge model: {args.judge_model}")
    judge_llm = None
    try:
        judge_llm = OllamaLLM(
            model=args.judge_model,
            base_url="http://localhost:11434",
            temperature=0.0,
            num_ctx=args.judge_num_ctx,
            keep_alive=-1,
            num_predict=args.judge_num_predict,
            stop=[
                "<|eot_id|>",
                "<|end_header_id|>",
                "\n\n",
                "---",
                "Note:", "Important:",
                "Q:", "Question:",
                "Đó là", "Do đó",
            ]
        )
        logger.info("🔧 Judge LLM constructed successfully")
        log_method = logger.info if args.test_phase_3 else logger.debug
        log_method(
            "🔧 Judge LLM internals: model=%s num_ctx=%s num_predict=%s temperature=%s stop=%s",
            getattr(judge_llm, 'model', None),
            getattr(judge_llm, 'num_ctx', None),
            getattr(judge_llm, 'num_predict', None),
            getattr(judge_llm, 'temperature', None),
            getattr(judge_llm, 'stop', None),
        )
        log_method("🔧 Judge LLM repr: %s", repr(judge_llm))
    except Exception as e:
        logger.error(f"Failed to construct judge LLM '{args.judge_model}': {e}")
        if args.require_local_judge:
            logger.error("--require-local-judge is set; aborting.")
            sys.exit(1)
        if not args.stub_judge:
            logger.error("Judge LLM unavailable and --stub-judge not set. Aborting.")
            sys.exit(1)
        logger.warning("Proceeding with stub judge due to judge LLM construction failure.")

    run_judge_and_report(
        rag_answers,
        final_report,
        judge_llm,
        phase3_timeout=args.phase3_timeout,
        selected_metrics=args.metrics,
        stub_judge=args.stub_judge,
        per_question_timing=args.per_question_timing,
        debug_phase3=args.test_phase_3,
    )

    if os.path.exists(final_report):
        with open(final_report, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                try:
                    first_result = json.loads(lines[0])
                    has_missing_metrics = any(
                        first_result.get(m) is None
                        for m in ['phoenix_faithfulness', 'phoenix_answer_accuracy', 'phoenix_context_precision']
                    )
                    if has_missing_metrics:
                        logger.warning("\n⚠️  Some evaluation metrics have None values (label not matched).")
                        logger.warning("   This suggests your judge model is not returning a recognised label.")
                        logger.warning(f"   Try a different model:")
                        logger.warning(f"   python -m src.eval.generate_evals --dataset {args.dataset} --judge-model neural-chat")
                        logger.warning(f"   or use stub judge for testing:")
                        logger.warning(f"   python -m src.eval.generate_evals --dataset {args.dataset} --stub-judge")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
