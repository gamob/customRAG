#!/usr/bin/env python3
"""
import_manual_eval.py

Import manually evaluated results from ChatGPT/Gemini and merge them with Phase 2 output
to create the final evaluation report.

Usage:
  python -m src.eval.import_manual_eval --dataset test --scores-file scores.json

This reads the JSON scores and merges them with Phase 2 RAG responses to create
evaluation/final_evaluation_<dataset>.jsonl
"""
import argparse
import json
import logging
import os
import sys

from .utils import get_data_file, EVAL_DIR, configure_logging

logger = logging.getLogger(__name__)


def import_manual_eval(rag_responses_file, scores_file, output_file):
    """
    Import manually evaluated scores and merge with Phase 2 RAG responses.
    
    Args:
        rag_responses_file: Path to Phase 2 output (rag_responses_*.jsonl)
        scores_file: Path to JSON scores file from ChatGPT/Gemini
        output_file: Path to final evaluation report
    
    Returns:
        bool: True if successful
    """
    if not os.path.exists(rag_responses_file):
        logger.error(f"❌ RAG responses file not found: {rag_responses_file}")
        return False
    
    if not os.path.exists(scores_file):
        logger.error(f"❌ Scores file not found: {scores_file}")
        return False
    
    # Load Phase 2 responses
    rag_responses = {}
    with open(rag_responses_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rag_responses[i + 1] = data
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Skipping malformed line {i+1}: {e}")
                continue
    
    logger.info(f"📋 Loaded {len(rag_responses)} Phase 2 responses")
    
    # Load manual evaluation scores
    try:
        with open(scores_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle both raw JSON array and wrapped JSON
            if '{' in content:
                json_start = content.index('{') if content.find('{') != -1 else 0
                json_end = content.rfind('}') + 1 if content.rfind('}') != -1 else len(content)
                content = content[json_start:json_end]
            scores_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse scores file as JSON: {e}")
        logger.error(f"   Make sure the file contains valid JSON array or object.")
        return False
    
    # Ensure it's a list
    if isinstance(scores_data, dict):
        scores_data = [scores_data]
    
    logger.info(f"📊 Loaded {len(scores_data)} evaluation scores")
    
    # Merge and create final report
    results = {}
    for score_entry in scores_data:
        q_num = score_entry.get('question_number')
        if q_num not in rag_responses:
            logger.warning(f"⚠️  Score for question {q_num} not found in Phase 2 responses")
            continue
        
        data = rag_responses[q_num].copy()
        
        # Add RAGAS metrics from manual evaluation
        data['ragas_faithfulness'] = score_entry.get('faithfulness', 0.0)
        data['ragas_context_precision'] = score_entry.get('context_precision', 0.0)
        data['ragas_context_recall'] = score_entry.get('context_recall', 0.0)
        data['ragas_answer_accuracy'] = score_entry.get('answer_accuracy', 0.0)
        data['ragas_answer_relevancy'] = score_entry.get('answer_accuracy', 0.0)  # fallback
        
        # Compute scores (same as Phase 3)
        accuracy = float(data['ragas_answer_accuracy'])
        faithfulness = float(data['ragas_faithfulness'])
        
        score = max(0, min(2, int(round(accuracy * 2))))
        combined_score = 0.7 * accuracy + 0.3 * faithfulness
        
        data['score'] = score
        data['combined_score'] = float(combined_score)
        data['combined_judgement'] = 2 if combined_score >= 0.85 else 1 if combined_score >= 0.6 else 0
        data['reason'] = (
            f"Manual evaluation: answer_accuracy={accuracy:.3f}; "
            f"faithfulness={faithfulness:.3f}"
        )
        
        results[q_num] = data
    
    if not results:
        logger.error("❌ No scores could be merged with Phase 2 responses")
        return False
    
    # Write final report
    with open(output_file, 'w', encoding='utf-8') as f:
        for q_num in sorted(results.keys()):
            f.write(json.dumps(results[q_num], ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Wrote {len(results)} final evaluation results to: {output_file}")
    
    # Summary statistics
    correct = sum(1 for d in results.values() if d['score'] == 2)
    partial = sum(1 for d in results.values() if d['score'] == 1)
    wrong = sum(1 for d in results.values() if d['score'] == 0)
    
    total = len(results)
    weighted_score = (sum(d['score'] for d in results.values()) / (total * 2) * 100) if total > 0 else 0
    strict_score = (correct / total * 100) if total > 0 else 0
    
    logger.info("")
    logger.info("=" * 40)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 40)
    logger.info(f"  Total evaluated:  {total}")
    logger.info(f"  ✅ Correct (2):   {correct}")
    logger.info(f"  ⚠️  Partial (1):   {partial}")
    logger.info(f"  ❌ Wrong    (0):   {wrong}")
    logger.info(f"  🎯 Weighted score: {weighted_score:.1f}%")
    logger.info(f"  📌 Strict score:   {strict_score:.1f}%")
    logger.info("=" * 40)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Import manual evaluation scores from ChatGPT/Gemini"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "train", "valid"],
        default="test",
        help="Which dataset to import (default: test)"
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default=None,
        help="Path to JSON scores file (default: evaluation/manual_eval_scores_<dataset>.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: evaluation/final_evaluation_<dataset>.jsonl)"
    )
    
    configure_logging()
    args = parser.parse_args()
    
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    rag_responses_file = os.path.join(EVAL_DIR, f"rag_responses_{args.dataset}.jsonl")
    scores_file = args.scores_file or os.path.join(EVAL_DIR, f"manual_eval_scores_{args.dataset}.json")
    output_file = args.output or os.path.join(EVAL_DIR, f"final_evaluation_{args.dataset}.jsonl")
    
    success = import_manual_eval(rag_responses_file, scores_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
