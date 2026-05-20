#!/usr/bin/env python3
"""
export_for_manual_eval.py

Export Phase 2 RAG responses in a format suitable for manual evaluation via ChatGPT/Gemini.
Generates a structured prompt that can be copy-pasted into ChatGPT or Gemini.

Usage:
  python -m src.eval.export_for_manual_eval --dataset test --max-questions 5

This creates: evaluation/manual_eval_input_<dataset>.txt
"""
import argparse
import json
import os
import sys

from .utils import get_data_file, EVAL_DIR, configure_logging

def export_for_manual_eval(rag_responses_file, output_file, max_questions=None):
    """
    Export Phase 2 results for manual evaluation via ChatGPT/Gemini.
    
    Args:
        rag_responses_file: Path to Phase 2 output (rag_responses_*.jsonl)
        output_file: Path to output file
        max_questions: Max questions to export (None = all)
    """
    if not os.path.exists(rag_responses_file):
        print(f"❌ RAG responses file not found: {rag_responses_file}")
        return False
    
    items = []
    with open(rag_responses_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                items.append((i + 1, data))
                if max_questions and len(items) >= max_questions:
                    break
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {i+1}: {e}")
                continue
    
    if not items:
        print(f"❌ No valid items found in {rag_responses_file}")
        return False
    
    # Build evaluation prompt
    output_lines = [
        "=" * 80,
        "MANUAL RAG EVALUATION FOR ChatGPT / Gemini",
        "=" * 80,
        "",
        "Instructions:",
        "1. For each question below, evaluate the RAG answer against the reference answer.",
        "2. Score each metric on a scale of 0-1 (0 = poor, 1 = excellent):",
        "   - faithfulness: Does the answer only use information from the context?",
        "   - context_precision: Is the retrieved context relevant to the question?",
        "   - context_recall: Does the context contain info needed to answer the question?",
        "   - answer_accuracy: Is the answer correct and complete?",
        "3. Copy the JSON block at the end and send it back.",
        "",
        "=" * 80,
        ""
    ]
    
    for q_num, item in items:
        output_lines.append(f"QUESTION {q_num}:")
        output_lines.append(f"Question: {item.get('question', 'N/A')}")
        output_lines.append("")
        output_lines.append(f"Reference Answer: {item.get('answer', 'N/A')}")
        output_lines.append("")
        output_lines.append("Retrieved Context:")
        contexts = item.get('contexts', [])
        if isinstance(contexts, list):
            for i, ctx in enumerate(contexts, 1):
                output_lines.append(f"  [{i}] {ctx[:200]}...")
        else:
            output_lines.append(f"  {str(contexts)[:500]}")
        output_lines.append("")
        output_lines.append(f"RAG Answer: {item.get('rag_answer', 'N/A')}")
        output_lines.append("")
        output_lines.append(f"Provide scores for Q{q_num} in the JSON format below.")
        output_lines.append("")
    
    # Output JSON template
    output_lines.append("=" * 80)
    output_lines.append("COPY AND PASTE THE JSON BELOW INTO THE SYSTEM AFTER EVALUATION:")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    json_results = []
    for q_num, item in items:
        json_results.append({
            "question_number": q_num,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_accuracy": 0.0
        })
    
    output_lines.append(json.dumps(json_results, indent=2))
    output_lines.append("")
    output_lines.append("(Fill in the scores above, then save and use with import_manual_eval.py)")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ Exported {len(items)} questions to: {output_file}")
    print(f"📋 Instructions:")
    print(f"   1. Open {output_file}")
    print(f"   2. Copy the evaluation questions and send to ChatGPT/Gemini")
    print(f"   3. Ask it to fill in the JSON scores")
    print(f"   4. Save the JSON results")
    print(f"   5. Run: python -m src.eval.import_manual_eval --dataset {os.path.basename(rag_responses_file).split('_')[2].split('.')[0]}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Phase 2 RAG responses for manual evaluation via ChatGPT/Gemini"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "train", "valid"],
        default="test",
        help="Which dataset to export (default: test)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to export (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: evaluation/manual_eval_input_<dataset>.txt)"
    )
    
    configure_logging()
    args = parser.parse_args()
    
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    rag_responses_file = os.path.join(EVAL_DIR, f"rag_responses_{args.dataset}.jsonl")
    output_file = args.output or os.path.join(EVAL_DIR, f"manual_eval_input_{args.dataset}.txt")
    
    success = export_for_manual_eval(rag_responses_file, output_file, args.max_questions)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
