# Manual Evaluation Workflow

When the local LLM model is unable to generate reliable JSON output for Phase 3 evaluation, you can use ChatGPT or Gemini to manually evaluate the RAG responses.

## Workflow

### Step 1: Export Phase 2 Results

First, ensure Phase 2 (RAG inference) is complete and you have `evaluation/rag_responses_test.jsonl` (or other dataset).

```bash
python -m src.eval.export_for_manual_eval --dataset test
```

This generates: `evaluation/manual_eval_input_test.txt`

The file contains:
- All question-answer pairs from Phase 2
- Retrieved context for each question
- Reference answer for comparison
- JSON template for scores

### Step 2: Manual Evaluation via ChatGPT/Gemini

1. Open `evaluation/manual_eval_input_test.txt`
2. Copy the content
3. Paste into ChatGPT or Gemini with instructions:
   ```
   Please evaluate each RAG answer and fill in the JSON scores at the end.
   For each metric (faithfulness, context_precision, context_recall, answer_accuracy),
   provide a score between 0 and 1, where 0 is poor and 1 is excellent.
   ```
4. The model will fill in the JSON with scores
5. Copy the JSON output and save to: `evaluation/manual_eval_scores_test.json`

### Step 3: Import Results Back

Once you have the scores file:

```bash
python -m src.eval.import_manual_eval --dataset test
```

This:
- Reads the manual scores from `evaluation/manual_eval_scores_test.json`
- Merges them with Phase 2 RAG responses
- Creates final evaluation report: `evaluation/final_evaluation_test.jsonl`
- Displays summary statistics

## Files Used

- **Input**: `evaluation/rag_responses_<dataset>.jsonl` (Phase 2 output)
- **Export**: `evaluation/manual_eval_input_<dataset>.txt`
- **Scores**: `evaluation/manual_eval_scores_<dataset>.json`
- **Output**: `evaluation/final_evaluation_<dataset>.jsonl`

## Example Usage

For all three datasets:

```bash
# Export
python -m src.eval.export_for_manual_eval --dataset test
python -m src.eval.export_for_manual_eval --dataset train
python -m src.eval.export_for_manual_eval --dataset valid

# [Use ChatGPT/Gemini to fill in scores]

# Import
python -m src.eval.import_manual_eval --dataset test
python -m src.eval.import_manual_eval --dataset train
python -m src.eval.import_manual_eval --dataset valid
```

## Options

### export_for_manual_eval.py

- `--dataset`: Which dataset to export (test, train, valid)
- `--max-questions`: Limit number of questions to export
- `--output`: Custom output file path

### import_manual_eval.py

- `--dataset`: Which dataset to import
- `--scores-file`: Path to manual scores JSON file
- `--output`: Custom output file path

## Notes

- The JSON template provided in the export file should be filled in completely
- Scores must be between 0 and 1
- The import script will compute overall weighted and strict scores
- The final evaluation file matches the format of Phase 3 output
