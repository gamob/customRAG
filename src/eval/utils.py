import logging
import os
import sys

logger = logging.getLogger(__name__)
BUG_ONLY = False
EVAL_DIR = "evaluation"


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger().setLevel(level)


def log_bug(message: str):
    if BUG_ONLY:
        print(message)
    else:
        logger.error(message)


def get_data_file(dataset_type="test"):
    if dataset_type not in ["test", "train", "valid"]:
        dataset_type = "test"
    return os.path.join(EVAL_DIR, f"eval_{dataset_type}.jsonl")


def build_ollama_llm(model_name, num_ctx, num_predict, stop=None):
    """
    Build a plain OllamaLLM for use as a judge or utility model.
    Moved here from llm_sanitizer.py after RAGAS/DeepEval removal.
    """
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.0,
        num_ctx=num_ctx,
        keep_alive=-1,
        num_predict=num_predict,
        stop=stop or ["\n\n", "---"],
    )
