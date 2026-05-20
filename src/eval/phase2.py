import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait

from .utils import log_bug
from ..core.brain_service import Brain
from ..core.generate import answer_question

logger = logging.getLogger(__name__)
PHASE2_WORKERS = max(1, min(2, os.cpu_count() or 1))
PHASE2_TIMEOUT = 1000  # seconds per question


def _answer_one(item):
    idx, data, brain = item
    try:
        logger.info(f"  ⏳ Starting Q{idx+1}: {data['question'][:100]}")
        docs, _, confidence_pct = brain.search(data["question"])
        logger.info(f"  🔎 Search returned {len(docs)} docs for Q{idx+1}")

        contexts = [doc.page_content for doc in docs]
        data["contexts"] = contexts

        rag_answer, sources = answer_question(
            data["question"],
            docs,
            confidence=confidence_pct / 100.0,
        )
        data["rag_answer"] = rag_answer

        src_label = f" [{sources[0]}]" if sources else ""
        logger.info(f"\n  Q{idx+1}{src_label}: {data['question'][:80]}")
        logger.info(f"  💬 {rag_answer[:200]}{'...' if len(rag_answer) > 200 else ''}")
        return idx, data
    except Exception as e:
        logger.error(f"  ❌ Q{idx+1} error: {e}")
        return idx, None


def run_rag_inference(brain, args, data_file, rag_answers):
    if args.sample is not None and args.sample > 0:
        logger.info(f"🤖 Phase 2: Answering (Random {args.sample} of {PHASE2_WORKERS} workers, timeout={PHASE2_TIMEOUT}s)...")
    else:
        logger.info(f"🤖 Phase 2: Answering (All questions with {PHASE2_WORKERS} workers, timeout={PHASE2_TIMEOUT}s)...")
    t0 = time.time()

    all_raw_data = []
    with open(data_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                all_raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"  ⚠️  Malformed line skipped: {line[:50]}...")

    if args.sample is not None and args.sample > 0:
        num_to_test = min(args.sample, len(all_raw_data))
        selected_questions = random.sample(all_raw_data, num_to_test)
        logger.info(f"  🎲 Randomly selected {num_to_test} questions for this test run.")
    else:
        num_to_test = len(all_raw_data)
        selected_questions = all_raw_data
        logger.info(f"  📊 Using all {num_to_test} questions from {args.dataset} dataset.")

    items = [(i, data, brain) for i, data in enumerate(selected_questions)]
    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=PHASE2_WORKERS) as pool:
        futures = {pool.submit(_answer_one, item): item[0] for item in items}
        done_futures, not_done_futures = wait(futures, timeout=PHASE2_TIMEOUT)
        for future in done_futures:
            idx = futures[future]
            try:
                idx, data = future.result()
            except Exception as e:
                log_bug(f"  💥 Q{idx+1} crashed: {e}. Bugged question: {items[idx][1].get('question','<unknown>')[:200]}")
                done += 1
                continue
            results[idx] = data
            done += 1
            if done == len(items):
                logger.info(f"  ✅ {done}/{len(items)} answered")
        for future in not_done_futures:
            idx = futures[future]
            log_bug(f"  ⏱️  Q{idx+1} timed out after {PHASE2_TIMEOUT}s. Bugged question: {items[idx][1].get('question','<unknown>')[:200]}")
            future.cancel()
            done += 1
        if not_done_futures:
            logger.warning(f"  ⚠️  {len(not_done_futures)} question(s) timed out and were canceled.")

    count = 0
    with open(rag_answers, "w", encoding="utf-8") as fout:
        for data in results:
            if data is not None:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"\n🎉 Phase 2 done — {count}/{len(items)} answered in {time.time()-t0:.1f}s")
