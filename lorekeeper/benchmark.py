from __future__ import annotations

import statistics
import time
from typing import Dict, List
import json
from tqdm import tqdm

from .query_engine import RagWithSemanticCache
from .utils import read_jsonl
from .config import AppConfig
from .cal_f1 import calc_f1_score


def run_benchmark(config: AppConfig) -> Dict:
    """Run the RAG+Cache benchmark with a single AppConfig.

    Args:
        config: The application configuration containing models, retrieval, cache, and paths.

    Returns:
        A dict of benchmark metrics.
    """
    engine = RagWithSemanticCache(config=config)

    latencies: List[float] = []
    hits = 0
    total = 0

    questions = {ex.get("id"): ex.get("query") for ex in read_jsonl(config.paths.questions_path)}
    num_questions = config.benchmark.num_questions
    if num_questions > 0 and num_questions < len(questions):
        questions = dict(list(questions.items())[:num_questions])

    # Clear the file before writing new results
    with open(config.paths.results_path, 'w', encoding='utf-8'):
        pass  # just open in write mode to truncate the file

    with open(config.paths.results_path, 'a', encoding='utf-8') as file:
        items = list(questions.items())
        with tqdm(total=len(items), desc="Benchmark", unit="q", dynamic_ncols=True) as pbar:
            for qid, q in items:
                if not q:
                    pbar.update(1)
                    continue
                res = engine.query(q)
                latencies.append(res["latency_sec"])
                hits += 1 if res["cache_hit"] else 0
                total += 1

                # record to file
                res["id"] = qid
                file.write(json.dumps(res, ensure_ascii=False) + '\n')

                # 更新进度条附加信息
                hit_rate = (hits / total) if total else 0.0
                avg_lat = (statistics.mean(latencies) if latencies else 0.0)
                pbar.set_postfix(
                    hit_rate=f"{hit_rate:.2%}",
                    avg_lat_s=f"{avg_lat:.2f}",
                    done=total,
                )
                pbar.update(1)
    avg_f1 = calc_f1_score(config.paths.answers_path, config.paths.results_path)


    metrics = {
        "count": total,
        "cache_hit_rate": (hits / total) if total else 0.0,
        "avg_latency_sec": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_sec": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0,
        "f1_score": avg_f1,
        "cache": engine.cache.stats(),
    }
    return metrics
