from __future__ import annotations

import json
import re
from typing import Tuple


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# def exact_or_f1(pred: str, gold: str) -> Tuple[float, float]:
#     p = normalize_text(pred)
#     g = normalize_text(gold)
#     exact = 1.0 if p == g else 0.0
#     p_tokens = p.split()
#     g_tokens = g.split()
#     if not p_tokens or not g_tokens:
#         return exact, 0.0
#     common = {}
#     for t in p_tokens:
#         common[t] = min(p_tokens.count(t), g_tokens.count(t))
#     num_same = sum(common.values())
#     if num_same == 0:
#         return exact, 0.0
#     precision = num_same / len(p_tokens)
#     recall = num_same / len(g_tokens)
#     f1 = 2 * precision * recall / (precision + recall)
#     return exact, f1


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
