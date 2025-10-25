from __future__ import annotations

from typing import List, Optional

import gc
import math
import torch
from transformers import AutoTokenizer
from FlagEmbedding import FlagReranker
import heapq


class BgeReranker:
    def __init__(self, model_path='BAAI/bge-reranker-v2-m3', device="cuda:0"):
        self.reranker = FlagReranker(
            model_path,
            use_fp16=True,
            devices=[device]
        )
        print(f'use local reranker: {model_path}')

    def rerank_nodes(self, query_text, nodes, top_k=8):
        pairs = [(query_text, node.text) for node in nodes]

        scores = self.reranker.compute_score(pairs)

        topk = heapq.nlargest(top_k, zip(scores, nodes), key=lambda x: x[0])
        return [node for _, node in topk]
