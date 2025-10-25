from __future__ import annotations

import time
from typing import Optional

import numpy as np
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer

from .semantic_cache.cache import SemanticCache, SemanticCacheFactory
from .embeddings import QwenHuggingFaceEmbedding
from .indexer import load_vec_index
from .slm import CustomModelWrapper
from .reranker import BgeReranker
from .config import AppConfig

class RagWithSemanticCache:
    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        # Models
        Settings.embed_model = QwenHuggingFaceEmbedding(config.models.embedding_model, config.models.device).create()
        Settings.llm = None
        self.reranker = BgeReranker(model_path=config.models.reranker_model, device=config.models.device)
        self.slm = CustomModelWrapper(model_path=config.models.llm_model, device=config.models.device)

        # Index
        self.index = load_vec_index(config.paths.persist_dir)
        self.retriever = self.index.as_retriever(similarity_top_k=config.retrieval.top_k)

        # Cache
        self.cache = SemanticCacheFactory.create(
            cache_type=config.cache.cache_type,
            capacity=config.cache.capacity,
            threshold=config.cache.similarity_threshold
        )
        self.embed_for_cache = Settings.embed_model  # reuse same embedder

    def embed_query(self, query: str) -> np.ndarray:
        # LlamaIndex embed_model exposes .get_query_embedding
        vec = self.embed_for_cache.get_query_embedding(query)
        return np.array(vec, dtype=np.float32)

    def query(self, question: str) -> dict:
        t0 = time.perf_counter()

        # 1) Semantic cache lookup
        qvec = self.embed_query(question)
        cache_hit = False
        cached_ans, cache_sim = self.cache.get_if_similar(question, qvec)
        if cached_ans is not None:
            answer = cached_ans
            cache_hit = True
            latency = time.perf_counter() - t0
            return {
                "answer": answer,
                "cache_hit": cache_hit,
                "cache_similarity": cache_sim,
                "latency_sec": latency,
            }

        # 2) Retrieve
        nodes = self.retriever.retrieve(question)

        # 3) Rerank
        reranked = self.reranker.rerank_nodes(question, nodes)

        # 4) Synthesize/generate
        response = self.slm.generate_answer(question, reranked)
        answer_text = str(response)

        # 5) Insert into cache
        self.cache.put(question, qvec, answer_text)

        latency = time.perf_counter() - t0
        return {
            "answer": answer_text,
            "cache_hit": cache_hit,
            "cache_similarity": cache_sim,
            "latency_sec": latency,
        }
