from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    timestamp: float


class SemanticCache(ABC):
    """Abstract base class for semantic caches.

    Contract:
    - get_if_similar(query, query_emb) -> Optional[(answer, similarity)]
    - put(query, embedding, answer) -> None
    - size() -> int
    - stats() -> dict
    """

    @abstractmethod
    def get_if_similar(self, query: str, query_emb: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def put(self, query: str, embedding: np.ndarray, answer: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class BasicSemanticLRUCache(SemanticCache):
    """LRU cache keyed by semantic similarity of query embeddings.

    - capacity: max entries in cache
    - threshold: cosine similarity threshold for hit
    """

    def __init__(self, capacity: int = 1000, threshold: float = 0.85) -> None:
        self.capacity = int(capacity)
        self.threshold = float(threshold)
        self._store: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._counter = 0  # monotonic key for LRU order
        self.hits = 0
        self.misses = 0

    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {"size": self.size(), "hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}

    def get_if_similar(self, query: str, query_emb: np.ndarray):
        # Find best match
        best_key = None
        best_sim = -1.0
        for k, entry in self._store.items():
            sim = cosine_sim(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = k
        if best_sim >= self.threshold and best_key is not None:
            # Update LRU order
            entry = self._store.pop(best_key)
            self._store[best_key] = entry
            self.hits += 1
            return entry.answer, best_sim
        self.misses += 1
        return None, best_sim

    def put(self, query: str, embedding: np.ndarray, answer: str) -> None:
        # Insert new entry and evict if needed
        key = self._counter
        self._counter += 1
        entry = CacheEntry(query=query, embedding=embedding.astype(np.float32), answer=answer, timestamp=time.time())
        self._store[key] = entry
        # LRU eviction
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)


class SemanticCacheFactory:
    """Factory for creating semantic cache instances.

    Supported types:
    - "basic": BasicSemanticLRUCache
    """

    @staticmethod
    def create(cache_type: str = "basic", **kwargs) -> SemanticCache:
        cache_type_norm = (cache_type or "").strip().lower()
        if cache_type_norm == "basic":
            return BasicSemanticLRUCache(**kwargs)
        raise ValueError(f"Unsupported cache_type: {cache_type}. Supported: 'basic'")
