from __future__ import annotations

import argparse
import dataclasses
import os
import tomllib
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda:0"


@dataclass
class CacheConfig:
    cache_type: str = "basic"
    capacity: int = 1000
    similarity_threshold: float = 0.85


@dataclass
class RetrievalConfig:
    top_k: int = 20
    rerank_top_k: int = 8


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 10


@dataclass
class PathsConfig:
    docs_dir: str = "./data/hotpotqa/documents"
    persist_dir: str = "./storage/index/hotpotqa"
    questions_path: str = "./data/hotpotqa/questions/questions.jsonl"
    answers_path: str = "./data/hotpotqa/answers/answers.jsonl"
    results_path: str = "./results.jsonl"

@dataclass
class BenchmarkConfig:
    num_questions: int = 0  # 0 means all

@dataclass
class AppConfig:
    models: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = dataclasses.field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = dataclasses.field(default_factory=ChunkConfig)
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    benchmark: BenchmarkConfig = dataclasses.field(default_factory=BenchmarkConfig)


def load_toml(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def from_dict(d: dict) -> AppConfig:
    def get(ns: str, dc):
        return dc(**d.get(ns, {}))

    return AppConfig(
        models=get("models", ModelConfig),
        cache=get("cache", CacheConfig),
        retrieval=get("retrieval", RetrievalConfig),
        chunk=get("chunk", ChunkConfig),
        paths=get("paths", PathsConfig),
        benchmark=get("benchmark", BenchmarkConfig),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoreKeeper RAG+Cache config")
    p.add_argument("--config", type=str, default=os.environ.get("LOREKEEPER_CONFIG"), help="Path to TOML config")
    # model overrides
    p.add_argument("--embedding_model", type=str)
    p.add_argument("--reranker_model", type=str)
    p.add_argument("--llm_model", type=str)
    p.add_argument("--device", type=str)
    # cache
    p.add_argument("--cache_type", type=str)
    p.add_argument("--capacity", type=int)
    p.add_argument("--similarity_threshold", type=float)
    # retrieval
    p.add_argument("--top_k", type=int)
    p.add_argument("--rerank_top_k", type=int)
    # chunk
    p.add_argument("--chunk_size", type=int)
    p.add_argument("--chunk_overlap", type=int)
    # paths
    p.add_argument("--docs_dir", type=str)
    p.add_argument("--persist_dir", type=str)
    p.add_argument("--questions_path", type=str)
    p.add_argument("--answers_path", type=str)
    p.add_argument("--results_path", type=str)
    # benchmark
    p.add_argument("--num_questions", type=int)
    return p.parse_args(argv)


def resolve_config(ns: argparse.Namespace) -> AppConfig:
    conf = from_dict(load_toml(getattr(ns, "config", None)))
    # overrides
    if ns.embedding_model:
        conf.models.embedding_model = ns.embedding_model
    if ns.reranker_model:
        conf.models.reranker_model = ns.reranker_model
    if ns.llm_model:
        conf.models.llm_model = ns.llm_model
    if ns.device:
        conf.models.device = ns.device

    if ns.cache_type:
        conf.cache.cache_type = ns.cache_type
    if ns.capacity:
        conf.cache.capacity = ns.capacity
    if ns.similarity_threshold is not None:
        conf.cache.similarity_threshold = ns.similarity_threshold

    if ns.top_k:
        conf.retrieval.top_k = ns.top_k
    if ns.rerank_top_k:
        conf.retrieval.rerank_top_k = ns.rerank_top_k

    if ns.chunk_size:
        conf.chunk.chunk_size = ns.chunk_size
    if ns.chunk_overlap:
        conf.chunk.chunk_overlap = ns.chunk_overlap

    if ns.docs_dir:
        conf.paths.docs_dir = ns.docs_dir
    if ns.persist_dir:
        conf.paths.persist_dir = ns.persist_dir
    if ns.questions_path:
        conf.paths.questions_path = ns.questions_path
    if ns.answers_path:
        conf.paths.answers_path = ns.answers_path
    if ns.results_path:
        conf.paths.results_path = ns.results_path

    if ns.num_questions is not None:
        conf.benchmark.num_questions = ns.num_questions

    return conf
