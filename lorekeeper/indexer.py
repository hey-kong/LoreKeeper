from __future__ import annotations

import os
from typing import Optional

from tqdm.auto import tqdm

from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader

from .embeddings import QwenHuggingFaceEmbedding


def build_and_persist_vec_index(
    data_dir: str,
    persist_dir: str,
    embed_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda:0",
    chunk_size: int = 512,
    chunk_overlap: int = 10,
    batch_size: int = 128,
    show_progress: bool = True,
) -> None:
    os.makedirs(persist_dir, exist_ok=True)

    # Configure embeddings in Settings so VectorStoreIndex uses it
    Settings.embed_model = QwenHuggingFaceEmbedding(embed_model_name, device).create()

    # Load and chunk documents
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
    docs = reader.load_data()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)

    # Build vector index with progress bar and persist
    total = len(nodes)
    if total == 0:
        print("No chunks found; nothing to index.")
        return

    print(f"Building vector index from {total} chunks...")

    index = None
    iterator = range(0, total, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Indexing chunks", unit="chunk", dynamic_ncols=True)

    for start in iterator:
        batch = nodes[start : start + batch_size]
        if index is None:
            # initialize index with first batch
            index = VectorStoreIndex(batch)
        else:
            # incrementally add nodes
            index.insert_nodes(batch)

    # Persist after all batches are indexed
    assert index is not None
    index.storage_context.persist(persist_dir=persist_dir)


def load_vec_index(persist_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)
