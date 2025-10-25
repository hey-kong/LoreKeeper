from __future__ import annotations

from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch



class QwenHuggingFaceEmbedding:
    """Factory to create a LlamaIndex-compatible embedding model.

    Uses sentence-transformers/HF under the hood via LlamaIndex's HuggingFaceEmbedding.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda:0") -> None:
        self.model_name = model_name
        self.device = device

    def create(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=self.model_name,
            device=self.device,
        )
