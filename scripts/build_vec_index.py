#!/usr/bin/env python3
from __future__ import annotations

import sys

from lorekeeper.config import parse_args, resolve_config
from lorekeeper.indexer import build_and_persist_vec_index


def main(argv=None):
    ns = parse_args(argv)
    conf = resolve_config(ns)

    build_and_persist_vec_index(
        data_dir=conf.paths.docs_dir,
        persist_dir=conf.paths.persist_dir,
        embed_model_name=conf.models.embedding_model,
        device=conf.models.device,
        chunk_size=conf.chunk.chunk_size,
        chunk_overlap=conf.chunk.chunk_overlap,
    )
    print(f"Index persisted to {conf.paths.persist_dir}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
