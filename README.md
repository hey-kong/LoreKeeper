# LoreKeeper

This repository provides the prototype implementation of our paper "LoreKeeper: Efficient and Precise Semantic Caching for Edge-Cloud RAG Systems".

## Environment Setup

Create a virtual environment using `uv` and install the project in editable mode:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## 快速开始

1) 构建索引（离线 chunk + 向量索引持久化）：

```bash
python3 -m scripts.build_vec_index --config ./config.example.toml
```

2) 运行基准测试：

```bash
python3 -m scripts.run_benchmark --config ./config.example.toml
```