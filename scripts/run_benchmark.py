#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from lorekeeper.benchmark import run_benchmark
from lorekeeper.config import parse_args, resolve_config


def main(argv=None):
    ns = parse_args(argv)
    conf = resolve_config(ns)

    metrics = run_benchmark(conf)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
