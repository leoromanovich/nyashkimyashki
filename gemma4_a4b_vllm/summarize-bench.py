#!/usr/bin/env python3
"""Build compact JSON and CSV summaries from a vllm bench serve matrix."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


FIELDS = (
    "mode",
    "concurrency",
    "input_tokens",
    "output_tokens",
    "num_prompts",
    "completed",
    "failed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "spec_decode_acceptance_rate",
)


def positive_values(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default).split()
    values = [int(item) for item in raw]
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize-bench.py RESULT_DIR")
    result_dir = Path(sys.argv[1])
    concurrencies = positive_values("BENCH_CONCURRENCIES", "2 4 8 16 32")
    input_lengths = positive_values("BENCH_INPUT_LENGTHS", "8192 16384 32768")
    output_len = int(os.environ.get("BENCH_OUTPUT_LEN", "1024"))
    expected = {
        (concurrency, input_len, output_len)
        for concurrency in concurrencies
        for input_len in input_lengths
    }

    rows: list[dict[str, Any]] = []
    observed: set[tuple[int, int, int]] = set()
    for path in sorted(result_dir.glob("c*-in*-out*.json")):
        data = json.loads(path.read_text())
        concurrency = int(data["concurrency"])
        input_tokens = int(data["input_tokens"])
        output_tokens = int(data["output_tokens"])
        observed.add((concurrency, input_tokens, output_tokens))
        row = {field: data.get(field) for field in FIELDS}
        row["concurrency"] = concurrency
        row["input_tokens"] = input_tokens
        row["output_tokens"] = output_tokens
        rows.append(row)

    rows.sort(key=lambda row: (row["input_tokens"], row["concurrency"]))
    missing = sorted(expected - observed)
    failed_cells = [
        (row["concurrency"], row["input_tokens"], row["output_tokens"])
        for row in rows
        if int(row.get("failed") or 0) > 0
        or int(row.get("completed") or 0) != int(row.get("num_prompts") or 0)
    ]
    summary = {
        "expected_cells": len(expected),
        "completed_cells": len(rows),
        "missing_cells": missing,
        "failed_cells": failed_cells,
        "rows": rows,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (result_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({key: summary[key] for key in summary if key != "rows"}, indent=2))
    return 1 if missing or failed_cells else 0


if __name__ == "__main__":
    raise SystemExit(main())
