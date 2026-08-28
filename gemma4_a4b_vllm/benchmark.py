#!/usr/bin/env python3
"""Small OpenAI-compatible throughput probe for base/MTP A/B runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Ты помощник для инженеров. Давай проверяемые краткие ответы, отделяй факты "
    "от предположений, опасные shell-команды помечай явно."
)
DEFAULT_PROMPTS = [
    "Объясни, как найти процесс, слушающий TCP-порт 8000 в Linux.",
    "Дай безопасный план диагностики заполненного NVMe без удаления данных.",
    "Суммируй преимущества prefix cache для повторных вопросов по документу.",
    "Напиши Bash-функцию, проверяющую наличие читаемого файла конфигурации.",
    "Объясни различие между throughput и latency при continuous batching.",
    "Перечисли метрики для A/B-теста speculative decoding.",
]
SPEC_METRICS = {
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
}


def request_text(
    method: str, url: str, api_key: str, payload: dict[str, Any] | None = None
) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.read().decode()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def metric_totals(text: str) -> dict[str, float]:
    totals = {name: 0.0 for name in SPEC_METRICS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, separator, raw_value = line.rpartition(" ")
        name = sample.split("{", 1)[0]
        if separator and name in totals:
            totals[name] += float(raw_value)
    return totals


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def load_messages(path: Path | None) -> list[list[dict[str, str]]]:
    if path is None:
        return [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in DEFAULT_PROMPTS
        ]
    messages: list[list[dict[str, str]]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        item = value.get("messages") if isinstance(value, dict) else value
        if not isinstance(item, list):
            raise ValueError(f"{path}:{line_number}: expected messages list")
        messages.append(item)
    if not messages:
        raise ValueError(f"{path}: no prompts")
    return messages


def run_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, float | int]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    response = json.loads(
        request_text("POST", f"{base_url}/chat/completions", api_key, payload)
    )
    elapsed = time.monotonic() - started
    usage = response.get("usage", {})
    return {
        "seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_MODEL"))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set VLLM_API_KEY or pass --api-key")
    if args.concurrency < 1 or args.requests < args.concurrency:
        parser.error("requests must be at least concurrency, both positive")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")

    base_url = args.base_url.rstrip("/")
    models = json.loads(request_text("GET", f"{base_url}/models", args.api_key))
    model = args.model or models["data"][0]["id"]
    messages = load_messages(args.prompt_file)
    metrics_url = f"{base_url.removesuffix('/v1')}/metrics"

    run_request(base_url, args.api_key, model, messages[0], min(32, args.max_tokens))
    metrics_before = metric_totals(request_text("GET", metrics_url, args.api_key))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_request,
                base_url,
                args.api_key,
                model,
                messages[index % len(messages)],
                args.max_tokens,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.monotonic() - started
    metrics_after = metric_totals(request_text("GET", metrics_url, args.api_key))

    latencies = [float(item["seconds"]) for item in results]
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in results)
    output_tokens = sum(int(item["output_tokens"]) for item in results)
    spec_delta = {
        name.removeprefix("vllm:").removesuffix("_total"): metrics_after[name] - value
        for name, value in metrics_before.items()
    }
    report: dict[str, Any] = {
        "label": args.label,
        "model": model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "wall_seconds": round(wall_seconds, 4),
        "requests_per_second": round(args.requests / wall_seconds, 4),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens_per_second": round(prompt_tokens / wall_seconds, 2),
        "output_tokens_per_second": round(output_tokens / wall_seconds, 2),
        "latency_seconds": {
            "p50": round(percentile(latencies, 0.50), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
            "max": round(max(latencies), 4),
        },
        "spec_decode_delta": {key: round(value, 3) for key, value in spec_delta.items()},
    }
    drafted = spec_delta["spec_decode_num_draft_tokens"]
    if drafted > 0:
        report["mtp_acceptance_rate"] = round(
            spec_delta["spec_decode_num_accepted_tokens"] / drafted, 4
        )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
