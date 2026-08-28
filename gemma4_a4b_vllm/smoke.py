#!/usr/bin/env python3
"""Text, cache, structured-output, tool and image smoke checks for Gemma 4."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CACHE_METRICS = {
    "vllm:external_prefix_cache_hits_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:mm_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
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
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read().decode()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def request_json(
    method: str, url: str, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    return json.loads(request_text(method, url, api_key, payload))


def metric_totals(text: str) -> dict[str, float]:
    totals = {name: 0.0 for name in CACHE_METRICS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, separator, raw_value = line.rpartition(" ")
        name = sample.split("{", 1)[0]
        if separator and name in totals:
            totals[name] += float(raw_value)
    return totals


def chat(
    base_url: str,
    api_key: str,
    model: str,
    content: str | list[dict[str, Any]],
    max_tokens: int,
    **extra: Any,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    started = time.monotonic()
    response = request_json("POST", f"{base_url}/chat/completions", api_key, payload)
    return response, time.monotonic() - started


def cached_tokens(response: dict[str, Any]) -> int | None:
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if value is not None else None


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_MODEL"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--tool", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set VLLM_API_KEY or pass --api-key")
    if args.image and not args.image.is_file():
        parser.error(f"image is not a file: {args.image}")

    base_url = args.base_url.rstrip("/")
    models = request_json("GET", f"{base_url}/models", args.api_key)
    model = args.model or models["data"][0]["id"]
    metrics_url = f"{base_url.removesuffix('/v1')}/metrics"
    metrics_before = metric_totals(request_text("GET", metrics_url, args.api_key))
    prompt = "Ответь одним предложением: какое свойство A100 полезно для inference?"

    first, first_s = chat(base_url, args.api_key, model, prompt, 128)
    second, second_s = chat(base_url, args.api_key, model, prompt, 128)
    print(
        json.dumps(
            {
                "text": first["choices"][0]["message"]["content"],
                "first_seconds": round(first_s, 3),
                "second_seconds": round(second_s, 3),
                "second_cached_tokens": cached_tokens(second),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.structured:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "shell_check",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "safe": {"type": "boolean"},
                        "summary": {"type": "string"},
                    },
                    "required": ["safe", "summary"],
                    "additionalProperties": False,
                },
            },
        }
        structured, elapsed = chat(
            base_url,
            args.api_key,
            model,
            "Оцени команду `pwd`.",
            128,
            response_format=response_format,
        )
        parsed = json.loads(structured["choices"][0]["message"]["content"])
        if set(parsed) != {"safe", "summary"}:
            raise RuntimeError("structured output does not match the requested schema")
        print(json.dumps({"structured_seconds": round(elapsed, 3), "structured": parsed}, ensure_ascii=False, indent=2))

    if args.tool:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one local text file by path.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        tool_response, elapsed = chat(
            base_url,
            args.api_key,
            model,
            "Используй read_file, чтобы прочитать /etc/os-release.",
            256,
            tools=tools,
            tool_choice="auto",
        )
        message = tool_response["choices"][0]["message"]
        if not message.get("tool_calls"):
            raise RuntimeError("Gemma 4 did not emit the requested tool call")
        print(json.dumps({"tool_seconds": round(elapsed, 3), "tool_calls": message["tool_calls"]}, ensure_ascii=False, indent=2))

    if args.image:
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(args.image)}},
            {
                "type": "text",
                "text": (
                    "Разбери screenshot для кодингового агента, максимум 20 строк. "
                    "Формат Markdown: Наблюдения; OCR (`дословный текст` — confidence "
                    "N% для каждого фрагмента); Ошибки с file:line; Вероятная причина; "
                    "Следующие проверки. Отделяй наблюдения от выводов. Нечитаемые "
                    "фрагменты помечай [неразборчиво]."
                ),
            },
        ]
        response, elapsed = chat(base_url, args.api_key, model, content, 768)
        repeated, repeated_elapsed = chat(base_url, args.api_key, model, content, 768)
        print(
            json.dumps(
                {
                    "image_seconds": round(elapsed, 3),
                    "image_repeat_seconds": round(repeated_elapsed, 3),
                    "image_answer": response["choices"][0]["message"]["content"],
                    "image_repeat_answer": repeated["choices"][0]["message"]["content"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    metrics_after = metric_totals(request_text("GET", metrics_url, args.api_key))
    deltas = {
        name.removeprefix("vllm:"): metrics_after[name] - value
        for name, value in sorted(metrics_before.items())
    }
    accepted = deltas["spec_decode_num_accepted_tokens_total"]
    drafted = deltas["spec_decode_num_draft_tokens_total"]
    result: dict[str, Any] = {
        "cache_metric_deltas": {name: round(value, 3) for name, value in deltas.items()}
    }
    if drafted > 0:
        result["mtp_acceptance_rate"] = round(accepted / drafted, 4)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
