#!/usr/bin/env python3
"""Text, prefix-cache and optional image smoke checks for the recipe."""

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


def request_json(
    method: str, url: str, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
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
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def chat(
    base_url: str,
    api_key: str,
    model: str,
    content: str | list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
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
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set VLLM_API_KEY or pass --api-key")

    base_url = args.base_url.rstrip("/")
    models = request_json("GET", f"{base_url}/models", args.api_key)
    model = args.model or models["data"][0]["id"]
    prompt = "Ответь одним предложением: какое главное свойство A100 полезно для inference?"

    first, first_s = chat(base_url, args.api_key, model, prompt, 128)
    second, second_s = chat(base_url, args.api_key, model, prompt, 128)
    print(json.dumps({
        "text": first["choices"][0]["message"]["content"],
        "first_seconds": round(first_s, 3),
        "second_seconds": round(second_s, 3),
        "second_cached_tokens": cached_tokens(second),
    }, ensure_ascii=False, indent=2))

    if args.image:
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(args.image)}},
            {
                "type": "text",
                "text": (
                    "Разбери изображение для кодингового агента. Укажи тип экрана, "
                    "видимые элементы, весь читаемый текст, ошибки и степень уверенности."
                ),
            },
        ]
        response, elapsed = chat(base_url, args.api_key, model, content, 768)
        print(json.dumps({
            "image_seconds": round(elapsed, 3),
            "image_answer": response["choices"][0]["message"]["content"],
        }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
