"""Synthetic text, SSE, tool round trip, and two-image smoke; no prompt logging."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import struct
import urllib.request
import zlib

BASE = os.environ.get("DSV4_BASE_URL", "http://127.0.0.1:30000/v1").rstrip("/")
KEY = os.environ["SGLANG_API_KEY"]
MODEL = "deepseek-v4-flash-vision"


def request(messages, rank=None, **kwargs):
    body = dict(model=MODEL, messages=messages, temperature=0, max_tokens=1024,
                chat_template_kwargs={"thinking": False})
    body.update(kwargs)
    headers = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
    if rank is not None:
        headers["X-Data-Parallel-Rank"] = str(rank)
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=300)


def chat(messages, **kwargs):
    with request(messages, **kwargs) as response:
        data = json.load(response)
    choice = data["choices"][0]
    assert choice["finish_reason"] != "length", "smoke output exhausted token budget"
    return choice["message"]


def solid_png(rgb):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    pixels = (b"\0" + bytes(rgb) * 448) * 448
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 448, 448, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    # Synthetic requests only; neither model replies nor tool payloads are persisted.
    message = chat([{"role": "user", "content": "Compute 17 + 25. Reply with the number only."}])
    assert "42" in (message.get("content") or ""), "text check failed"
    print("OK: text")

    content, done = [], False
    with request([{"role": "user", "content": "Compute 17 + 25. Reply with the number only."}], stream=True) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            assert "error" not in event, "SSE error"
            for choice in event.get("choices", []):
                content.append(choice.get("delta", {}).get("content") or "")
    assert done and "42" in "".join(content), "streaming check failed"
    print("OK: SSE")

    tools = [{"type": "function", "function": {"name": "lookup_build", "description": "Get the build status by ID.",
              "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
    history = [{"role": "user", "content": "Use lookup_build to check build test-42."}]
    message = chat(history, tools=tools, tool_choice="auto")
    calls = message.get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_build", "tool parser failed"
    assert json.loads(calls[0]["function"]["arguments"])["id"] == "test-42", "tool arguments failed"
    history += [message, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"status":"passed"}'}]
    message = chat(history, tools=tools, tool_choice="none")
    assert "passed" in (message.get("content") or "").lower(), "tool result continuation failed"
    print("OK: tool call and result continuation")

    # Same text, changing image, repeated image: catches absent vision and simple cache aliasing.
    for rank, color, rgb in [(0, "red", (255, 0, 0)), (1, "blue", (0, 0, 255)), (7, "red", (255, 0, 0))]:
        message = chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": solid_png(rgb)}},
            {"type": "text", "text": "Name the solid color in this image. One English word only."},
        ]}], rank=rank)
        assert color in (message.get("content") or "").lower(), "vision/color/cache check failed"
    print("OK: native vision red/blue/red on DP ranks 0/1/7 with idle peers")
    # Concurrent text and image requests exercise cross-rank Vision routing.
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = []
        for rank in (0, 2, 4, 7):
            content = "Compute 17 + 25. Reply with the number only."
            expected = "42"
            if rank == 7:
                content = [
                    {"type": "image_url", "image_url": {"url": solid_png((0, 0, 255))}},
                    {"type": "text", "text": "Name the solid color. One English word only."},
                ]
                expected = "blue"
            checks.append((pool.submit(chat, [{"role": "user", "content": content}], rank=rank), expected))
        for future, expected in checks:
            assert expected in (future.result().get("content") or "").lower(), "mixed DPA check failed"
    print("OK: concurrent text/image DP requests")
    print("GPU smoke passed. Long-context, concurrent load and L2/L3 restoration require separate acceptance.")


if __name__ == "__main__":
    main()
