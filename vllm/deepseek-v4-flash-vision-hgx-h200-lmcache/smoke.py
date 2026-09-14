"""Synthetic text, JSON/SSE multi-turn tools and vision; no prompt logging."""
import argparse
import base64
import json
import os
import struct
import time
import urllib.request
import zlib

BASE = os.environ.get("VLLM_DSV4_BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
MODEL = "deepseek-v4-flash-vision"


def headers(rank=None):
    result = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"], "Content-Type": "application/json"}
    if rank is not None:
        result["X-data-parallel-rank"] = str(rank)
    return result


def request(messages, rank=None, timeout=300, **kwargs):
    body = dict(model=MODEL, messages=messages, temperature=0, max_tokens=1024,
                chat_template_kwargs={"thinking": False})
    body.update(kwargs)
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers(rank))
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, stream=False, **kwargs):
    if stream:
        return stream_chat(messages, **kwargs)[0]
    with request(messages, **kwargs) as response:
        data = json.load(response)
    choice = data["choices"][0]
    assert choice["finish_reason"] in {"stop", "tool_calls"}, "incomplete JSON response"
    return choice["message"]


def collect_stream(response, started):
    """Reassemble OpenAI deltas, including tool arguments split across events."""
    content, reasoning, calls, usage = [], [], {}, {}
    done, finish, first = False, None, None
    for line in response:
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        event = json.loads(payload)
        assert "error" not in event, "SSE error"
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            assert choice["index"] == 0, "unexpected multiple choices"
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                first = time.monotonic() - started
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            for fragment in delta.get("tool_calls") or []:
                call = calls.setdefault(fragment["index"], {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                })
                call["id"] += fragment.get("id") or ""
                for key in ("name", "arguments"):
                    call["function"][key] += fragment.get("function", {}).get(key) or ""
    assert done and finish in {"stop", "tool_calls"}, "incomplete SSE response"
    message = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        assert sorted(calls) == list(range(len(calls))), "missing tool delta index"
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message, {"ttft_seconds": first, "elapsed_seconds": time.monotonic() - started, "usage": usage}


def stream_chat(messages, **kwargs):
    started = time.monotonic()
    with request(messages, stream=True, stream_options={"include_usage": True}, **kwargs) as response:
        return collect_stream(response, started)


def check_tool_round_trips(stream):
    tools = [{"type": "function", "function": {"name": "lookup_build", "description": "Get the build status by ID.",
              "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
    history, ids = [], set()
    for build in ("test-42", "test-43"):
        history.append({"role": "user", "content": f"Use lookup_build to check build {build}."})
        message = chat(history, stream=stream, tools=tools, tool_choice="auto")
        calls = message.get("tool_calls") or []
        assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_build", "tool parser failed"
        assert json.loads(calls[0]["function"]["arguments"]) == {"id": build}, "tool arguments failed"
        call_id = calls[0]["id"]
        assert call_id and call_id not in ids, "missing or reused tool call ID"
        ids.add(call_id)
        history += [message, {"role": "tool", "tool_call_id": call_id, "content": '{"status":"passed"}'}]
        # Keep auto enabled after a tool result, as coding clients do.
        message = chat(history, stream=stream, tools=tools, tool_choice="auto")
        assert not message.get("tool_calls"), "unexpected repeated lookup after result"
        assert "passed" in (message.get("content") or "").lower(), "tool result continuation failed"
        history.append(message)
    print("OK: two tool round trips including tool-result history; " + ("SSE" if stream else "JSON"))


def solid_png(rgb):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    pixels = (b"\0" + bytes(rgb) * 448) * 448
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 448, 448, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    # Synthetic requests only; neither model replies nor tool payloads are persisted.
    message = chat([{"role": "user", "content": "Compute 17 + 25. Reply with the number only."}])
    assert "42" in (message.get("content") or ""), "text check failed"
    print("OK: text")

    message = chat([{"role": "user", "content": "Compute 17 + 25. Reply with the number only."}], stream=True)
    assert "42" in (message.get("content") or ""), "streaming check failed"
    print("OK: SSE")

    check_tool_round_trips(stream=False)
    check_tool_round_trips(stream=True)

    # Same text, changing image, repeated image: catches absent vision and simple cache aliasing.
    for rank, color, rgb in [(0, "red", (255, 0, 0)), (1, "blue", (0, 0, 255)), (7, "red", (255, 0, 0))]:
        message = chat([{"role": "system", "content": "Synthetic cache check. " * 1024}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": solid_png(rgb)}},
            {"type": "text", "text": "Name the solid color in this image. One English word only."},
        ]}], rank=rank)
        assert color in (message.get("content") or "").lower(), "vision/color/cache check failed"
    print("OK: native vision red/blue/red across DP ranks 0/1/7")
    print("GPU smoke passed. Run acceptance for long context and cache restore, then qualify concurrent load separately.")


if __name__ == "__main__":
    main()
