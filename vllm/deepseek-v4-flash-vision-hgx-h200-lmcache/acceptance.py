"""Explicit synthetic long-context/cache probe; stdout contains metrics only."""
import argparse
import hashlib
import json
import math
import re
import urllib.request

from smoke import BASE, MODEL, headers, stream_chat

COUNTERS = {
    "lmcache_mp_num_chunks_loaded_total",
    "lmcache_mp_l1_write_chunks_total",
    "lmcache_mp_l2_store_completed_objects_chunks_total",
    "lmcache_mp_l2_store_completed_requests_total",
    "lmcache_mp_l2_load_completed_requests_total",
    "lmcache_mp_l2_prefetch_hit_chunks_total",
}
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def parse_metrics(body):
    # Aggregate known numeric counters; never retain labels, salts or request IDs.
    values = {}
    for line in body.splitlines():
        match = SAMPLE.fullmatch(line)
        if match and match[1] in COUNTERS:
            value = float(match[2])
            assert math.isfinite(value) and value >= 0, "invalid cache counter"
            values[match[1]] = values.get(match[1], 0) + value
    return values


def metrics(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return parse_metrics(response.read().decode())


def delta_metrics(before, after):
    assert not (before.keys() - after.keys()), "cache counter disappeared; retry on an idle instance"
    delta = {key: value - before.get(key, 0) for key, value in after.items()}
    assert all(value >= 0 for value in delta.values()), "cache restarted during request"
    return delta


def messages_for(rows, run_id):
    seed = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    values = {key: hashlib.sha256((seed + key).encode()).hexdigest()[:12]
              for key in ("START", "MIDDLE", "END")}
    filler = [f"// fixture {i:06d}: module=cache; test=completed; value={(i * 7919) % 100003}\n"
              for i in range(rows)]
    middle = rows // 2
    content = (f"Synthetic repository {seed}.\nMARKER START={values['START']}\n"
               + "".join(filler[:middle]) + f"MARKER MIDDLE={values['MIDDLE']}\n"
               + "".join(filler[middle:]) + f"MARKER END={values['END']}\n"
               + 'Return only a JSON object with START, MIDDLE and END marker values. Ignore fixture values.')
    return [{"role": "user", "content": content}], values


def token_count(messages, timeout):
    req = urllib.request.Request(BASE.removesuffix("/v1") + "/tokenize", headers=headers(),
                                 data=json.dumps({"model": MODEL, "messages": messages,
                                                  "chat_template_kwargs": {"thinking": False}}).encode())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.load(response)
    assert data["max_model_len"] == 400000, "server is not running the 400000-token recipe"
    return data["count"]


def fit_prompt(target, run_id, timeout):
    low, high, best = 0, 1024, None
    while True:
        messages, expected = messages_for(high, run_id)
        count = token_count(messages, timeout)
        if count > target:
            break
        best = messages, expected, count
        low, high = high + 1, high * 2
        assert high <= target, "unexpectedly few tokens in synthetic fixture"
    while low <= high:
        rows = (low + high) // 2
        messages, expected = messages_for(rows, run_id)
        count = token_count(messages, timeout)
        if count <= target:
            best = messages, expected, count
            low = rows + 1
        else:
            high = rows - 1
    assert best and target - 128 <= best[2] <= target, "could not fit requested prompt budget"
    return best


def answer_matches(content, expected):
    text = (content or "").strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        return json.loads(text) == expected
    except json.JSONDecodeError:
        return False


def cache_evidence(delta):
    loaded = delta.get("lmcache_mp_num_chunks_loaded_total", 0) > 0
    l2 = delta.get("lmcache_mp_l2_load_completed_requests_total", 0) > 0
    return {"ram_to_gpu_observed": loaded, "ssd_to_ram_observed": l2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", type=int, default=384000)
    parser.add_argument("--run-id", required=True, help="unique synthetic prefix identity; reuse across restore stages")
    parser.add_argument("--stage", choices=("warm", "ram-restore", "ssd-restore"), default="warm")
    parser.add_argument("--timeout", type=float, default=1800, help="HTTP socket timeout in seconds")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8081/metrics")
    args = parser.parse_args()
    if not 4096 <= args.prompt_tokens <= 384000 or args.timeout <= 0:
        parser.error("prompt-tokens must be 4096..384000; timeout must be positive")
    messages, expected, count = fit_prompt(args.prompt_tokens, args.run_id, args.timeout)
    print(json.dumps({"stage": args.stage, "prompt_tokens": count, "max_tokens": 16000}), flush=True)
    ranks = (0, 0, 1, 7) if args.stage == "warm" else (0,)
    for index, rank in enumerate(ranks):
        before = metrics(args.metrics_url)
        message, timing = stream_chat(messages, rank=rank, timeout=args.timeout, max_tokens=16000)
        after = metrics(args.metrics_url)
        assert after, "LMCache /metrics lacks the pinned MP counters after generation"
        delta = delta_metrics(before, after)
        valid = answer_matches(message.get("content"), expected)
        usage = timing.pop("usage")
        assert usage.get("prompt_tokens") == count, "tokenize and generation prompt counts differ"
        # Select numeric fields explicitly; never serialize a full API response.
        report = {"request": index, "rank": rank, "correct": valid, **timing,
                  "prompt_tokens": count, "completion_tokens": usage.get("completion_tokens"),
                  "cache_delta": delta, **cache_evidence(delta)}
        print(json.dumps(report, sort_keys=True), flush=True)
        assert valid, "long-context marker retrieval failed"
        if args.stage != "warm":
            assert report["ram_to_gpu_observed"], "RAM-to-GPU restore was not observed"
        if args.stage == "ssd-restore":
            assert report["ssd_to_ram_observed"], "SSD-to-RAM restore was not observed"
    print(json.dumps({"correctness_passed": True, "stage": args.stage,
                      "requires_idle_instance_for_cache_attribution": True}), flush=True)


if __name__ == "__main__":
    main()
