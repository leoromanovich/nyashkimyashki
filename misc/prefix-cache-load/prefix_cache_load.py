#!/usr/bin/env python3
"""Small stdlib-only prefix-cache load probe for SGLang/OpenAI chat API."""

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

LOG_LOCK = threading.Lock()


def split_base(url):
    url = url.rstrip("/")
    return (url[:-3], url) if url.endswith("/v1") else (url, url + "/v1")


def http(method, url, body=None, headers=None, timeout=900):
    hdr = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        hdr["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    if headers:
        hdr.update(headers)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(str(e))


def maybe_get(method, url, timeout):
    try:
        return http(method, url, timeout=timeout)
    except Exception:
        return None


def model_id(v1, timeout):
    data = http("GET", v1 + "/models", timeout=timeout)
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError("GET /v1/models returned no models")
    return rows[0].get("id") or rows[0].get("root") or rows[0].get("object")


def make_prefix(group, approx_tokens):
    words = ["function", "module", "class", "import", "return", "cache", "prefix"]
    out = [f"repo_group_{group}", "shared repository context"]
    while len(out) < approx_tokens + 2:
        out.extend(words)
    return " ".join(out[: approx_tokens + 2])


def make_suffix(group, user, batch, approx_tokens):
    out = [f"user={user}", f"group={group}", f"batch={batch}", "generate continuation"]
    while len(out) < approx_tokens + 4:
        out.extend(["analyze", "change", "verify", "answer"])
    return " ".join(out[: approx_tokens + 4])


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[round((len(vals) - 1) * p / 100)]


def ms(x):
    return "n/a" if x is None else f"{x * 1000:.0f}ms"


def load_line(loads):
    rows = (loads or {}).get("loads") or []
    if not rows:
        return ""
    hit = statistics.mean(r.get("cache_hit_rate", 0.0) for r in rows) * 100
    usage = statistics.mean(r.get("token_usage", 0.0) for r in rows) * 100
    waiting = sum(r.get("num_waiting_reqs", 0) for r in rows)
    return f" load_hit={hit:.1f}% token_usage={usage:.1f}% q={waiting}"


def payload(args, model, prefix, suffix):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": suffix},
        ],
        "max_tokens": args.gen_tokens,
        "temperature": args.temperature,
        "stream": False,
        "ignore_eos": not args.respect_eos,
        "min_tokens": 0 if args.respect_eos else args.gen_tokens,
        "return_cached_tokens_details": True,
        "rid": "pcl-" + uuid.uuid4().hex[:12],
    }
    body.update(args.extra_body)
    return body


def log_request(path, row):
    if not path:
        return
    with LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def one(args, v1, model, prefixes, dp_size, group, user, batch, barrier):
    barrier.wait()
    hdr = {}
    if args.sticky_dp and dp_size > 1:
        hdr["X-Data-Parallel-Rank"] = str(group % dp_size)
    body = payload(
        args,
        model,
        prefixes[group],
        make_suffix(group, user, batch, args.suffix_tokens),
    )
    log_request(
        args.request_log,
        {
            "batch": batch,
            "group": group,
            "user": user,
            "method": "POST",
            "url": v1 + "/chat/completions",
            "headers": hdr,
            "body": body,
        },
    )
    if args.dry_run:
        return {"ok": True, "lat": 0.0, "prompt": 0, "out": 0, "cached": None}
    t0 = time.perf_counter()
    try:
        r = http("POST", v1 + "/chat/completions", body, hdr, args.timeout)
        u = r.get("usage") or {}
        d = u.get("prompt_tokens_details") or {}
        return {
            "ok": True,
            "lat": time.perf_counter() - t0,
            "prompt": u.get("prompt_tokens") or 0,
            "out": u.get("completion_tokens") or 0,
            "cached": d.get("cached_tokens"),
        }
    except Exception as e:
        return {
            "ok": False,
            "lat": time.perf_counter() - t0,
            "prompt": 0,
            "out": 0,
            "cached": None,
            "err": str(e),
        }


def run_batch(args, v1, model, prefixes, dp_size, batch, parallel):
    n = args.groups * parallel
    barrier = threading.Barrier(n + 1)
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [
            ex.submit(one, args, v1, model, prefixes, dp_size, g, u, batch, barrier)
            for g in range(args.groups)
            for u in range(parallel)
        ]
        t0 = time.perf_counter()
        barrier.wait()
        rows = [f.result() for f in cf.as_completed(futs)]
    return rows, time.perf_counter() - t0


def summary(rows, wall):
    ok = [r for r in rows if r["ok"]]
    lat = [r["lat"] for r in ok]
    prompt = sum(r["prompt"] for r in ok)
    out = sum(r["out"] for r in ok)
    cached_vals = [r["cached"] for r in ok if r["cached"] is not None]
    cached = sum(cached_vals) if cached_vals else None
    return {
        "ok": len(ok),
        "err": len(rows) - len(ok),
        "wall": wall,
        "req_s": len(ok) / wall if wall else 0,
        "out_s": out / wall if wall else 0,
        "prompt": prompt,
        "out": out,
        "cached": cached,
        "cached_ratio": None if cached is None or not prompt else cached / prompt,
        "p50": pct(lat, 50),
        "p95": pct(lat, 95),
        "max": max(lat) if lat else None,
    }


def print_summary(label, s, loads):
    cr = "n/a" if s["cached_ratio"] is None else f"{s['cached_ratio'] * 100:.1f}%"
    print(
        f"{label}: ok={s['ok']} err={s['err']} wall={s['wall']:.2f}s "
        f"req/s={s['req_s']:.2f} out_tok/s={s['out_s']:.1f} "
        f"p50={ms(s['p50'])} p95={ms(s['p95'])} max={ms(s['max'])} "
        f"cached={cr}{load_line(loads)}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--suffix-tokens", type=int, default=0)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--parallel-per-group", type=int, default=10)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--batch-gap", type=float, default=0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--flush-cache", action="store_true")
    p.add_argument("--flush-timeout", type=float, default=0)
    p.add_argument("--sticky-dp", action="store_true")
    p.add_argument("--respect-eos", action="store_true")
    p.add_argument("--extra-body", default="{}")
    p.add_argument("--jsonl")
    p.add_argument("--request-log", help="append sent chat payloads as JSONL")
    p.add_argument("--dry-run", action="store_true", help="build/log requests without HTTP")
    p.add_argument("--model", help="override model id; default: GET /v1/models")
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.prompt_tokens, args.gen_tokens, args.groups, args.parallel_per_group, args.batches) <= 0:
        raise SystemExit("prompt/gen/groups/parallel/batches must be > 0")
    args.extra_body = json.loads(args.extra_body)

    root, v1 = split_base(args.base_url)
    model = args.model or ("dry-run-model" if args.dry_run else model_id(v1, args.timeout))
    info = {} if args.dry_run else (maybe_get("GET", root + "/server_info", args.timeout) or {})
    dp_size = int(info.get("dp_size") or 1)
    print(
        f"server: model={model} dp={dp_size} "
        f"cache_report={info.get('enable_cache_report')} "
        f"disable_radix_cache={info.get('disable_radix_cache')}",
        flush=True,
    )
    if not info.get("enable_cache_report"):
        print("note: cached_tokens needs SGLang --enable-cache-report", flush=True)

    if args.flush_cache and not args.dry_run:
        http("POST", f"{root}/flush_cache?timeout={args.flush_timeout}", timeout=args.timeout)

    prefixes = [make_prefix(g, args.prompt_tokens) for g in range(args.groups)]
    for i in range(args.warmup):
        rows, wall = run_batch(args, v1, model, prefixes, dp_size, -i - 1, 1)
        bad = [r for r in rows if not r["ok"]]
        if bad:
            raise RuntimeError("warmup failed: " + bad[0]["err"])
        print_summary("warmup", summary(rows, wall), None)

    all_rows, total_wall = [], 0.0
    for b in range(1, args.batches + 1):
        rows, wall = run_batch(args, v1, model, prefixes, dp_size, b, args.parallel_per_group)
        loads = None if args.dry_run else maybe_get("GET", v1 + "/loads?include=core,memory", args.timeout)
        s = summary(rows, wall)
        print_summary(f"batch {b}/{args.batches}", s, loads)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": b, "summary": s, "loads": loads}) + "\n")
        all_rows.extend(rows)
        total_wall += wall
        if args.batch_gap and b != args.batches:
            time.sleep(args.batch_gap)

    s = summary(all_rows, total_wall)
    cr = "n/a" if s["cached_ratio"] is None else f"{s['cached_ratio'] * 100:.1f}%"
    print(f"total: ok={s['ok']} err={s['err']} prompt_tok={s['prompt']} out_tok={s['out']} cached={cr}")
    bad = [r for r in all_rows if not r["ok"]]
    if bad:
        print("first_error:", bad[0]["err"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
