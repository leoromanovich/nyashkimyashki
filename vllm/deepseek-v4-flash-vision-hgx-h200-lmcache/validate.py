"""Validate Compose argv, DEP topology and shared LMCache resource budgets."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def compose_args(root, env_file):
    binary = os.environ.get("VLLM_DSV4_COMPOSE_BIN")
    prefix = [binary] if binary else ["docker", "compose"]
    return prefix + ["--project-name", "deepseek-v4-vision-vllm-lmcache-h200",
                     "--file", str(root / "docker-compose.yaml"), "--env-file", str(env_file)]


def parse_options(argv, prefix):
    assert argv[:len(prefix)] == prefix, "unexpected command entry point"
    assert all(isinstance(v, str) and v == v.strip() and "\n" not in v for v in argv)
    out, i = {}, len(prefix)
    while i < len(argv):
        flag = argv[i]
        assert re.fullmatch(r"--[a-z][a-z0-9-]*", flag), "invalid flag/argv boundary"
        assert flag not in out, "duplicate flag: " + flag
        i += 1
        value = True
        if i < len(argv) and not argv[i].startswith("--"):
            value, i = argv[i], i + 1
        out[flag] = value
    return out


def validate(root=ROOT, env_file=None):
    env_file = env_file or root / ".env.example"
    raw = (root / "docker-compose.yaml").read_text()
    blocks = re.findall(r"^    command: >-\n((?:      .*\n)+)", raw, re.M)
    assert len(blocks) == 3, "engine, cache and SMG commands must use folded scalars"
    for block in blocks:
        assert all(line.startswith("      --") and "\\" not in line
                   for line in block.splitlines()[1:])
    command = compose_args(root, env_file)
    subprocess.run(command + ["config", "--quiet"], check=True)
    config = json.loads(subprocess.check_output(command + ["config", "--format", "json"]))
    services = config["services"]
    assert set(services) == {"vllm", "lmcache", "smg", "otel-collector"}
    versions = json.loads((root / "versions.json").read_text())
    assert (root / "Dockerfile").read_text().startswith("FROM " + versions["base_image"] + "\n")
    assert services["vllm"]["image"] == services["lmcache"]["image"]
    for service in (services["vllm"], services["lmcache"]):
        assert service["network_mode"] == "host" and service["ipc"] == "host"
        assert service["platform"] == "linux/amd64"
        assert service["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 8
    v = parse_options(services["vllm"]["command"], ["vllm", "serve", versions["model"]])
    l = parse_options(services["lmcache"]["command"], ["lmcache", "server"])
    required = {
        "--revision": versions["model_revision"], "--api-server-count": "1",
        "--tensor-parallel-size": "1", "--data-parallel-size": "8",
        "--enable-expert-parallel": True, "--moe-backend": "marlin",
        "--max-model-len": "400000", "--kv-cache-dtype": "fp8_ds_mla", "--block-size": "256",
        "--enable-prefix-caching": True, "--enable-chunked-prefill": True,
        "--no-disable-hybrid-kv-cache-manager": True,
        "--disable-chunked-mm-input": True, "--tokenizer-mode": "deepseek_v4",
        "--reasoning-parser": "deepseek_v4", "--tool-call-parser": "deepseek_v4",
        "--enable-auto-tool-choice": True,
    }
    for flag, value in required.items():
        assert v.get(flag) == value, "contract mismatch: " + flag
    excluded = {"--disable-hybrid-kv-cache-manager", "--speculative-config", "--quantization",
                "--enforce-eager", "--kv-offloading-size", "--kv-offloading-backend"}
    assert not excluded.intersection(v), "unqualified feature combination"
    assert v["--all2all-backend"] in {"allgather_reducescatter", "deepep_high_throughput"}
    assert 8 <= int(v["--max-num-seqs"]) <= 64
    assert int(v["--max-num-batched-tokens"]) in {2048, 4096, 8192, 16384}
    assert 400 <= int(v["--max-num-queued-reqs"]) <= 1024
    assert 0.80 <= float(v["--gpu-memory-utilization"]) <= 0.92
    capture = json.loads(v["--compilation-config"])
    assert capture["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert 1 <= capture["max_cudagraph_capture_size"] <= int(v["--max-num-seqs"])
    kv = json.loads(v["--kv-transfer-config"])
    assert kv["kv_connector"] == "LMCacheMPConnector" and kv["kv_role"] == "kv_both"
    assert kv["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_mp_connector"
    extra = kv["kv_connector_extra_config"]
    assert extra["lmcache.mp.host"] == "tcp://127.0.0.1"
    assert extra["lmcache.mp.port"] == int(l["--port"]) == 5555
    assert extra["lmcache.mp.eager_prefetch"] is True
    assert l["--host"] == l["--http-host"] == "127.0.0.1"
    assert l["--http-port"] == "8081" and l["--chunk-size"] == "256"
    assert l["--separate-object-groups"] is True
    assert l["--max-gpu-workers"] == "8" and l["--eviction-policy"] == "LRU"
    assert l["--l2-prefetch-policy"] == "retain"
    assert 64 <= float(l["--l1-size-gb"]) <= 1200
    l2 = json.loads(l["--l2-adapter"])
    assert l2["type"] == "nixl_store_dynamic" and l2["backend"] == "POSIX"
    assert l2["persist_enabled"] is True
    assert l2["backend_params"]["file_path"] == "/lmcache"
    assert l2["backend_params"]["shard_dirs"] == "true"
    assert 1 <= float(l2["backend_params"]["max_capacity_gb"]) <= 8192
    assert l2["eviction"] == {"eviction_policy": "LRU", "trigger_watermark": 0.85, "eviction_ratio": 0.1}
    assert services["vllm"]["environment"]["VLLM_API_KEY"]
    # Inspect the real argv for the inherited gateway command too.
    gateway = parse_options(services["smg"]["command"], [])
    assert gateway["--backend"] == "vllm" and gateway["--policy"] == "cache_aware"
    assert gateway["--worker-urls"] == f"http://127.0.0.1:{v['--port']}"
    assert gateway["--history-backend"] == "none"
    assert all(s["restart"] in {"unless-stopped", "no"} for s in services.values())
    assert len({s["restart"] for s in services.values()}) == 1
    for script in root.glob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
    return config, v, l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    ns = parser.parse_args()
    validate(env_file=ns.env_file)
    print("OK: Compose quiet + JSON argv, TP1/DP8/EP8, shared LMCache budgets, Python syntax")
