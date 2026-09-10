"""Daemon-free Compose/argv contract. Never print resolved credentials."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess

IMAGE = "lmsysorg/sglang:dev-dsv4-flash-vision@sha256:44a113290011bf87fcfeabc2ed94966bfc90f91b2dbe6239a69de3f35e686cba"
REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
SOURCE_REVISION = "40b3e15ddbd9a1067e181283d9900dd3f4d76ed7"


def compose_args(directory, env_file):
    executable = os.environ.get("DSV4_COMPOSE_BIN")
    prefix = [executable] if executable else ["docker", "compose"]
    return prefix + ["--project-name", "deepseek-v4-flash-vision-h200",
            "--file", str(directory / "docker-compose.yaml"), "--env-file", str(env_file)]


def resolve(directory, env_file):
    command = compose_args(directory, env_file)
    subprocess.run(command + ["config", "--quiet"], check=True)
    return json.loads(subprocess.check_output(command + ["config", "--format", "json"]))


def options(command):
    assert command[:3] == ["python3", "-m", "sglang.launch_server"]
    result = {}
    index = 3
    while index < len(command):
        flag = command[index]
        assert re.fullmatch(r"--[a-z][a-z0-9-]*", flag), "invalid flag or broken argv"
        assert flag not in result, "duplicate flag"
        index += 1
        value = True
        if index < len(command) and not command[index].startswith("--"):
            value = command[index]
            assert value == value.strip() and "\n" not in value, "whitespace in argv"
            index += 1
        result[flag] = value
    return result


def validate(directory, env_file):
    raw = (directory / "docker-compose.yaml").read_text()
    block = re.search(r"^    command: >-\n((?:      .*\n)+)", raw, re.M)
    assert block, "command must be a folded scalar >-"
    lines = block.group(1).splitlines()
    assert lines[0] == "      python3 -m sglang.launch_server"
    assert all(line.startswith("      --") and "\\" not in line for line in lines[1:])
    config = resolve(directory, env_file)
    service = config["services"]["sglang"]
    assert service["build"]["args"]["SGLANG_BASE_IMAGE"] == IMAGE and service["platform"] == "linux/amd64"
    args = options(service["command"])
    required = {
        "--model-path": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "--revision": REVISION,
        "--tp-size": "8", "--context-length": "200000", "--page-size": "256",
        "--dp-size": "8", "--enable-dp-attention": True, "--ep-size": "1",
        "--moe-a2a-backend": "none", "--load-balance-method": "total_tokens",
        "--moe-runner-backend": "flashinfer_mxfp4", "--kv-cache-dtype": "fp8_e4m3",
        "--reasoning-parser": "deepseek-v4", "--tool-call-parser": "deepseekv4",
        "--disable-shared-experts-fusion": True, "--enable-hierarchical-cache": True,
        "--hicache-io-backend": "direct", "--hicache-mem-layout": "page_first_direct",
        "--hicache-write-policy": "write_through_selective", "--hicache-storage-backend": "file",
        "--hicache-storage-prefetch-policy": "timeout", "--enable-metrics": True,
        "--enable-cache-report": True,
    }
    for flag, value in required.items():
        assert args.get(flag) == value, "contract mismatch: " + flag
    prohibited = {"--speculative-algorithm", "--hicache-size", "--disable-radix-cache",
                  "--enable-prefill-cp", "--enable-dsa-prefill-context-parallel",
                  "--enable-deepseek-v4-fp4-indexer", "--quantization"}
    assert not prohibited.intersection(args), "flags outside the DPA throughput contract"
    dp = int(args["--dp-size"])
    running = int(args["--max-running-requests"])
    assert 64 <= running <= 512 and running % dp == 0, "running CLI budget must divide evenly across DP8"
    assert 16 <= int(args["--max-queued-requests"]) <= 128, "queue limit is per DP rank"
    assert 1 <= int(args["--cuda-graph-max-bs-decode"]) <= running // dp, "graph cap exceeds per-rank running budget"
    assert int(args["--chunked-prefill-size"]) in (16384, 32768, 65536), "prefill CLI budget is divided by DP8"
    assert 0.70 <= float(args["--mem-fraction-static"]) <= 0.90
    assert 0 < float(args["--hicache-ratio"]) <= 1.5
    for flag in ("--api-key", "--admin-api-key"):
        assert isinstance(args.get(flag), str) and args[flag], "missing auth"
    assert args["--api-key"] != args["--admin-api-key"], "use distinct auth keys"
    env = service["environment"]
    assert env["SGLANG_DSV4_REASONING_EFFORT"] == "high"
    for key in ("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE"):
        assert re.fullmatch(r"[1-9][0-9]*(?:[kMGT]|[kMGT]i)?", str(env[key])), "invalid file-cache size"
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 8
    assert service["restart"] == "unless-stopped", "published recipe must retain its production restart policy"
    for script in directory.glob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
    return config, args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--env-file", type=Path)
    ns = parser.parse_args()
    validate(ns.directory, ns.env_file or ns.directory / ".env.example")
    print("OK: Compose config --quiet, JSON argv, deployment contract, Python syntax")
