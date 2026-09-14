"""Control Center entry point; GPU deployment is explicit."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from validate import ROOT, compose_args, validate


def check_storage(config):
    for name, service in config["services"].items():
        for mount in service.get("volumes", []):
            if mount["type"] != "bind":
                continue
            path = Path(mount["source"])
            if name not in {"vllm", "lmcache"}:
                if not path.exists():
                    raise RuntimeError("missing service bind source: " + str(path))
                continue
            if not path.is_dir():
                raise RuntimeError("create configured directory: " + str(path))
            reserve = 3_000_000_000_000 if name == "lmcache" else 500_000_000_000
            if shutil.disk_usage(path).free < reserve:
                raise RuntimeError("insufficient free storage: " + str(path))


def preflight(config, v, l, command):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("requires the Linux x86_64 HGX host")
    if config["services"]["vllm"]["environment"]["VLLM_API_KEY"].startswith("replace-with-"):
        raise RuntimeError("replace the example API key")
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    if len(rows) != 8 or any("H200" not in r or float(r.split(",")[1]) < 130000 for r in rows):
        raise RuntimeError("expected 8 H200 GPUs with at least 130000 MiB each")
    if any(int(r.split(",")[2].strip().split(".")[0]) < 580 for r in rows):
        raise RuntimeError("CUDA 13 image requires a compatible R580+ driver")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    running = subprocess.check_output(command + ["ps", "--status", "running", "--services"], text=True).splitlines()
    cache_budget = 0 if "lmcache" in running else float(l["--l1-size-gb"]) * 1024**3
    required = cache_budget + 400_000_000_000
    if available < required:
        raise RuntimeError("RAM must cover the shared LMCache budget plus 400 GB headroom")
    check_storage(config)
    image = config["services"]["vllm"]["image"]
    subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--network", "none",
                    "--entrypoint", "python3", image, "/opt/recipe/image_check.py"], check=True)
    # Check CLI against the actual digest-derived image, whose upstream source label is unknown.
    help_text = subprocess.check_output(["docker", "run", "--rm", "--gpus", "all",
                                        "--network", "none", "--entrypoint", "vllm",
                                        image, "serve", "--help=all"], text=True)
    missing = [flag for flag in v if flag not in help_text]
    if missing:
        raise RuntimeError("image lacks requested CLI flags: " + ", ".join(missing))
    print("OK: HGX, RAM, storage, installed connector and image CLI")
    print("Verify the dedicated SSD quota (10 TiB), NVLink topology and GPU cache restore before admission.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["config", "build-image", "preflight", "smoke", "acceptance", "up", "down", "stop", "restart", "logs", "ps"])
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    env_file = Path(os.environ.get("VLLM_DSV4_ENV_FILE", str(ROOT / ".env"))).resolve()
    if ns.action in {"config", "build-image"} and not env_file.exists():
        env_file = ROOT / ".env.example"
    config, v, l = validate(env_file=env_file)
    command = compose_args(ROOT, env_file)
    if ns.action == "config":
        print(f"OK: credentials omitted; DEP8, context {v['--max-model-len']}; "
              f"sequence ceiling {8 * int(v['--max-num-seqs'])}; "
              f"shared RAM {l['--l1-size-gb']} GiB, "
              f"SSD {json.loads(l['--l2-adapter'])['backend_params']['max_capacity_gb']} GiB")
        return
    if ns.action == "build-image":
        subprocess.run(command + ["build", "vllm", *ns.extra], check=True)
        return
    if ns.action == "preflight":
        preflight(config, v, l, command)
        return
    if ns.action in {"smoke", "acceptance"}:
        address = v["--host"]
        if address in {"0.0.0.0", "::"}:
            address = "127.0.0.1"
        env = dict(os.environ, VLLM_API_KEY=config["services"]["vllm"]["environment"]["VLLM_API_KEY"])
        env.setdefault("VLLM_DSV4_BASE_URL", f"http://{address}:{v['--port']}/v1")
        subprocess.run([sys.executable, str(ROOT / f"{ns.action}.py"), *ns.extra], env=env, check=True)
        return
    if ns.action in {"up", "down", "stop", "restart"}:
        if os.environ.get("VLLM_DSV4_CONFIRM") != "mutate-vllm-dsv4-h200":
            raise RuntimeError("set VLLM_DSV4_CONFIRM=mutate-vllm-dsv4-h200 for deployment changes")
    if ns.action in {"up", "restart"}:
        preflight(config, v, l, command)
    subprocess.run(command + [ns.action, *(ns.extra or (["-d", "--no-build"] if ns.action == "up" else []))], check=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
