"""Nix-backed lifecycle for the HGX recipe; no automatic production launch."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from validate import IMAGE, SOURCE_REVISION, compose_args, validate

ROOT = Path(__file__).resolve().parent


def preflight(config, args):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("preflight requires the Linux x86_64 HGX host")
    for key in ("--api-key", "--admin-api-key"):
        if args[key].startswith("replace-with-"):
            raise RuntimeError("replace example credentials in the external env file")
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    if len(rows) != 8 or any("H200" not in row or float(row.split(",")[1]) < 130000 for row in rows):
        raise RuntimeError("expected 8 H200 GPUs with at least 130000 MiB each")
    # Image CUDA is 13.0.3. The container runtime remains the driver-compatibility authority.
    if any(int(row.split(",")[2].strip().split(".")[0]) < 580 for row in rows):
        raise RuntimeError("this CUDA 13 image requires an appropriate R580+ driver")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 1_500_000_000_000:
        raise RuntimeError("baseline requires >=1.5 TB MemAvailable; review host use and HiCache ratio")
    service = config["services"]["sglang"]
    for mount in service["volumes"]:
        if mount["type"] != "bind":
            continue
        directory = Path(mount["source"])
        if not directory.is_dir():
            raise RuntimeError("create the configured bind directory: " + str(directory))
        minimum = 4_000_000_000_000 if mount["target"] == "/hicache" else 500_000_000_000
        if shutil.disk_usage(directory).free < minimum:
            raise RuntimeError("insufficient free space at " + str(directory))
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE]))[0]
    if info["Config"].get("Labels", {}).get("org.opencontainers.image.revision") != SOURCE_REVISION:
        raise RuntimeError("unexpected image source revision")
    print("OK: Compose, credentials, 8xH200, driver, RAM, storage and local image revision")
    print("GPU inference/vision/cache correctness still requires smoke and workload acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["config", "preflight", "smoke", "pull", "up", "down", "stop", "restart", "logs", "ps"])
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    env_file = Path(os.environ.get("DSV4_ENV_FILE", str(ROOT / ".env"))).resolve()
    if ns.action == "config" and not env_file.exists():
        env_file = ROOT / ".env.example"
    config, args = validate(ROOT, env_file)
    if ns.action == "config":
        print("OK: Compose schema and resolved argv; credentials omitted")
        dp = int(args["--dp-size"])
        print(f"TP8/DP8/DPA/EP1: running {args['--max-running-requests']} total / "
              f"{int(args['--max-running-requests']) // dp} per rank; "
              f"prefill {int(args['--chunked-prefill-size']) // dp} tokens per rank; "
              f"queue {args['--max-queued-requests']} per rank")
        return
    if ns.action == "preflight":
        preflight(config, args)
        return
    if ns.action == "smoke":
        env = dict(os.environ, SGLANG_API_KEY=args["--api-key"])
        port = config["services"]["sglang"]["ports"][0]["published"]
        address = config["services"]["sglang"]["ports"][0].get("host_ip", "127.0.0.1")
        if address in ("0.0.0.0", "::"):
            address = "127.0.0.1"
        env.setdefault("DSV4_BASE_URL", f"http://{address}:{port}/v1")
        subprocess.run([sys.executable, str(ROOT / "smoke.py"), *ns.extra], env=env, check=True)
        return
    if ns.action in ("pull", "up", "down", "stop", "restart"):
        if os.environ.get("DSV4_CONFIRM") != "mutate-dsv4-vision-h200":
            raise RuntimeError("set DSV4_CONFIRM=mutate-dsv4-vision-h200 for runtime changes")
    if ns.action in ("up", "restart"):
        preflight(config, args)
    extra = ns.extra or (["-d"] if ns.action == "up" else [])
    subprocess.run(compose_args(ROOT, env_file) + [ns.action, *extra], check=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
