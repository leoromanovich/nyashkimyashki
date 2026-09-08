"""Build LMCache CUDA extensions against the serving image's torch ABI."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def critical_versions():
    protected = {"torch", "torchvision", "torchaudio", "triton", "vllm", "transformers"}
    return {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"].lower().replace("_", "-") in protected
        or dist.metadata["Name"].lower().startswith(("nvidia-", "flashinfer"))
    }


def main():
    import torch
    versions = json.loads((ROOT / "versions.json").read_text())
    assert shutil.which("nvcc"), "base image must include CUDA compiler"
    before = critical_versions()
    assert "torch" in before and "vllm" in before
    with tempfile.TemporaryDirectory(prefix="build-lmcache-") as temporary:
        work = Path(temporary)
        archive = work / "source.tar.gz"
        url = "https://codeload.github.com/LMCache/LMCache/tar.gz/" + versions["lmcache_revision"]
        subprocess.run(["curl", "-fSL", "--retry", "3", url, "-o", str(archive)], check=True)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == versions["lmcache_archive_sha256"]
        with tarfile.open(archive) as package:
            package.extractall(work / "source", filter="data")
        source = next((work / "source").iterdir())
        constraints = work / "constraints.txt"
        constraints.write_text("".join(f"{name}=={value}\n" for name, value in sorted(before.items())))
        install = ["uv", "pip", "install", "--system", "--constraint", str(constraints)]
        subprocess.run(install + ["-r", str(source / "requirements/build.txt")], check=True)
        env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION=versions["lmcache_build_version"],
                   ENABLE_CXX11_ABI=str(int(torch.compiled_with_cxx11_abi())))
        subprocess.run(install + ["--no-build-isolation", str(source) + "[nixl]"], env=env, check=True)
    assert critical_versions() == before, "serving/CUDA dependency versions changed"
    (ROOT / "build-packages.json").write_text(json.dumps({
        "base_packages": before, "lmcache": metadata.version("lmcache"),
        "nixl": metadata.version("nixl"),
    }, indent=2) + "\n")
    subprocess.run(["python3", str(ROOT / "image_check.py"), "--build-only"], check=True)


if __name__ == "__main__":
    main()
