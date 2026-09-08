"""Check installed connector and model features without loading weights."""
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sys


def main():
    import torch
    import lmcache.cuda_ops
    import lmcache.lmcache_native
    versions = json.loads((Path(__file__).parent / "versions.json").read_text())
    assert metadata.version("lmcache") == versions["lmcache_build_version"]
    assert torch.version.cuda and torch.version.cuda.split(".")[0] == "13"
    if "--build-only" in sys.argv:
        print("OK: pinned LMCache and CUDA extensions import; runtime connector probe deferred to GPU preflight")
        return
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.utils import mm_hash_to_token_values
    from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (
        DynamicNixlStoreL2Adapter,
    )
    from vllm.models.deepseek_v4.common import vision

    assert issubclass(LMCacheMPConnector, SupportsHMA), "HMA connector required"
    assert LMCacheMPConnector.get_required_kvcache_layout(None) is None, "preserve V4 NHD layout"
    assert callable(mm_hash_to_token_values), "image-aware cache keys required"
    assert hasattr(DynamicNixlStoreL2Adapter, "submit_store_task")
    assert "vision" in inspect.getfile(vision)
    print("OK: pinned LMCache, CUDA extension, HMA, native V4 layout, MM keys, Vision, NIXL L2")
    print("Installed vLLM:", metadata.version("vllm"), "torch:", metadata.version("torch"))


if __name__ == "__main__":
    main()
