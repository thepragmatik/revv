"""Small GPU availability probe; runs as a private Kaggle script."""

import json
import platform
import time
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available in this Kaggle session")

    torch.manual_seed(17)
    device = torch.device("cuda:0")
    a = torch.randn((512, 512), device=device)
    b = torch.randn((512, 512), device=device)
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(8):
        result = torch.mm(a, b)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    try:
        peak_allocated = torch.cuda.max_memory_allocated(device)
    except RuntimeError:
        peak_allocated = None

    report = {
        "kind": "gpu_availability_probe_not_model_benchmark",
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated_bytes": peak_allocated,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": 17,
        "matrix_shape": [512, 512],
        "timed_multiplications": 8,
        "elapsed_seconds": seconds,
        "checksum": result.sum().item(),
    }
    out = Path("/kaggle/working/revv-smoke.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GPU probe complete: {report['gpu']}; result in {out.name}")


if __name__ == "__main__":
    main()
