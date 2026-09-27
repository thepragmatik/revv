"""Submit and collect the registered synthetic forward-pass pilot on Kaggle."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-forward-pilot"
SOURCE = Path("experiments/kaggle/forward_pilot.py")
DEST = Path("results/kaggle-forward-pilot.json")
WAIT_SECONDS = 12 * 60


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-forward-") as temporary:
        work = Path(temporary)
        shutil.copyfile(SOURCE, work / "forward_pilot.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle did not identify the token account")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv bounded forward pilot",
            "code_file": "forward_pilot.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "false",
            "machine_shape": "NvidiaTeslaT4",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Submitting private {reference} with a 300-second kernel cap", flush=True)
        submitted = api.kernels_push(str(work), timeout="300", acc="NvidiaTeslaT4")
        if getattr(submitted, "error", None):
            raise RuntimeError(f"Kaggle rejected submission: {submitted.error}")
        version = getattr(submitted, "version_number", None)
        pinned = f"{reference}/{version}" if version else reference
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            response = api.kernels_status(pinned)
            status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                logs = api.kernels_logs(pinned)
                raise RuntimeError(f"Kaggle status {status}; final log excerpt: {str(logs)[-3000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle did not finish within {WAIT_SECONDS} seconds; inspect {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        result = output / "revv-forward-pilot.json"
        if not result.is_file():
            raise FileNotFoundError("Completed Kaggle run did not return revv-forward-pilot.json")
        report = json.loads(result.read_text(encoding="utf-8"))
        if report.get("kind") != "synthetic_t4_forward_crossover_not_quality_or_cpu_benchmark":
            raise ValueError("Unexpected experiment output")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
