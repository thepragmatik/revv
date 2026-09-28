"""Submit a short private Kaggle CPU internet preflight and collect its report."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-internet-preflight"
SOURCE = Path("experiments/kaggle/internet_preflight.py")
DEST = Path("results/kaggle-internet-preflight.json")


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-preflight-") as directory:
        work = Path(directory)
        shutil.copyfile(SOURCE, work / "internet_preflight.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Could not determine Kaggle token account")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv internet preflight",
            "code_file": "internet_preflight.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_internet": "true",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Submitting private CPU preflight {reference}, 180-second cap", flush=True)
        submitted = api.kernels_push(str(work), timeout="180")
        if getattr(submitted, "error", None):
            raise RuntimeError(f"Kaggle rejected preflight: {submitted.error}")
        version = getattr(submitted, "version_number", None)
        pinned = f"{reference}/{version}" if version else reference
        deadline = time.monotonic() + 8 * 60
        while time.monotonic() < deadline:
            status_response = api.kernels_status(pinned)
            status = getattr(status_response.status, "name", str(status_response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                logs = api.kernels_logs(pinned)
                raise RuntimeError(f"Kaggle preflight {status}; last log: {str(logs)[-3000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle preflight exceeded collector window: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        source = output / "revv-internet-preflight.json"
        if not source.is_file():
            raise FileNotFoundError("Kaggle completed but did not return preflight report")
        result = json.loads(source.read_text(encoding="utf-8"))
        if result.get("kind") != "kaggle_cpu_external_access_preflight_not_model_result":
            raise ValueError("Unexpected preflight report kind")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **result}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved preflight report {DEST}; ready={result['ready_for_gpu_pilot']}", flush=True)
        if not result["ready_for_gpu_pilot"]:
            raise RuntimeError("Kaggle lacks transformers; prepare/install it on CPU before GPU pilot")


if __name__ == "__main__":
    main()
