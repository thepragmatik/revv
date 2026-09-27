"""Submit a bounded private Kaggle CPU preflight for frozen NLI controls."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-nli-access-preflight"
SOURCE = Path("experiments/kaggle/nli_access_preflight.py")
DEST = Path("results/kaggle-nli-access-preflight.json")


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-nli-preflight-") as directory:
        work = Path(directory)
        shutil.copyfile(SOURCE, work / "nli_access_preflight.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text())
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle account not found")
        reference = f"{owner}/{SLUG}"
        metadata.update({"id": reference, "title": "revv nli access preflight", "code_file": "nli_access_preflight.py",
                         "language": "python", "kernel_type": "script", "is_private": "true",
                         "enable_gpu": "false", "enable_internet": "true"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Submitting CPU-only private {reference}; 180-second kernel cap", flush=True)
        submission = api.kernels_push(str(work), timeout="180")
        if getattr(submission, "error", None):
            raise RuntimeError(f"Kaggle rejected NLI preflight: {submission.error}")
        version = getattr(submission, "version_number", None)
        pinned = f"{reference}/{version}" if version else reference
        deadline = time.monotonic() + 8 * 60
        while time.monotonic() < deadline:
            response = api.kernels_status(pinned)
            status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                logs = api.kernels_logs(pinned)
                raise RuntimeError(f"Kaggle preflight {status}; log: {str(logs)[-3500:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle CPU preflight exceeded collector window: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        source = output / "revv-nli-access-preflight.json"
        if not source.is_file():
            raise FileNotFoundError("Kaggle CPU preflight has no report")
        report = json.loads(source.read_text())
        if report.get("kind") != "kaggle_cpu_nli_access_preflight_no_gpu_no_model_weights" or not report.get("ready_for_frozen_gpu_pilot"):
            raise ValueError("Unexpected NLI access report")
        DEST.parent.mkdir(exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n")
        print(f"Saved {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
