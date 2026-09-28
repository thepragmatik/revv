"""Submit and collect one bounded GPU availability probe from Kaggle."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-gpu-smoke"
SOURCE = Path("experiments/kaggle/smoke.py")
DEST = Path("results/kaggle-smoke.json")
POLL_SECONDS = 15
MAX_WAIT_SECONDS = 12 * 60


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-kaggle-") as temporary:
        work = Path(temporary)
        shutil.copyfile(SOURCE, work / "smoke.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle did not identify an account for the token")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv GPU smoke",
            "code_file": "smoke.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "false",
            "machine_shape": "NvidiaTeslaT4",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        print(f"Submitting private Kaggle script {reference} (300-second runtime cap)", flush=True)
        submitted = api.kernels_push(str(work), timeout="300", acc="NvidiaTeslaT4")
        if getattr(submitted, "error", None):
            raise RuntimeError(f"Kaggle rejected the script: {submitted.error}")
        version = getattr(submitted, "version_number", None)
        pinned_reference = f"{reference}/{version}" if version else reference
        print(f"Submitted {pinned_reference}", flush=True)

        deadline = time.monotonic() + MAX_WAIT_SECONDS
        while time.monotonic() < deadline:
            response = api.kernels_status(pinned_reference)
            status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                log = api.kernels_logs(pinned_reference)
                raise RuntimeError(f"Kaggle GPU probe ended with status {status}; kernel log: {log[-3000:]}")
            time.sleep(POLL_SECONDS)
        else:
            raise TimeoutError("Kaggle probe did not finish within 12 minutes; check Kaggle session status")

        output = work / "output"
        api.kernels_output(pinned_reference, str(output), file_pattern=r"^revv-smoke\.json$")
        source = output / "revv-smoke.json"
        if not source.is_file():
            raise FileNotFoundError("Kaggle finished but did not return revv-smoke.json")
        report = json.loads(source.read_text(encoding="utf-8"))
        if report.get("kind") != "gpu_availability_probe_not_model_benchmark":
            raise ValueError("Unexpected Kaggle probe output")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned_reference, **report}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved GPU availability report to {DEST}", flush=True)


if __name__ == "__main__":
    main()
