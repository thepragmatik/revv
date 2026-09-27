"""Submit and collect the preregistered frozen pooled/MaxSim Kaggle pilot."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-frozen-matching"
SOURCE = Path("experiments/kaggle/frozen_matching.py")
DEST = Path("results/kaggle-frozen-matching.json")
WAIT_SECONDS = 14 * 60


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-frozen-") as directory:
        work = Path(directory)
        shutil.copyfile(SOURCE, work / "frozen_matching.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Could not determine Kaggle token account")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv frozen matching",
            "code_file": "frozen_matching.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Submitting private {reference}, 420-second Kaggle cap", flush=True)
        submission = api.kernels_push(str(work), timeout="420", acc="NvidiaTeslaT4")
        if getattr(submission, "error", None):
            raise RuntimeError(f"Kaggle rejected submission: {submission.error}")
        version = getattr(submission, "version_number", None)
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
                raise RuntimeError(f"Kaggle pilot {status}; last log: {str(logs)[-4000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle collector window exceeded: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        result = output / "revv-frozen-matching.json"
        if not result.is_file():
            raise FileNotFoundError("Completed Kaggle kernel has no frozen matching report")
        report = json.loads(result.read_text(encoding="utf-8"))
        if report.get("kind") != "frozen_pretrained_clinc_matching_pilot_not_decision_model_win":
            raise ValueError("Unexpected experiment output kind")
        if report.get("model_revision") != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
            raise ValueError("Unexpected encoder revision")
        if report.get("development_rows") != 1547 or len(report.get("predictions", [])) != 1547:
            raise ValueError("Incomplete fixed development fold")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
