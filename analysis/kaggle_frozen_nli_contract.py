"""Submit and collect one bounded private Kaggle frozen NLI control."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-frozen-nli-contract"
SOURCE = Path("experiments/kaggle/frozen_nli_contract.py")
DEST = Path("results/kaggle-frozen-nli-contract.json")


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-nli-contract-") as directory:
        work = Path(directory)
        shutil.copyfile(SOURCE, work / "frozen_nli_contract.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text())
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle account not found")
        reference = f"{owner}/{SLUG}"
        metadata.update({"id": reference, "title": "revv frozen nli contract", "code_file": "frozen_nli_contract.py",
                         "language": "python", "kernel_type": "script", "is_private": "true",
                         "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Submitting private {reference}, 600-second Kaggle cap", flush=True)
        submission = api.kernels_push(str(work), timeout="600", acc="NvidiaTeslaT4")
        if getattr(submission, "error", None):
            raise RuntimeError(f"Kaggle rejected frozen NLI pilot: {submission.error}")
        version = getattr(submission, "version_number", None)
        pinned = f"{reference}/{version}" if version else reference
        deadline = time.monotonic() + 14 * 60
        while time.monotonic() < deadline:
            response = api.kernels_status(pinned)
            status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                logs = api.kernels_logs(pinned)
                raise RuntimeError(f"Kaggle frozen NLI {status}; log: {str(logs)[-4000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle NLI collector window exceeded: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        source = output / "revv-frozen-nli-contract.json"
        if not source.is_file():
            raise FileNotFoundError("Completed Kaggle run has no frozen NLI report")
        report = json.loads(source.read_text())
        if report.get("kind") != "frozen_nli_cross_on_selected_contract_spans_dev_control_not_local_win":
            raise ValueError("Unexpected NLI output kind")
        if report.get("development_documents") != 61 or report.get("nli_pairs") != 11645:
            raise ValueError("Incomplete NLI fixed fold or pair count")
        DEST.parent.mkdir(exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n")
        print(f"Saved frozen NLI report {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
