"""Submit and collect one bounded private Kaggle gold-span stance diagnostic."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SOURCE = Path("experiments/kaggle/nli_gold_stance_diagnostic.py")
DEST = Path("results/kaggle-nli-gold-stances.json")
SLUG = "revv-nli-gold-stances"


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-nli-oracle-") as directory:
        work = Path(directory)
        shutil.copyfile(SOURCE, work / "nli_gold_stance_diagnostic.py")
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text())
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle account missing")
        ref = f"{owner}/{SLUG}"
        metadata.update({"id": ref, "title": "revv nli gold stances", "code_file": "nli_gold_stance_diagnostic.py",
                         "language": "python", "kernel_type": "script", "is_private": "true",
                         "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Submitting private {ref}, 300-second Kaggle kernel cap", flush=True)
        submission = api.kernels_push(str(work), timeout="300", acc="NvidiaTeslaT4")
        if getattr(submission, "error", None):
            raise RuntimeError(f"Kaggle rejected oracle diagnostic: {submission.error}")
        version = getattr(submission, "version_number", None)
        pinned = f"{ref}/{version}" if version else ref
        deadline = time.monotonic() + 10 * 60
        while time.monotonic() < deadline:
            response = api.kernels_status(pinned)
            status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
            print(f"Kaggle status: {status}", flush=True)
            if status == "COMPLETE":
                break
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                logs = api.kernels_logs(pinned)
                raise RuntimeError(f"Kaggle oracle {status}; log: {str(logs)[-4000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle oracle collector window exceeded: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        source = output / "revv-nli-gold-stances.json"
        if not source.is_file():
            raise FileNotFoundError("Completed Kaggle kernel lacks diagnostic report")
        report = json.loads(source.read_text())
        if report.get("kind") != "frozen_nli_contract_gold_span_stance_oracle_diagnostic_not_deployable":
            raise ValueError("Unexpected oracle report kind")
        if report.get("positive_decisions") != 614 or report.get("test_documents_examined") != 0:
            raise ValueError("Incomplete or unlocked oracle report")
        DEST.parent.mkdir(exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n")
        print(f"Saved {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
