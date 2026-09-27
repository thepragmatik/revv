"""Submit and collect the registered CLINC frozen prototype screen."""

import json
import shutil
import tempfile
import time
from pathlib import Path

from kaggle import api


SLUG = "revv-prototype-screen"
DEST = Path("results/kaggle-prototype-screen.json")


def main():
    api.authenticate()
    with tempfile.TemporaryDirectory(prefix="revv-prototype-") as directory:
        work = Path(directory)
        for name in ("prototype_screen.py", "frozen_matching.py"):
            shutil.copyfile(Path("experiments/kaggle") / name, work / name)
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["id"].split("/", 1)[0]
        if not owner or owner == "None":
            raise RuntimeError("Kaggle token account missing")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv prototype screen",
            "code_file": "prototype_screen.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Submitting private frozen prototype screen {reference}, 420-second cap", flush=True)
        submitted = api.kernels_push(str(work), timeout="420", acc="NvidiaTeslaT4")
        if getattr(submitted, "error", None):
            raise RuntimeError(f"Kaggle rejected prototype screen: {submitted.error}")
        version = getattr(submitted, "version_number", None)
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
                raise RuntimeError(f"Kaggle screen {status}; last log: {str(logs)[-4000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle collector window exceeded: {pinned}")
        output = work / "output"
        api.kernels_output(pinned, str(output))
        path = output / "revv-prototype-screen.json"
        if not path.is_file():
            raise FileNotFoundError("Completed Kaggle kernel lacks prototype report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("kind") != "frozen_clinc_label_vs_prototype_development_not_model_win":
            raise ValueError("Unexpected prototype report kind")
        if report.get("model_revision") != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
            raise ValueError("Unexpected checkpoint revision")
        if report.get("development_rows") != 1547 or len(report.get("predictions", [])) != 1547:
            raise ValueError("Unexpected development fold")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({"kaggle_ref": pinned, **report}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {DEST} from {pinned}", flush=True)


if __name__ == "__main__":
    main()
