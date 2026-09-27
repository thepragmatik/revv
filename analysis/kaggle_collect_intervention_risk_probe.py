"""Collect an existing queued Kaggle version without submitting new compute."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from kaggle_intervention_risk_probe import (
    EXPECTED_GENERATOR_SHA256,
    EXPECTED_SCREEN_SHA256,
    GENERATOR,
    SOURCE,
    validate_report,
)


KAGGLE_REF = "rathworx/revv-intervention-risk-probe/1"
SOURCE_ACTION_RUN = 36314796409
DEST = Path("results/kaggle-intervention-risk-probe.json")
COLLECTOR_SECONDS = 25 * 60


def main() -> None:
    from kaggle import api

    generator_sha = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    screen_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if generator_sha != EXPECTED_GENERATOR_SHA256 or screen_sha != EXPECTED_SCREEN_SHA256:
        raise ValueError("The registered generator or scoring screen changed")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN is missing")
    api.authenticate()

    deadline = time.monotonic() + COLLECTOR_SECONDS
    while time.monotonic() < deadline:
        response = api.kernels_status(KAGGLE_REF)
        status = getattr(response.status, "name", str(response.status)).split(".")[-1].upper()
        print(f"Existing Kaggle kernel status: {status}", flush=True)
        if status == "COMPLETE":
            break
        if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            logs = api.kernels_logs(KAGGLE_REF)
            raise RuntimeError(f"Existing Kaggle screen {status}; log tail: {str(logs)[-3000:]}")
        time.sleep(15)
    else:
        raise TimeoutError(f"Existing Kaggle kernel remained queued/running: {KAGGLE_REF}")

    with tempfile.TemporaryDirectory(prefix="revv-intervention-collect-") as directory:
        output = Path(directory) / "output"
        api.kernels_output(KAGGLE_REF, str(output))
        report_path = output / "revv-intervention-risk-probe.json"
        if not report_path.is_file():
            raise FileNotFoundError("Completed Kaggle kernel has no intervention-risk output")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation = validate_report(report, generator_sha)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({
            "kaggle_ref": KAGGLE_REF,
            "github_sha": os.environ.get("GITHUB_SHA"),
            "recovered_from_action_run": SOURCE_ACTION_RUN,
            "screen_source_sha256": screen_sha,
            "generator_sha256": generator_sha,
            "orchestrator_validation": validation,
            **report,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"kaggle_ref": KAGGLE_REF, "orchestrator_validation": validation},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
