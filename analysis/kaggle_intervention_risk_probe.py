"""Submit and collect one bounded private Kaggle intervention-risk screen."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

SLUG = "revv-intervention-risk-probe"
SOURCE = Path("experiments/kaggle/intervention_risk_probe.py")
GENERATOR = Path("analysis/intervention_probe.py")
DEST = Path("results/kaggle-intervention-risk-probe.json")
EXPECTED_KIND = "frozen_synthetic_intervention_risk_vs_confidence_screen_not_benchmark_or_model_win"
EXPECTED_GENERATOR_SHA256 = "0203b3157c322ecf5050af898b71ba8a78512e0bdb1f05e8c42a155c9a5422ae"
EXPECTED_RECORDS_SHA256 = "6612571e3625e412eb24f3549e0d21dd3bcc0a609990a13e34ed3314aaa794d1"
EXPECTED_SCREEN_SHA256 = "e73d3784f1dd000b19aeb266734864593e5f450a8a7d04f3bde305a9421aec9c"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_standalone_script(screen_source: str, generator_source: str, generator_sha256: str) -> str:
    """Bundle the pinned generator into Kaggle's single executable code file."""
    _require(generator_source.count("\ndef run(") == 1,
             "Could not identify the generator library/CLI boundary")
    _require(screen_source.count("from __future__ import annotations\n") == 1,
             "Unexpected screen future-import layout")
    _require(screen_source.count("from intervention_probe import generate_state\n") == 1,
             "Unexpected screen generator import")
    old_hash_block = (
        "generator_path = Path(__import__(\"intervention_probe\").__file__)\n"
        "    generator_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()"
    )
    _require(screen_source.count(old_hash_block) == 1,
             "Unexpected screen generator-hash block")

    generator_core = generator_source.split("\ndef run(", 1)[0].rstrip()
    screen_body = screen_source.replace("from __future__ import annotations\n", "", 1)
    screen_body = screen_body.replace("from intervention_probe import generate_state\n", "", 1)
    screen_body = screen_body.replace(old_hash_block, "generator_sha = GENERATOR_SHA256", 1)
    bundle = (
        generator_core
        + "\n\nGENERATOR_SHA256 = "
        + json.dumps(generator_sha256)
        + "\n\n"
        + screen_body
    )
    _require("intervention_probe" not in bundle,
             "Kaggle script still depends on an unbundled sibling module")
    _require("def generate_state(" in bundle,
             "Standalone Kaggle script is missing the frozen state generator")
    _require(bundle.count('if __name__ == "__main__":') == 1,
             "Standalone Kaggle script has an unexpected entry point count")
    compile(bundle, "<revv-kaggle-intervention-risk-bundle>", "exec")
    return bundle


def validate_report(report: dict, generator_sha256: str) -> dict:
    """Reject incomplete or mismatched kernel output; preserve low-power results."""
    _require(report.get("kind") == EXPECTED_KIND, "Unexpected intervention screen output kind")
    _require(report.get("seed") == 20260928, "Unexpected primary seed")
    _require(report.get("states") == 1024, "Incomplete generated state set")
    _require(report.get("composition_group_size") == 8, "Unexpected group size")
    _require(report.get("generator_sha256") == generator_sha256,
             "Kaggle used a different intervention generator")
    _require(report.get("generated_records_sha256") == EXPECTED_RECORDS_SHA256,
             "Kaggle generated different records than the pre-score audit")
    _require(generator_sha256 == EXPECTED_GENERATOR_SHA256,
             "Local generator changed after preregistration; revise before running")
    _require(report.get("sentence_model") == "sentence-transformers/all-MiniLM-L6-v2",
             "Unexpected sentence encoder")
    _require(report.get("sentence_revision") == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
             "Unexpected sentence encoder revision")
    _require(report.get("nli_model") == "cross-encoder/nli-MiniLM2-L6-H768",
             "Unexpected NLI verifier")
    _require(report.get("nli_revision") == "c4d86af4493123990d7762712de9ed730c876161",
             "Unexpected NLI verifier revision")

    split_groups = report.get("groups_by_split", {})
    _require(all(split_groups.get(name, 0) >= 2 for name in ("fit", "calibration", "heldout")),
             "Missing independent composition groups in a split")
    _require(sum(split_groups.values()) == report.get("composition_groups"),
             "Composition group split counts do not sum to total")

    heldout = report.get("heldout_results", {})
    selection = heldout.get("selection", {})
    q20 = selection.get("q20", {})
    risk = q20.get("intervention_risk", {})
    confidence = q20.get("confidence_only", {})
    risk25, confidence25 = risk.get("0.25", {}), confidence.get("0.25", {})
    _require(risk25 and confidence25, "Missing primary Q20 comparison")
    changed_fields = risk25.get("changed_fields", 0)
    changed_groups = risk25.get("changed_rule_groups", 0)
    recall_delta = heldout.get("selection", {}).get("q20", {}).get(
        "cluster_bootstrap_recall_comparisons_at_25pct", {}).get(
            "intervention_risk_minus_confidence", {})
    interval = recall_delta.get("cluster_bootstrap_95pct_interval")
    _require(recall_delta.get("changed_fields") == changed_fields,
             "Primary changed-field counts disagree")
    _require(isinstance(interval, list) and len(interval) == 2,
             "Missing group-bootstrap interval for primary comparison")
    _require(recall_delta.get("bootstrap_replicates", 0) >= 1900,
             "Too few bootstrap samples in primary comparison")

    q20_n = heldout.get("update_quality", {}).get("q20", {}).get(
        "cached_pre_edit_prediction", {}).get("n")
    expected_q20_n = report.get("states_by_split", {}).get("heldout", 0) * 20
    _require(q20_n == expected_q20_n and q20_n > 0,
             "Held-out Q20 evaluation count is incomplete")
    _require(report.get("nli_heldout_timing", {}).get("pairs_per_route") == q20_n,
             "Held-out NLI scoring did not cover the expected fields")

    accuracy = heldout.get("paired_accuracy_deltas", {}).get(
        "q20_risk_gate25_minus_cached", {})
    _require(isinstance(accuracy.get("cluster_bootstrap_95pct_interval"), list),
             "Missing paired update-quality interval")
    cpu = report.get("kaggle_cpu_reference_profile", {})
    latency = cpu.get("request_latency", {})
    _require(all(route in latency and "20" in latency[route] for route in
                 ("direct_full", "intervention_risk_gate25")),
             "Missing Q20 CPU reference timing")

    sample_size_pass = changed_fields >= 50 and changed_groups >= 10
    recall_pass = (recall_delta.get("delta_recall_a_minus_b", -1) >= 0.10
                   and interval[0] > 0)
    return {
        "validated": True,
        "primary_sample_size_gate_pass": sample_size_pass,
        "primary_recall_gate_pass": recall_pass if sample_size_pass else None,
        "primary_gate_status": ("inconclusive_underpowered" if not sample_size_pass
                                 else "pass" if recall_pass else "fail"),
        "heldout_q20_changed_fields": changed_fields,
        "heldout_q20_changed_rule_groups": changed_groups,
        "q20_risk_minus_confidence_recall_delta": recall_delta.get("delta_recall_a_minus_b"),
        "q20_risk_minus_confidence_group_bootstrap_95pct": interval,
        "q20_risk_gate_minus_cached_accuracy_delta": accuracy.get("delta_accuracy_a_minus_b"),
        "q20_risk_gate_minus_cached_accuracy_group_bootstrap_95pct": accuracy.get(
            "cluster_bootstrap_95pct_interval"),
        "q20_risk_gate25_cpu_p50_ms": latency["intervention_risk_gate25"]["20"].get("p50_ms"),
        "q20_direct_full_cpu_p50_ms": latency["direct_full"]["20"].get("p50_ms"),
        "note": "Screening result only; synthetic grammar and Kaggle CPU do not establish novelty, transfer, or target-device speed.",
    }


def main() -> None:
    from kaggle import api

    if not SOURCE.is_file() or not GENERATOR.is_file():
        raise FileNotFoundError("Run from the repository root")
    generator_sha256 = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    screen_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    _require(screen_sha256 == EXPECTED_SCREEN_SHA256,
             "Experiment screen changed after preregistration")
    _require(generator_sha256 == EXPECTED_GENERATOR_SHA256,
             "Generator hash differs from the frozen preregistration")
    _require(bool(os.environ.get("KAGGLE_API_TOKEN")), "KAGGLE_API_TOKEN is missing")
    api.authenticate()

    with tempfile.TemporaryDirectory(prefix="revv-intervention-risk-") as directory:
        work = Path(directory)
        screen_source = SOURCE.read_text(encoding="utf-8")
        generator_source = GENERATOR.read_text(encoding="utf-8")
        bundle = build_standalone_script(screen_source, generator_source, generator_sha256)
        bundle_bytes = bundle.encode("utf-8")
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        (work / "intervention_risk_probe.py").write_bytes(bundle_bytes)
        metadata_path = Path(api.kernels_initialize(str(work)))
        metadata = json.loads(metadata_path.read_text())
        owner = metadata["id"].split("/", 1)[0]
        _require(bool(owner) and owner != "None", "Kaggle account not found")
        reference = f"{owner}/{SLUG}"
        metadata.update({
            "id": reference,
            "title": "revv intervention risk probe",
            "code_file": "intervention_risk_probe.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Submitting private {reference}, 600-second Kaggle cap, script sha256 {bundle_sha256}", flush=True)
        submission = api.kernels_push(str(work), timeout="600", acc="NvidiaTeslaT4")
        if getattr(submission, "error", None):
            raise RuntimeError(f"Kaggle rejected intervention screen: {submission.error}")
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
                raise RuntimeError(f"Kaggle intervention screen {status}; log tail: {str(logs)[-3000:]}")
            time.sleep(15)
        else:
            raise TimeoutError(f"Kaggle collector window exceeded: {pinned}")

        output = work / "output"
        api.kernels_output(pinned, str(output))
        result_path = output / "revv-intervention-risk-probe.json"
        if not result_path.is_file():
            raise FileNotFoundError("Completed Kaggle run has no intervention-risk report")
        report = json.loads(result_path.read_text())
        validation = validate_report(report, generator_sha256)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps({
            "kaggle_ref": pinned,
            "github_sha": os.environ.get("GITHUB_SHA"),
            "screen_source_sha256": screen_sha256,
            "generator_sha256": generator_sha256,
            "kaggle_script_sha256": bundle_sha256,
            "kaggle_script_bytes": len(bundle_bytes),
            "kaggle_script_bundle_format": "standalone_concat_v1",
            "orchestrator_validation": validation,
            **report,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"kaggle_ref": pinned, "orchestrator_validation": validation},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
