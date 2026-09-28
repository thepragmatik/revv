"""Private CPU-only Kaggle check for pinned dataset and encoder access."""

import hashlib
import importlib.metadata
import json
import platform
import urllib.request
from pathlib import Path


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
DATA_BLOB_SHA1 = "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
OUT = Path("/kaggle/working/revv-internet-preflight.json")


def read_bounded(url, max_bytes):
    with urllib.request.urlopen(url, timeout=35) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("External file exceeded declared cap")
    return data


def main():
    config = read_bounded(f"https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/config.json", 100_000)
    tokenizer = read_bounded(f"https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/tokenizer.json", 2_000_000)
    raw = read_bounded(f"https://raw.githubusercontent.com/clinc/oos-eval/{DATA_COMMIT}/data/data_full.json", 4_000_000)
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != DATA_BLOB_SHA1:
        raise ValueError("Downloaded CLINC source does not match pinned Git blob")
    parsed = json.loads(config)
    if parsed.get("model_type") != "bert":
        raise ValueError("Unexpected pinned encoder config")
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    report = {
        "kind": "kaggle_cpu_external_access_preflight_not_model_result",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": hashlib.sha256(config).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
        "data_commit": DATA_COMMIT,
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "data_blob_sha1": blob,
        "model_hidden_size": parsed.get("hidden_size"),
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "ready_for_gpu_pilot": transformers_version is not None,
        "limits": "Checks external access and package presence only; no model weights downloaded, GPU allocated, examples published or decision predictions made."
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned Kaggle CPU preflight ready={report['ready_for_gpu_pilot']}; report {OUT.name}")


if __name__ == "__main__":
    main()
