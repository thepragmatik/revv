"""CPU-only source and package check for a frozen NLI cross-encoder pilot."""

import hashlib
import json
import platform
import urllib.request
from pathlib import Path

import sklearn
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "c4d86af4493123990d7762712de9ed730c876161"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
OUT = Path("/kaggle/working/revv-nli-access-preflight.json")


def read_small(filename, limit):
    with urllib.request.urlopen(f"{BASE}/{filename}", timeout=45) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Unexpectedly large {filename}")
    return raw


def main():
    config_raw = read_small("config.json", 10_000)
    config = json.loads(config_raw)
    labels = {int(k): v.lower() for k, v in config["id2label"].items()}
    if labels != {0: "contradiction", 1: "entailment", 2: "neutral"}:
        raise ValueError("NLI label mapping changed")
    tokenizer_raw = read_small("tokenizer.json", 4_000_000)
    cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True, trust_remote_code=False)
    if cfg.num_labels != 3 or not tokenizer.is_fast:
        raise ValueError("Expected three NLI logits and fast tokenizer")
    # Check weight availability only; do not download hundreds of MB on CPU preflight.
    request = urllib.request.Request(f"{BASE}/model.safetensors", method="HEAD")
    with urllib.request.urlopen(request, timeout=45) as response:
        weight_bytes = int(response.headers.get("Content-Length", "0"))
    if not 250_000_000 < weight_bytes < 450_000_000:
        raise ValueError(f"Unexpected NLI checkpoint weight size {weight_bytes}")
    report = {
        "kind": "kaggle_cpu_nli_access_preflight_no_gpu_no_model_weights",
        "model": MODEL,
        "revision": REVISION,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_raw).hexdigest(),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "id2label": labels,
        "weights_head_content_length_bytes": weight_bytes,
        "auto_sequence_classification_imported": AutoModelForSequenceClassification is not None,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "ready_for_frozen_gpu_pilot": True,
        "limits": "No weights downloaded, GPU allocated, model inference, training, ContractNLI content or local CPU serving measurements."
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"NLI source access and label map verified, weight size {weight_bytes} bytes; no GPU allocated", flush=True)


if __name__ == "__main__":
    main()
