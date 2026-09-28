"""Audit pinned public CLINC train/validation data; never print utterances or inspect test rows."""

import hashlib
import json
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BASE = f"https://raw.githubusercontent.com/clinc/oos-eval/{COMMIT}/data"
BLOBS = {
    "data_full.json": "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa",
    "domains.json": "60a74358e52060e60b6128ae4efe679e9d6ba69c",
}
OUT = Path("results/clinc-audit.json")


def fetch_pinned(name):
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=45) as response:
        content = response.read(4_000_001)
    if len(content) > 4_000_000:
        raise ValueError(f"{name} exceeded its 4 MB audit cap")
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob != BLOBS[name]:
        raise ValueError(f"{name} blob hash mismatch; refuse unpinned source")
    return json.loads(content), hashlib.sha256(content).hexdigest()


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def examine(records):
    labels = Counter()
    by_text = defaultdict(set)
    for row in records:
        if not isinstance(row, list) or len(row) != 2 or not all(isinstance(x, str) for x in row):
            raise ValueError("Unexpected example schema")
        utterance, label = row
        key = normal(utterance)
        if not key:
            raise ValueError("Empty normalized utterance")
        labels[label] += 1
        by_text[key].add(label)
    return labels, by_text


def main():
    dataset, data_sha256 = fetch_pinned("data_full.json")
    domains, domain_sha256 = fetch_pinned("domains.json")
    # OOS lives in separate source fields; deliberately leave test and oos_test untouched.
    train_rows = dataset["train"] + dataset["oos_train"]
    val_rows = dataset["val"] + dataset["oos_val"]
    train_labels, train_texts = examine(train_rows)
    val_labels, val_texts = examine(val_rows)
    if any(row[1] != "oos" for row in dataset["oos_train"] + dataset["oos_val"]):
        raise ValueError("Unexpected label in an out-of-scope source field")
    mapped = [name for labels in domains.values() for name in labels]
    if len(mapped) != 150 or len(set(mapped)) != 150:
        raise ValueError("Domain mapping is not one-to-one across 150 intents")
    train_intents = set(train_labels) - {"oos"}
    val_intents = set(val_labels) - {"oos"}
    if train_intents != set(mapped) or val_intents != set(mapped):
        raise ValueError("Intent labels disagree with the pinned domain mapping")
    overlap = train_texts.keys() & val_texts.keys()
    report = {
        "kind": "pinned_public_clinc_train_val_integrity_audit_not_model_result",
        "source_commit": COMMIT,
        "source_url": BASE,
        "data_blob_sha1": BLOBS["data_full.json"],
        "data_sha256": data_sha256,
        "domains_blob_sha1": BLOBS["domains.json"],
        "domains_sha256": domain_sha256,
        "license": "CC BY 3.0 (upstream LICENSE); attribute original authors; do not commit raw examples",
        "examined_partitions": ["train", "val", "oos_train", "oos_val"],
        "locked_test_examples_examined": 0,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_oos": train_labels["oos"],
        "val_oos": val_labels["oos"],
        "in_scope_intents": len(train_intents),
        "domain_intent_counts": {name: len(labels) for name, labels in domains.items()},
        "train_in_scope_min_max": [min(train_labels[k] for k in train_intents), max(train_labels[k] for k in train_intents)],
        "val_in_scope_min_max": [min(val_labels[k] for k in val_intents), max(val_labels[k] for k in val_intents)],
        "train_normalized_duplicate_rows": len(train_rows) - len(train_texts),
        "val_normalized_duplicate_rows": len(val_rows) - len(val_texts),
        "train_conflicting_label_texts": sum(len(v) > 1 for v in train_texts.values()),
        "val_conflicting_label_texts": sum(len(v) > 1 for v in val_texts.values()),
        "train_val_exact_normalized_overlap_texts": len(overlap),
        "train_val_conflicting_label_texts": sum(train_texts[k] != val_texts[k] for k in overlap),
        "scope_limit": "Exact normalized duplicates only; does not detect paraphrase/template families. Test examples and benchmark quality are not audited."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned CLINC train/val + OOS audit completed: {len(train_rows)} train, {len(val_rows)} val; {len(overlap)} exact normalized overlaps. JSON artifact only; no texts printed.")


if __name__ == "__main__":
    main()
