"""Audit only the author-released ContractNLI train/dev metadata and labels."""

import hashlib
import json
import statistics
import tempfile
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


SOURCE_COMMIT = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
SOURCE_BLOB = "757fd1dafd29a997fba00c60c6d40b2930a36159"
SOURCE_URL = f"https://raw.githubusercontent.com/stanfordnlp/contract-nli/{SOURCE_COMMIT}/resources/contract-nli.zip"
MAX_ARCHIVE_BYTES = 70_000_000
OUT = Path("results/contractnli-audit.json")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quantile(values, q):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * q)]


def summary(dataset):
    docs = dataset["documents"]
    hypotheses = dataset["labels"]
    if len(hypotheses) != 17:
        raise ValueError("Expected 17 fixed hypotheses")
    choices = Counter()
    evidence_nonempty = Counter()
    missing_annotations = Counter()
    word_lengths = []
    ids, hashed_texts, urls = set(), set(), set()
    for doc in docs:
        text = doc["text"]
        if doc["id"] in ids:
            raise ValueError("Duplicate document ID within split")
        ids.add(doc["id"])
        hashed_texts.add(hashlib.sha256(normalize(text).encode()).hexdigest())
        if doc.get("url"):
            urls.add(doc["url"])
        word_lengths.append(len(text.split()))
        annotations = doc["annotation_sets"][0]["annotations"]
        for key in hypotheses:
            if key not in annotations:
                missing_annotations[key] += 1
                continue
            entry = annotations[key]
            choice = entry["choice"]
            if choice not in {"Entailment", "Contradiction", "NotMentioned"}:
                raise ValueError("Unexpected NLI label")
            choices[choice] += 1
            selected = entry["spans"]
            if any(index < 0 or index >= len(doc["spans"]) for index in selected):
                raise ValueError("Evidence index outside document span list")
            if choice == "NotMentioned" and selected:
                raise ValueError("NotMentioned contains evidence")
            evidence_nonempty[choice] += bool(selected)
    if not word_lengths:
        raise ValueError("Empty dataset split")
    return {
        "documents": len(docs),
        "hypotheses": len(hypotheses),
        "document_ids": ids,
        "hashed_normalized_texts": hashed_texts,
        "urls": urls,
        "choices": dict(choices),
        "missing_annotations": dict(missing_annotations),
        "with_nonempty_evidence": dict(evidence_nonempty),
        "document_words_min_median_p95_max": [min(word_lengths), statistics.median(word_lengths), quantile(word_lengths, 0.95), max(word_lengths)],
        "documents_over_512_words": sum(n > 512 for n in word_lengths),
    }


def main():
    began = time.monotonic()
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="revv-contract-audit-") as directory:
        archive = Path(directory) / "author-release.zip"
        source_hash = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, archive.open("wb") as dest:
            while block := response.read(4_000_000):
                size += len(block)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Author archive exceeds preregistered 70 MB cap")
                source_hash.update(block)
                dest.write(block)
        blob_hash = hashlib.sha1()
        blob_hash.update(f"blob {size}\0".encode())
        with archive.open("rb") as stream:
            while block := stream.read(4_000_000):
                blob_hash.update(block)
        if blob_hash.hexdigest() != SOURCE_BLOB:
            raise ValueError("Author release does not match pinned Git blob")
        with zipfile.ZipFile(archive) as source:
            filenames = [x.filename for x in source.infolist() if not x.is_dir()]
            chosen = {}
            for split in ("train", "dev"):
                matches = [x for x in source.infolist() if Path(x.filename).name == f"{split}.json"]
                if len(matches) != 1 or matches[0].file_size > 25_000_000:
                    raise ValueError(f"Cannot uniquely identify bounded {split}.json")
                chosen[split] = matches[0]
            raw = {}
            stats = {}
            label_hashes = {}
            for split, member in chosen.items():
                # Intentionally never open, parse, or hash test.json or original PDF files.
                raw[split] = source.read(member)
                dataset = json.loads(raw[split])
                label_hashes[split] = hashlib.sha256(json.dumps(dataset["labels"], sort_keys=True).encode()).hexdigest()
                stats[split] = summary(dataset)
            if label_hashes["train"] != label_hashes["dev"]:
                raise ValueError("Hypothesis descriptions differ across train/dev")
        overlap = {
            "document_ids": len(stats["train"]["document_ids"] & stats["dev"]["document_ids"]),
            "normalized_document_texts": len(stats["train"]["hashed_normalized_texts"] & stats["dev"]["hashed_normalized_texts"]),
            "source_urls": len(stats["train"]["urls"] & stats["dev"]["urls"]),
        }
        for item in stats.values():
            for private_key in ("document_ids", "hashed_normalized_texts", "urls"):
                del item[private_key]
        report = {
            "kind": "contractnli_author_train_dev_integrity_audit_no_test_read",
            "source_url": SOURCE_URL,
            "source_commit": SOURCE_COMMIT,
            "source_git_blob": SOURCE_BLOB,
            "archive_sha256": source_hash.hexdigest(),
            "archive_bytes": size,
            "archive_member_count": len(filenames),
            "train_dev_members": {k: v.filename for k, v in chosen.items()},
            "train_dev_json_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
            "fixed_hypothesis_description_sha256": label_hashes["train"],
            "splits": stats,
            "train_dev_overlap": overlap,
            "locked_test_documents_examined": 0,
            "elapsed_seconds": time.monotonic() - began,
            "limits": "Word-count length proxy, no tokenizer inference, no content excerpts, test JSON and original PDFs unopened. Audit only; no decision benchmark or local memory/latency claim.",
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Author ContractNLI audit complete: train {stats['train']['documents']}, dev {stats['dev']['documents']}, source SHA-256 {source_hash.hexdigest()}")


if __name__ == "__main__":
    main()
