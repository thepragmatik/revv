"""Non-deployable gold-span versus retrieved-span frozen NLI stance diagnostic."""

import hashlib
import io
import json
import math
import statistics
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "c4d86af4493123990d7762712de9ed730c876161"
SOURCE = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip"
SOURCE_SHA = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
OUT = Path("/kaggle/working/revv-nli-gold-stances.json")
BATCH = 32
TOKEN_LIMIT = 256


def author_data():
    with urllib.request.urlopen(SOURCE, timeout=90) as response:
        raw = response.read(70_000_001)
    if len(raw) > 70_000_000 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Pinned author archive mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        data = {}
        for part in ("train", "dev"):
            member = source.getinfo(f"contract-nli/{part}.json")
            if member.file_size > 25_000_000:
                raise ValueError("Unexpectedly large author JSON")
            data[part] = json.loads(source.read(member))
        return data


def span_text(doc):
    return [doc["text"][start:end] for start, end in doc["spans"]]


def wilson(hits, total):
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / total
    denom = 1 + z*z/total
    mid = (p + z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [mid-half, mid+half]


def main():
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen NLI diagnostic requires Kaggle GPU")
    torch.manual_seed(907)
    torch.set_num_threads(2)
    data = author_data()
    train, dev = data["train"]["documents"], data["dev"]["documents"]
    hypotheses = data["train"]["labels"]
    keys = sorted(hypotheses)
    if len(train) != 423 or len(dev) != 61 or len(keys) != 17 or keys != sorted(data["dev"]["labels"]):
        raise ValueError("Author split size changed")
    train_spans = [span_text(doc) for doc in train]
    corpus = [item for group in train_spans for item in group]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True)
    vec.fit(corpus + [hypotheses[k]["hypothesis"] for k in keys])
    train_vec = vec.transform(corpus)
    evidence = defaultdict(list)
    offset = 0
    for doc, group in zip(train, train_spans):
        ann = doc["annotation_sets"][0]["annotations"]
        for k in keys:
            if ann[k]["choice"] != "NotMentioned":
                evidence[k].extend(offset + int(index) for index in ann[k]["spans"])
        offset += len(group)
    proto = np.stack([normalize(np.asarray(train_vec[evidence[k]].mean(axis=0)).reshape(1,-1))[0] for k in keys])
    cases = []
    for doc_index, doc in enumerate(dev):
        texts = span_text(doc)
        scores = np.asarray((vec.transform(texts) @ proto.T).T)
        ann = doc["annotation_sets"][0]["annotations"]
        for index, k in enumerate(keys):
            choice = ann[k]["choice"]
            if choice == "NotMentioned":
                continue
            gold = sorted(set(int(i) for i in ann[k]["spans"]))
            if not gold:
                raise ValueError("Positive case missing marked evidence")
            rank = np.argsort(-scores[index], kind="stable")[:5]
            cases.append({"document_index": doc_index, "gold_choice": choice,
                          "question": hypotheses[k]["hypothesis"],
                          "retrieved": [texts[int(i)] for i in rank],
                          "gold": [texts[i] for i in gold],
                          "retrieval_any_gold": bool(set(rank) & set(gold))})
    if len(cases) != 614 or sum(case["retrieval_any_gold"] for case in cases) != 584:
        raise ValueError("Selected-dev evidence differs from registered pooled retrieval")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION,
        use_safetensors=True, trust_remote_code=False).cuda().eval()
    mapping = {int(k): str(v).lower() for k,v in model.config.id2label.items()}
    if mapping != {0:"contradiction",1:"entailment",2:"neutral"}:
        raise ValueError("Checkpoint mapping changed")
    grouped = {}
    infer_start = time.monotonic()
    with torch.inference_mode():
        for arm in ("retrieved", "gold"):
            premises = [s for case in cases for s in case[arm]]
            questions = [case["question"] for case in cases for _ in case[arm]]
            result, truncated = [], 0
            for offset in range(0,len(premises),BATCH):
                a = premises[offset:offset+BATCH]
                b = questions[offset:offset+BATCH]
                truncated += sum(len(ids)>TOKEN_LIMIT for ids in tokenizer(a,b,truncation=False)["input_ids"])
                enc = tokenizer(a,b,padding=True,truncation=True,max_length=TOKEN_LIMIT,return_tensors="pt")
                prob = F.softmax(model(**{k:v.cuda() for k,v in enc.items()}).logits.float(),dim=-1)
                result.append(prob.cpu().numpy())
                if time.monotonic()-started > 240:
                    raise TimeoutError("Oracle stance diagnostic exceeded internal 240-second budget")
            torch.cuda.synchronize()
            probs = np.concatenate(result)
            cursor = 0
            predictions = []
            for case in cases:
                part = probs[cursor:cursor+len(case[arm])]
                cursor += len(case[arm])
                predictions.append("Entailment" if float(part[:,1].max()) >= float(part[:,0].max()) else "Contradiction")
            grouped[arm] = {"predictions": predictions, "nli_pairs": len(premises), "truncated_pairs": truncated}
    infer_seconds = time.monotonic()-infer_start
    def measure(arm):
        guesses=grouped[arm]["predictions"]
        true=[c["gold_choice"] for c in cases]
        hits=sum(a==b for a,b in zip(true,guesses))
        by_label={}
        for label in ("Entailment","Contradiction"):
            n=sum(x==label for x in true)
            h=sum(t==label and p==label for t,p in zip(true,guesses))
            by_label[label]={"n":n,"recall":h/n,"wilson95":wilson(h,n)}
        return {"binary_positive_only_accuracy": hits/len(cases),"binary_positive_only_wilson95":wilson(hits,len(cases)),
                "balanced_accuracy": sum(by_label[label]["recall"] for label in by_label)/2,
                "by_true_label":by_label, "nli_pairs":grouped[arm]["nli_pairs"],
                "truncated_pairs":grouped[arm]["truncated_pairs"]}
    change = np.zeros(len(dev), dtype=np.float64)
    for i,case in enumerate(cases):
        y=case["gold_choice"]
        change[case["document_index"]]+=int(grouped["gold"]["predictions"][i]==y)-int(grouped["retrieved"]["predictions"][i]==y)
    random=np.random.default_rng(907)
    # Vary both the numerator and the positive-case denominator by sampled document.
    counts=np.bincount([case["document_index"] for case in cases],minlength=len(dev))
    bootstrap=[]
    for _ in range(1000):
        ids=random.integers(0,len(dev),len(dev))
        bootstrap.append(float(change[ids].sum()/counts[ids].sum()))
    observed={arm:measure(arm) for arm in ("retrieved","gold")}
    report={
        "kind":"frozen_nli_contract_gold_span_stance_oracle_diagnostic_not_deployable",
        "source_sha256":SOURCE_SHA,"model":MODEL,"revision":REVISION,
        "development_documents":len(dev),"positive_decisions":len(cases),
        "positive_class_counts":dict(Counter(case["gold_choice"] for case in cases)),
        "entailment_only_accuracy_floor":sum(case["gold_choice"]=="Entailment" for case in cases)/len(cases),
        "evidence_any_gold_at_5":sum(case["retrieval_any_gold"] for case in cases)/len(cases),
        "gold_evidence_span_count_min_mean_max":[min(len(c["gold"]) for c in cases),
            statistics.mean(len(c["gold"]) for c in cases),max(len(c["gold"]) for c in cases)],
        "methods":observed,
        "paired_gold_minus_retrieved_binary_accuracy":observed["gold"]["binary_positive_only_accuracy"]-observed["retrieved"]["binary_positive_only_accuracy"],
        "paired_document_cluster_bootstrap95":[float(x) for x in np.quantile(bootstrap,[0.025,0.975])],
        "gpu":torch.cuda.get_device_name(0),"nli_inference_seconds_for_both_arms":infer_seconds,
        "peak_torch_gpu_allocated_bytes":torch.cuda.max_memory_allocated(0),
        "test_documents_examined":0,
        "limitations":"Gold spans from positive development labels are an oracle diagnostic, impossible as deployable inputs. Binary stance conditional on Entailment/Contradiction ignores NotMentioned and uses no trained aggregation or calibration. No local CPU latency/RSS or locked test."
    }
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(f"Oracle stance: retrieved balanced {observed['retrieved']['balanced_accuracy']:.3f}, gold {observed['gold']['balanced_accuracy']:.3f}",flush=True)


if __name__=="__main__":
    main()
