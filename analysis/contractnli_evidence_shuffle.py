"""Preregistered document evidence removal and deterministic shuffle controls."""

import hashlib
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from contractnli_controls import ARCHIVE_SHA256, LABELS, download_train_dev, metrics, text_spans


OUT = Path("results/contractnli-evidence-shuffle.json")
K = 5
N_FOLDS = 5


def document_fold(doc):
    return int(hashlib.sha256(str(doc["id"]).encode()).hexdigest(), 16) % N_FOLDS


def results(true, probabilities):
    guessed = [LABELS[int(i)] for i in probabilities.argmax(axis=1)]
    probs = [{label: float(row[j]) for j, label in enumerate(LABELS)} for row in probabilities]
    return metrics(true, guessed, probs), guessed


def main():
    began = time.monotonic()
    data = download_train_dev()
    train, dev = data["train"]["documents"], data["dev"]["documents"]
    hypotheses = data["train"]["labels"]
    ids = sorted(hypotheses)
    if len(train) != 423 or len(dev) != 61 or len(ids) != 17 or ids != sorted(data["dev"]["labels"]):
        raise ValueError("Dataset dimensions differ from author audit")
    train_spans = [text_spans(doc) for doc in train]
    corpus = [span for doc in train_spans for span in doc]
    retriever = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    retriever.fit(corpus + [hypotheses[h]["hypothesis"] for h in ids])
    span_matrix = retriever.transform(corpus)
    query = retriever.transform(hypotheses[h]["hypothesis"] for h in ids)
    if (span_matrix.shape[0], span_matrix.shape[1]) != (32895, 30000):
        raise ValueError("Vocabulary mismatch versus evidence controls")
    starts = np.cumsum([0] + [len(spans) for spans in train_spans])
    folds = [document_fold(doc) for doc in train]
    fold_sizes = dict(Counter(folds))
    if len(fold_sizes) != N_FOLDS or min(fold_sizes.values()) < 60:
        raise ValueError("Unexpected crossfit document folds")
    def prototypes(excluded_fold):
        selected = defaultdict(list)
        for doc_index, doc in enumerate(train):
            if excluded_fold is not None and folds[doc_index] == excluded_fold:
                continue
            annotations = doc["annotation_sets"][0]["annotations"]
            for h in ids:
                entry = annotations[h]
                if entry["choice"] != "NotMentioned":
                    selected[h].extend(int(starts[doc_index]) + i for i in entry["spans"])
        vectors = []
        for h in ids:
            if not selected[h]:
                raise ValueError("No remaining evidence for a hypothesis in a crossfit fold")
            avg = np.asarray(span_matrix[selected[h]].mean(axis=0)).reshape(1, -1)
            vectors.append(normalize(avg)[0])
        return np.stack(vectors)
    oof_prototypes = {fold: prototypes(fold) for fold in range(N_FOLDS)}
    full_prototypes = prototypes(None)
    def make_rows(documents, is_train):
        text = {"hypothesis": [], "pooled_evidence": []}
        truth, doc_indexes, hyp_indexes = [], [], []
        for doc_index, doc in enumerate(documents):
            spans = train_spans[doc_index] if is_train else text_spans(doc)
            vectors = retriever.transform(spans)
            qscore = (query @ vectors.T).toarray()
            prototype = oof_prototypes[folds[doc_index]] if is_train else full_prototypes
            pscore = np.asarray((vectors @ prototype.T).T)
            annotations = doc["annotation_sets"][0]["annotations"]
            for j, h in enumerate(ids):
                label = annotations[h]["choice"]
                if label not in LABELS:
                    raise ValueError("Unexpected ternary label")
                truth.append(label)
                doc_indexes.append(doc_index)
                hyp_indexes.append(j)
                for method, scores in (("hypothesis", qscore), ("pooled_evidence", pscore)):
                    top = np.argsort(-scores[j], kind="stable")[:K]
                    # Fixed description gives both heads the same question information.
                    text[method].append("Hypothesis: " + hypotheses[h]["hypothesis"] + " Evidence: " + " ".join(spans[int(i)] for i in top))
        return text, truth, doc_indexes, hyp_indexes
    xtrain, ytrain, _, htrain = make_rows(train, True)
    xdev, ydev, docdev, hdev = make_rows(dev, False)
    if len(ytrain) != 7191 or len(ydev) != 1037:
        raise ValueError("Wrong number of document-hypothesis rows")
    # Shared vocabulary across both methods isolates evidence selection.
    head_vocab = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    head_vocab.fit(xtrain["hypothesis"] + xtrain["pooled_evidence"])
    def features(texts, h_indices):
        word = head_vocab.transform(texts)
        onehot = scipy.sparse.csr_matrix((np.ones(len(h_indices)), (np.arange(len(h_indices)), h_indices)), shape=(len(h_indices), len(ids)))
        return hstack((word, onehot), format="csr")
    outputs, predictions, iterations = {}, {}, {}
    for method in ("hypothesis_only", "pooled_evidence"):
        classifier = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs")
        training_text = (["Hypothesis: " + hypotheses[ids[j]]["hypothesis"] for j in htrain]
                         if method == "hypothesis_only" else xtrain[method])
        dev_text = (["Hypothesis: " + hypotheses[ids[j]]["hypothesis"] for j in hdev]
                    if method == "hypothesis_only" else xdev[method])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features(training_text, htrain), ytrain)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("Logistic head did not converge under registered cap")
        raw = classifier.predict_proba(features(dev_text, hdev))
        ordered = raw[:, [list(classifier.classes_).index(label) for label in LABELS]]
        outputs[method], predictions[method] = results(ydev, ordered)
        iterations[method] = int(classifier.n_iter_[0])
        if method == "pooled_evidence":
            # Keep each fixed hypothesis and original classifier; swap its retrieved
            # text for that from the next development contract, cyclic by document.
            shuffled = []
            for doc_i in range(len(dev)):
                for question_i in range(len(ids)):
                    row = doc_i * len(ids) + question_i
                    other = ((doc_i + 1) % len(dev)) * len(ids) + question_i
                    left = xdev[method][row].split(" Evidence: ", 1)
                    right = xdev[method][other].split(" Evidence: ", 1)
                    if len(left) != 2 or len(right) != 2:
                        raise ValueError("Cannot find question/evidence delimiter")
                    shuffled.append(left[0] + " Evidence: " + right[1])
            shuffled_raw = classifier.predict_proba(features(shuffled, hdev))
            shuffled_probs = shuffled_raw[:, [list(classifier.classes_).index(label) for label in LABELS]]
            outputs["shuffled_development_document_evidence"], predictions["shuffled_development_document_evidence"] = results(ydev, shuffled_probs)
    if abs(outputs["pooled_evidence"]["accuracy"] - 0.7020250723240116) > 1e-12:
        raise ValueError("Pooled control failed to reproduce registered accuracy")
    if abs(outputs["pooled_evidence"]["macro_f1"] - 0.6587908259170189) > 1e-12:
        raise ValueError("Pooled control failed to reproduce registered macro F1")
    effects_by_doc = np.zeros((len(dev), 3), dtype=int)
    for i, truth in enumerate(ydev):
        effects_by_doc[docdev[i], 0] += int(predictions["pooled_evidence"][i] == truth) - int(predictions["hypothesis_only"][i] == truth)
        effects_by_doc[docdev[i], 1] += int(predictions["pooled_evidence"][i] == truth) - int(predictions["shuffled_development_document_evidence"][i] == truth)
        effects_by_doc[docdev[i], 2] += 1
    if not all(count == 17 for count in effects_by_doc[:, 2]):
        raise ValueError("Document bootstrap groups lost a hypothesis")
    generator = np.random.default_rng(629)
    boot = np.array([effects_by_doc[generator.integers(0, len(dev), len(dev)), :2].sum(axis=0) / 1037 for _ in range(1000)])
    report = {
        "kind": "contractnli_matched_hypothesis_only_and_cross_document_evidence_shuffle_cpu_negative_controls",
        "source_sha256": ARCHIVE_SHA256,
        "train_documents": len(train), "development_documents": len(dev),
        "hypotheses": len(ids), "train_rows": len(ytrain), "development_rows": len(ydev),
        "crossfit_document_fold_counts": fold_sizes,
        "evidence_k": K,
        "retriever_features_fit": span_matrix.shape[1],
        "classifier_features_fit": len(head_vocab.get_feature_names_out()),
        "classifier": "multinomial logistic regression, C=1, lbfgs max_iter=250; hypothesis-only trained without evidence, pooled evidence trained with crossfit retrieval; shared training-only TF-IDF vocabulary from registered original two arms, and question onehot",
        "classifier_iterations": iterations,
        "methods": outputs,
        "paired_pooled_minus_hypothesis_only_accuracy": outputs["pooled_evidence"]["accuracy"] - outputs["hypothesis_only"]["accuracy"],
        "paired_pooled_minus_hypothesis_only_document_bootstrap95": [float(x) for x in np.quantile(boot[:, 0], [0.025, 0.975])],
        "paired_pooled_minus_shuffled_evidence_accuracy": outputs["pooled_evidence"]["accuracy"] - outputs["shuffled_development_document_evidence"]["accuracy"],
        "paired_pooled_minus_shuffled_evidence_document_bootstrap95": [float(x) for x in np.quantile(boot[:, 1], [0.025, 0.975])],
        "shuffle": "For each development document d and fixed hypothesis h, replace the selected evidence text with the same hypothesis evidence from development document (d+1) modulo 61, leaving hypothesis text and onehot unchanged; no label lookup or refitting",
        "test_documents_examined": 0,
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "The shuffled evidence comes from another development document with unknown label and is an out-of-distribution input, not independent-domain generalization. Fixed 17 hypotheses and sparse head. Hypothesis-only uses same vocabulary fitted on full training examples, never dev. No causal identification of evidence use, separate probability calibration, local deployment latency/RSS, GPU or locked test."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ContractNLI evidence control: only {outputs['hypothesis_only']['accuracy']:.3f}, pooled {outputs['pooled_evidence']['accuracy']:.3f}, cross-document shuffled {outputs['shuffled_development_document_evidence']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
