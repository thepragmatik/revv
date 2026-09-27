"""Document-disjoint evidence-prototype retrieval plus matched ternary CPU heads."""

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


OUT = Path("results/contractnli-lexical-heads.json")
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
    for method in ("hypothesis", "pooled_evidence"):
        classifier = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features(xtrain[method], htrain), ytrain)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("Logistic head did not converge under registered cap")
        raw = classifier.predict_proba(features(xdev[method], hdev))
        ordered = raw[:, [list(classifier.classes_).index(label) for label in LABELS]]
        outputs[method], predictions[method] = results(ydev, ordered)
        iterations[method] = int(classifier.n_iter_[0])
    effects_by_doc = np.zeros((len(dev), 2), dtype=int)
    for i, truth in enumerate(ydev):
        effects_by_doc[docdev[i], 0] += int(predictions["pooled_evidence"][i] == truth) - int(predictions["hypothesis"][i] == truth)
        effects_by_doc[docdev[i], 1] += 1
    if not all(count == 17 for count in effects_by_doc[:, 1]):
        raise ValueError("Document bootstrap groups lost a hypothesis")
    generator = np.random.default_rng(629)
    boot = [effects_by_doc[generator.integers(0, len(dev), len(dev)), 0].sum() / 1037 for _ in range(1000)]
    report = {
        "kind": "contractnli_crossfit_lexical_evidence_three_way_cpu_controls_not_neural_model",
        "source_sha256": ARCHIVE_SHA256,
        "train_documents": len(train), "development_documents": len(dev),
        "hypotheses": len(ids), "train_rows": len(ytrain), "development_rows": len(ydev),
        "crossfit_document_fold_counts": fold_sizes,
        "evidence_k": K,
        "retriever_features_fit": span_matrix.shape[1],
        "classifier_features_fit": len(head_vocab.get_feature_names_out()),
        "classifier": "shared multinomial logistic regression, C=1, lbfgs max_iter=250; same question text, word unigram/bigram TF-IDF vocabulary and hypothesis onehot in both arms",
        "classifier_iterations": iterations,
        "methods": outputs,
        "paired_pooled_evidence_minus_hypothesis_accuracy": outputs["pooled_evidence"]["accuracy"] - outputs["hypothesis"]["accuracy"],
        "paired_document_cluster_bootstrap95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "test_documents_examined": 0,
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "Public development, author spans, fixed 17 hypotheses and sparse lexical classifier. Word vocabulary fit on training texts including crossfit held fold without labels; train evidence prototype excluded each held document fold. Classifier probabilities not separately calibrated; no GPU or deployment CPU per-request latency/RSS or locked test."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Matched ContractNLI heads: query {outputs['hypothesis']['accuracy']:.3f}, pooled evidence {outputs['pooled_evidence']['accuracy']:.3f}, prior floor 0.681")


if __name__ == "__main__":
    main()
