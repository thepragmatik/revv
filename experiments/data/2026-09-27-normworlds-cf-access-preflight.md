# NormWorlds-CF access and rights preflight

**As of:** 27 September 2026  
**Outcome:** Relevant prior art confirmed; dataset and code reuse are not cleared.

## Plain-English result

NormWorlds-CF is a close research neighbor: it uses solver-generated pairs of rule worlds and records which answers and supporting structures change. The paper describes 270 root families and 1,080 canonical-to-variant pairs. This means “learn from verified paired-world answer changes” is already established as a broad idea ([paper](https://arxiv.org/html/2607.03957v2)).

I checked the accessible arXiv v2 page and searched for an official code/data release through targeted GitHub and Hugging Face queries. I did not find a project repository, dataset download, or separate data/code license. The page displays an arXiv.org perpetual non-exclusive license for the paper, but does not document reuse terms for a separate dataset or implementation. The checked paper page also has no “Code availability” or “data availability” text. These are bounded search findings, not proof that a private or unindexed release does not exist.

No external dataset or code was downloaded, copied, or redistributed. For now, cite the work as prior art only.

## What the source establishes

- The benchmark is built from executable normative rule worlds with deterministic solver outputs, certificates, and paired-world change fields.
- Its compact change task covers answer, support, status, and attack changes; the paper reports 270 root families and 1,080 canonical-to-variant pairs ([paper abstract](https://arxiv.org/abs/2607.03957)).
- It is a synthetic, controlled reasoning environment. It does not establish natural-language transfer, local CPU latency, an 8 GiB process footprint, or the value of selectively refreshing cached typed outputs.
- Its close overlap narrows our candidate further: the potentially useful question is expected **gold-loss reduction** from refreshing a typed output after an input edit, measured against confidence, edit similarity, unconditional local repair, and full recomputation. This remains unvalidated and is not a novelty claim.

## Access checks

- Paper: [arXiv HTML v2](https://arxiv.org/html/2607.03957v2) and [abstract/version record](https://arxiv.org/abs/2607.03957).
- The checked paper text has no code/data availability section or official project link. The displayed arXiv license is not a distinct dataset/code license.
- Targeted searches: [GitHub repositories](https://github.com/search?q=%22NormWorlds-CF%22&type=repositories) and [Hugging Face datasets](https://huggingface.co/datasets?search=NormWorlds-CF); neither surfaced an author-maintained release during this check.
- This was a search-only preflight. No author contact, repository clone, dataset download, or redistribution occurred.

## Reuse decision and next step

**No-go for importing or redistributing the dataset/code** until an official artifact and separate terms are identified, or the rights holder confirms permission. If later cleared, use it as a secondary transfer or robustness test, not as the sole evidence for performance on natural-language decisions: its generated rule-world distribution is itself a major limitation.

Continue CPU-only work on a rights-clear independent workload and keep a transparent in-repo generator as diagnostic data. Any future compute run must first pass a free-provider availability preflight. Keep paid compute out of scope.
