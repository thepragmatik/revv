# Preregistered ContractNLI author-source audit

**[Plain explanation](../../docs/start-here.md) · [CER-1 hypotheses](../../docs/ideas.md) · [Source ledger](../../docs/sources.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete, 27 September 2026.** The [authors' dataset](https://stanfordnlp.github.io/contract-nli/) gives 17 fixed hypotheses with entailment, contradiction, not-mentioned and evidence spans over 607 long contracts. It directly probes the multi-question state-reuse and missing-evidence claim that single-utterance CLINC cannot test. The author states CC BY 4.0 plus terms on that page. Cite Koreeda and Manning (2021) when using it. This is research about a benchmark, not legal advice.

Before modelling, download only the [pinned author release](https://github.com/stanfordnlp/contract-nli/tree/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9), Git blob `757fd1dafd29a997fba00c60c6d40b2930a36159` for `resources/contract-nli.zip`, with a 70 MB cap. The zip is about 65 MB and includes source files; open **only** `train.json` and `dev.json`, capped at 25 MB uncompressed each. Never open `test.json` or original contract PDFs. Record source archive and two opened JSON SHA-256 hashes, member count, document/hypothesis and label counts, lengths as word-count proxies, missing annotations/evidence consistency and duplicate document IDs, normalized text and URLs between train/dev. Do not publish contracts, text excerpts, URLs, hashed original documents, or entire examples. Fail if the archive/hash/schema/17 hypotheses differ. One free Actions CPU job capped at 15 minutes.

**After audit:** choose a document-level development split that cannot leak the same contract, define the typed three-way prediction and evidence metrics, and register a short-document/long-document coverage rule before any GPU experiment. A 512-token truncation is inadequate if most documents are longer. The official Span NLI BERT baseline uses context segmentation; model comparisons must account for it. Test labels stay sealed until a complete candidate and margin are registered.

## Observed audit and implication

The [CPU Actions run](https://github.com/thepragmatik/revv/actions/runs/36293891523) verified the author Git blob and saved [aggregate hashes and statistics](2026-09-27-contractnli-audit.json). It opened only `contract-nli/train.json` and `contract-nli/dev.json`; no test JSON or contract PDFs were opened and no document text was published. The archive SHA-256 is `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b`.

| Split | Contracts | Hypothesis decisions | Entail / contradict / not mentioned | Median document words | Over 512 words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 423 | 7,191 | 3,530 / 841 / 2,820 | 1,504 | 401/423 (94.8%) |
| Development | 61 | 1,037 | 519 / 95 / 423 | 1,727 | 59/61 (96.7%) |

All 17 hypotheses have the same descriptions in train/dev; all decisions have annotations; positive labels have evidence and `NotMentioned` has none. No exact document ID, normalized document text or source URL overlaps train/dev. These checks do **not** rule out shared templates or related contracts. Word count is not model token count: the high long-document fraction makes a 512-token truncation likely to miss evidence, so the first real workload needs full-document span or chunk coverage and metrics that distinguish evidence retrieval from answer selection. The test split remains sealed.
