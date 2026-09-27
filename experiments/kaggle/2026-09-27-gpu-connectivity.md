# Kaggle GPU connectivity, first attempt

**Date:** 27 September 2026. **Status:** GPU unavailable in the submitted session; no model experiment was run.

## What we tried

The [connection check](https://github.com/thepragmatik/revv/actions/runs/36289835397) authenticated with the repository's `KAGGLE_API_TOKEN`. We submitted the [small private GPU probe](../../experiments/kaggle/smoke.py) through [this GitHub Actions run](https://github.com/thepragmatik/revv/actions/runs/36289867446), requesting a T4 accelerator, disabling Kaggle notebook internet access, and capping execution at 300 seconds. It contains no dataset and trains no model.

The Kaggle session progressed `QUEUED → RUNNING → ERROR`. Its [execution log retrieved in a separate read-only diagnostic run](https://github.com/thepragmatik/revv/actions/runs/36289944393) shows `torch.cuda.is_available()` was false, so the script raised `No CUDA GPU is available in this Kaggle session` before performing matrix multiplication.

A [quota check](https://github.com/thepragmatik/revv/actions/runs/36290063376) reported 30.00 GPU hours remaining and 0.00 used, with a 3 October 2026 refresh date. A private kernel metadata readback returned HTTP 403, so we have no independent confirmation of the saved accelerator setting. The available quota does not by itself prove GPU allocation.

**Account follow-up:** the owner confirmed the Kaggle account is not phone verified and sees no Accelerator option under notebook Settings. Kaggle staff [state that phone verification is required](https://www.kaggle.com/discussions/product-feedback/451678) for accelerator use and recommend reloading the editor after completing it. This explains the missing UI control; the failed API allocation has not been isolated independently from account eligibility.

## Decision

Do not label this as a GPU benchmark or model performance result. Complete account phone verification in Kaggle, reload the notebook editor, and confirm GPU is selectable. Retry the same bounded probe only then; inspect the artifact for an actual device name. The [runner](../../analysis/kaggle_smoke.py) includes the Kaggle execution log in its error when a submitted session fails.
