# Kaggle GPU connectivity, first attempts and success

**Date:** 27 September 2026. **Status:** CUDA connectivity confirmed on a Tesla T4 after account verification and a probe fix. No decision model was trained or benchmarked.

## What we tried

The [connection check](https://github.com/thepragmatik/revv/actions/runs/36289835397) authenticated with the repository's `KAGGLE_API_TOKEN`. We submitted the [small private GPU probe](../../experiments/kaggle/smoke.py) through [this GitHub Actions run](https://github.com/thepragmatik/revv/actions/runs/36289867446), requesting a T4 accelerator, disabling Kaggle notebook internet access, and capping execution at 300 seconds. It contains no dataset and trains no model.

The Kaggle session progressed `QUEUED → RUNNING → ERROR`. Its [execution log retrieved in a separate read-only diagnostic run](https://github.com/thepragmatik/revv/actions/runs/36289944393) shows `torch.cuda.is_available()` was false, so the script raised `No CUDA GPU is available in this Kaggle session` before performing matrix multiplication.

A [quota check](https://github.com/thepragmatik/revv/actions/runs/36290063376) reported 30.00 GPU hours remaining and 0.00 used, with a 3 October 2026 refresh date. A private kernel metadata readback returned HTTP 403, so we have no independent confirmation of the saved accelerator setting. The available quota does not by itself prove GPU allocation.

**Account follow-up:** the owner confirmed the Kaggle account is not phone verified and sees no Accelerator option under notebook Settings. Kaggle staff [state that phone verification is required](https://www.kaggle.com/discussions/product-feedback/451678) for accelerator use and recommend reloading the editor after completing it. This explains the missing UI control; the failed API allocation has not been isolated independently from account eligibility.

## Decision

## After verification

The [second run](https://github.com/thepragmatik/revv/actions/runs/36290661389) reached CUDA but failed when the probe reset its peak-memory counter before allocating tensors. This is a probe implementation error. We removed that optional reset, leaving the calculation and post-run memory read intact.

The [third run](https://github.com/thepragmatik/revv/actions/runs/36290758927) succeeded and uploaded [this raw JSON report](2026-09-27-gpu-smoke.json). It identifies one **Tesla T4** with 15,636,037,632 bytes of device memory, PyTorch `2.10.0+cu128`, eight 512 × 512 matrix multiplications, a finite checksum, and 12,713,984 bytes of peak PyTorch tensor allocation. The reported 0.000841148 seconds is a tiny warmed-up device computation; it is **not an end-to-end latency benchmark**, CPU comparison, model inference result, or process memory measurement. The JSON is pinned to Kaggle kernel version `rathworx/revv-gpu-smoke/3` and the Actions run used commit `c1f906bc152abaaa85483e5f4eb1e2c5b1c39fda`.

## Decision

The free GPU path is ready for a *bounded, preregistered pilot* after data and baseline gates. Keep the 8 GiB local CPU inference target separate from the Kaggle T4's device memory. Use the [runner](../../analysis/kaggle_smoke.py) and GitHub Actions artifact path as the template for collecting future results; a successful smoke does not validate the proposed architecture or a performance claim.
