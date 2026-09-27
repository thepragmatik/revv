"""Bounded synthetic forward-pass screen for state reuse, not a model benchmark."""

import json
import platform
import statistics
import time
from pathlib import Path

import torch


SEED = 271
WIDTH = 256
LAYERS = 2
HEADS = 4
FF_WIDTH = 1024
OPTION_TOKENS = 32
SLICES = [(128, 2), (128, 40), (512, 2), (512, 40), (2048, 8)]
WARMUP_PAIRS = 2
TIMED_PAIRS = 5
STOP_AFTER_SECONDS = 210
OUTPUT = Path("/kaggle/working/revv-forward-pilot.json")


def elapsed_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, float(value)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not reinterpret this as a CPU result")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    layer = torch.nn.TransformerEncoderLayer(
        d_model=WIDTH,
        nhead=HEADS,
        dim_feedforward=FF_WIDTH,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
    )
    encoder = torch.nn.TransformerEncoder(layer, num_layers=LAYERS, enable_nested_tensor=False)
    encoder = encoder.to(device).eval()
    started = time.perf_counter()
    rows = []

    with torch.inference_mode():
        for length, count in SLICES:
            if time.perf_counter() - started > STOP_AFTER_SECONDS:
                break
            state = torch.randn(1, length, WIDTH, device=device)
            candidates = torch.randn(count, OPTION_TOKENS, WIDTH, device=device)

            def shared():
                state_vector = encoder(state).mean(dim=1)
                option_vectors = encoder(candidates).mean(dim=1)
                return (state_vector * option_vectors).sum().item()

            def repeated_cross():
                joint = torch.cat((state.expand(count, -1, -1), candidates), dim=1)
                return encoder(joint).mean(dim=1).sum().item()

            measurements = {"shared": [], "cross": []}
            for pair in range(WARMUP_PAIRS + TIMED_PAIRS):
                # Reverse order on alternating pairs to reduce thermal/order bias.
                order = ("shared", "cross") if pair % 2 == 0 else ("cross", "shared")
                for name in order:
                    elapsed, checksum = elapsed_ms(shared if name == "shared" else repeated_cross)
                    if pair >= WARMUP_PAIRS:
                        measurements[name].append(elapsed)
                    if not torch.isfinite(torch.tensor(checksum)):
                        raise ValueError("Non-finite output from encoder")
            median_shared = statistics.median(measurements["shared"])
            median_cross = statistics.median(measurements["cross"])
            row = {
                "state_tokens": length,
                "candidates": count,
                "option_tokens": OPTION_TOKENS,
                "shared_ms": measurements["shared"],
                "cross_ms": measurements["cross"],
                "median_shared_ms": median_shared,
                "median_cross_ms": median_cross,
                "cross_over_shared": median_cross / median_shared,
                "linear_proxy_cross_over_shared": count * (length + OPTION_TOKENS) / (length + count * OPTION_TOKENS),
            }
            rows.append(row)
            print(f"L={length} N={count}: shared {median_shared:.2f}ms, cross {median_cross:.2f}ms", flush=True)

    report = {
        "kind": "synthetic_t4_forward_crossover_not_quality_or_cpu_benchmark",
        "seed": SEED,
        "model": {"type": "same_untrained_transformer_encoder_both_paths", "layers": LAYERS, "width": WIDTH, "heads": HEADS, "ff_width": FF_WIDTH, "parameters": sum(p.numel() for p in encoder.parameters())},
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "warmup_pairs": WARMUP_PAIRS,
        "timed_pairs": TIMED_PAIRS,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "time_cap_seconds": STOP_AFTER_SECONDS,
        "completed_all_slices": len(rows) == len(SLICES),
        "rows": rows,
        "limitations": "Synthetic vectors; no tokenizer, training, labels, probabilities, CPU latency or process RSS. Outputs are different functions."
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.name}; {len(rows)}/{len(SLICES)} slices", flush=True)


if __name__ == "__main__":
    main()
