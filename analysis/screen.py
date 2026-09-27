"""Dependency-free upper-level memory and linear-layer compute screen.

These are analytical floors/proxies, never measured RSS, latency or quality.
Run: python analysis/screen.py
"""

GIB = 1024 ** 3


def weight_gib(params: int, bits: int) -> float:
    return params * bits / 8 / GIB


def shared_linear_flops(params: int, state: int, candidate: int, count: int) -> int:
    return 2 * params * (state + count * candidate)


def repeated_linear_flops(params: int, state: int, candidate: int, count: int) -> int:
    return 2 * params * count * (state + candidate)


def main() -> None:
    # Nominal counts; actual architecture may include embedding lookups and tied weights.
    candidates = [("small encoder", 100_000_000), ("ModernBERT base", 149_000_000),
                  ("Laya-sized", 421_000_000), ("0.8B", 800_000_000),
                  ("4B", 4_000_000_000), ("9B", 9_000_000_000)]
    print("Nominal WEIGHT-ONLY GiB; quantized storage omits scales, zero points and runtime")
    print(f"{'model':<20} {'16-bit':>8} {'8-bit':>8} {'4-bit':>8} {'8GiB headroom at 16b':>22}")
    for name, count in candidates:
        w = [weight_gib(count, bits) for bits in (16, 8, 4)]
        print(f"{name:<20} {w[0]:8.2f} {w[1]:8.2f} {w[2]:8.2f} {8-w[0]:22.2f}")

    p, state, option = 100_000_000, 128, 32
    print("\nLinear-layer FLOP proxy: same 100M body, L=128, candidate length=32")
    print(f"{'questions':>9} {'options':>7} {'shared GF':>10} {'repeated GF':>12} {'ratio':>8}")
    for q, k in [(1, 2), (1, 5), (8, 5)]:
        n = q * k
        a = shared_linear_flops(p, state, option, n)
        b = repeated_linear_flops(p, state, option, n)
        print(f"{q:9d} {k:7d} {a/1e9:10.1f} {b/1e9:12.1f} {b/a:8.2f}")


if __name__ == "__main__":
    main()
