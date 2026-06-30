"""Latency benchmarking harness.

Warm up, then time many runs and report p50/p95/p99 latency + FPS — the discipline
behind the inference benchmark table in the README. Backend-agnostic: pass any
callable (PyTorch CPU/GPU module, an ONNX Runtime session wrapper, a TensorRT runner).
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict

import numpy as np


def _cuda_sync_if_needed():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def benchmark_latency(
    fn: Callable[[], object], warmup: int = 50, runs: int = 500, label: str = "model"
) -> Dict[str, float]:
    """Time ``fn`` (a zero-arg thunk) and report latency percentiles.

    Returns ms latencies (p50/p95/p99/mean/min/max) and FPS (from the mean).
    """
    for _ in range(warmup):
        fn()
    _cuda_sync_if_needed()

    times = np.empty(runs)
    for i in range(runs):
        t0 = perf_counter()
        fn()
        _cuda_sync_if_needed()
        times[i] = (perf_counter() - t0) * 1000.0

    mean = float(times.mean())
    return {
        "label": label,
        "runs": runs,
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": mean,
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }


def format_benchmark_table(results: Dict[str, Dict[str, float]]) -> str:
    """Render ``{backend_name: benchmark_dict}`` as a markdown table (for the README)."""
    header = "| Backend | p50 (ms) | p95 (ms) | p99 (ms) | FPS |\n|---|---|---|---|---|"
    rows = [
        f"| {name} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['fps']:.1f} |"
        for name, r in results.items()
    ]
    return "\n".join([header, *rows])
