"""Tests for ONNX export/verify (CPU), the benchmark harness, and the TRT guard."""

import pytest

from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.optimization.benchmark import benchmark_latency, format_benchmark_table
from perceptnet.optimization.build_trt import build_engine, tensorrt_available


def small_cfg():
    return PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0))


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #
def test_benchmark_latency_keys_and_ordering():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    res = benchmark_latency(fn, warmup=3, runs=20, label="noop")
    assert counter["n"] == 23                       # warmup + runs called
    for k in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps"):
        assert k in res
    assert res["p99_ms"] >= res["p50_ms"] >= 0
    assert res["fps"] > 0


def test_format_benchmark_table():
    table = format_benchmark_table({"cpu": benchmark_latency(lambda: None, warmup=1, runs=5)})
    assert table.startswith("| Backend |") and "cpu" in table


# --------------------------------------------------------------------------- #
# ONNX export + ORT-CPU verification
# --------------------------------------------------------------------------- #
def test_export_and_verify_onnx(tmp_path):
    from perceptnet.optimization.export_onnx import export_backbone_head, verify_onnx

    cfg = small_cfg()
    model = PointPillars(cfg).eval()
    onnx_path = str(tmp_path / "backbone_head.onnx")
    export_backbone_head(model, onnx_path)
    assert verify_onnx(onnx_path, model, atol=1e-3)


# --------------------------------------------------------------------------- #
# TensorRT guard (no GPU on this machine)
# --------------------------------------------------------------------------- #
def test_build_trt_raises_without_tensorrt():
    if tensorrt_available():
        pytest.skip("TensorRT is installed; guard not exercised")
    with pytest.raises(RuntimeError, match="TensorRT is not installed"):
        build_engine("nonexistent.onnx", "out.trt")
