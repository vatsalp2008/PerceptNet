#!/usr/bin/env python
"""Benchmark PointPillars forward-pass latency on the available device.

Prints a p50/p95/p99 + FPS table (the README inference table is filled from runs like
this on each backend). On the GPU box, compare against the TensorRT engine via the
same harness.
"""

from __future__ import annotations

import argparse

import torch

from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.optimization.benchmark import benchmark_latency, format_benchmark_table
from perceptnet.utils import get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Benchmark PointPillars latency")
    parser.add_argument("--num-points", type=int, default=60000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    seed_everything(0)
    device = get_device()
    cfg = PointPillarsConfig()
    model = PointPillars(cfg).to(device).eval()

    x0, y0, z0, x1, y1, z1 = cfg.pc_range
    cloud = torch.rand(args.num_points, 4)
    cloud[:, 0] = cloud[:, 0] * (x1 - x0) + x0
    cloud[:, 1] = cloud[:, 1] * (y1 - y0) + y0
    cloud[:, 2] = cloud[:, 2] * (z1 - z0) + z0

    @torch.no_grad()
    def run():
        model([cloud])

    print(f"PointPillars forward, {args.num_points} pts, device={device.type} "
          f"(warmup={args.warmup}, runs={args.runs})")
    results = {f"PyTorch {device.type.upper()}": benchmark_latency(run, args.warmup, args.runs)}
    print(format_benchmark_table(results))


if __name__ == "__main__":
    main()
