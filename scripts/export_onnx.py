#!/usr/bin/env python
"""Export the PointPillars backbone+head to ONNX and verify against ONNX Runtime.

Runs on CPU/Mac. The pillarization/scatter stay outside the graph (ADR-004); only the
dense BEV -> detection-head subgraph is exported. The verification step is what guards
against op-support surprises before the engine is built on the GPU box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.optimization.export_onnx import export_backbone_head, verify_onnx

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Export PointPillars backbone+head to ONNX")
    parser.add_argument("--out", default=str(REPO / "outputs" / "perceptnet_backbone.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = PointPillarsConfig()
    model = PointPillars(cfg).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_backbone_head(model, str(out), opset=args.opset)
    ok = verify_onnx(str(out), model)
    print(f"exported -> {out}")
    print(f"ONNX Runtime parity vs PyTorch: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
