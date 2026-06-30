"""Export the PointPillars backbone + detection head to ONNX, verified against
ONNX Runtime on CPU.

Only the dense part of the network (BEV pseudo-image -> detection maps) is exported.
The pillarization / scatter step stays in Python pre-processing **outside** the ONNX
graph: it depends on dynamic point counts and uses scatter ops that TensorRT ingests
poorly (ADR-004). The camera branch (Ultralytics) exports through its own
``yolo export format=onnx`` path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from perceptnet.models.lidar_branch import PointPillars


class BackboneHead(nn.Module):
    """ONNX-friendly wrapper: BEV pseudo-image in, (cls, box, dir) tensors out."""

    def __init__(self, model: PointPillars):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, bev: torch.Tensor):
        feats = self.backbone(bev)
        out = self.head(feats)
        return out["cls_preds"], out["box_preds"], out["dir_preds"]


def export_backbone_head(
    model: PointPillars,
    output_path: str,
    grid_size: Optional[Tuple[int, int]] = None,
    opset: int = 17,
    dynamic_batch: bool = False,
) -> str:
    """Export ``model``'s backbone+head to ONNX. Returns the output path."""
    wrapper = BackboneHead(model).eval()
    nx, ny = grid_size or model.cfg.grid_size
    dummy = torch.randn(1, model.cfg.pillar_feat_channels, ny, nx)

    dynamic_axes = {"bev": {0: "batch"}} if dynamic_batch else None
    kwargs = dict(
        opset_version=opset,
        input_names=["bev"],
        output_names=["cls_preds", "box_preds", "dir_preds"],
        dynamic_axes=dynamic_axes,
    )
    # Use the legacy TorchScript exporter (stable for conv/bn/convtranspose/resize and
    # free of the onnxscript dependency the new dynamo exporter pulls in). Fall back
    # for torch versions that don't accept the `dynamo` kwarg.
    try:
        torch.onnx.export(wrapper, dummy, output_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, output_path, **kwargs)
    return output_path


def verify_onnx(onnx_path: str, model: PointPillars, grid_size: Optional[Tuple[int, int]] = None,
                atol: float = 1e-3) -> bool:
    """Check the ONNX graph matches PyTorch on a random BEV input (CPU)."""
    import numpy as np
    import onnxruntime as ort

    wrapper = BackboneHead(model).eval()
    nx, ny = grid_size or model.cfg.grid_size
    dummy = torch.randn(1, model.cfg.pillar_feat_channels, ny, nx)

    with torch.no_grad():
        torch_out = wrapper(dummy)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"bev": dummy.numpy()})

    return all(
        np.allclose(t.numpy(), o, atol=atol)
        for t, o in zip(torch_out, ort_out)
    )
