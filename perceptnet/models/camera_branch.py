"""Camera branch — YOLO v8 2D detection + FPN feature extraction.

Two responsibilities (Module 2):
  - run a (pretrained or fine-tuned) YOLO v8 detector for 2D boxes, and
  - tap the neck's multi-scale feature maps (P3/P4/P5) for the fusion head.

The feature tap uses a **forward-pre-hook on the Detect layer** (its input is exactly
the list of FPN maps), rather than indexing into ``model.model[i]``. Ultralytics
renumbers internal layers across versions, so hooking the well-known Detect module is
the stable way to grab features (ADR / pitfall: Ultralytics internals churn). Weights
download on first use, so construction and inference are network-bound, not import-bound.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from perceptnet.utils import get_device


class CameraBranch:
    """Wraps an Ultralytics YOLO v8 model with FPN feature capture.

    Args:
        weights: model weights. Default ``yolov8n.pt`` keeps the smoke test light;
            the training config (``configs/fusion.yaml``) uses ``yolov8m.pt`` per spec.
        device: torch device string; ``None`` auto-selects (CUDA > MPS > CPU).
        capture_features: register the FPN hook (set False for detection-only use).
    """

    def __init__(self, weights: str = "yolov8n.pt", device: Optional[str] = None, capture_features: bool = True):
        from ultralytics import YOLO

        self.device = str(get_device(device or "auto"))
        self.yolo = YOLO(weights)
        self._features: Optional[List] = None
        self._hook_handle = None
        if capture_features:
            self._register_feature_hook()

    # ------------------------------------------------------------------ #
    def _register_feature_hook(self) -> None:
        detect_layer = self.yolo.model.model[-1]      # the Detect head

        def _capture(_module, inputs):
            # Detect receives a list of FPN feature maps [P3, P4, P5].
            self._features = list(inputs[0])

        self._hook_handle = detect_layer.register_forward_pre_hook(_capture)

    @property
    def fpn_channels(self) -> Optional[List[int]]:
        if self._features is None:
            return None
        return [f.shape[1] for f in self._features]

    # ------------------------------------------------------------------ #
    def detect(self, image, conf: float = 0.25, iou: float = 0.45) -> Dict[str, np.ndarray]:
        """Run detection on a single image (path / numpy HWC / torch CHW).

        Returns ``{boxes (N,4) xyxy, scores (N,), labels (N,)}`` as NumPy arrays.
        """
        results = self.yolo.predict(image, conf=conf, iou=iou, device=self.device, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=int)}
        return {
            "boxes": r.boxes.xyxy.cpu().numpy(),
            "scores": r.boxes.conf.cpu().numpy(),
            "labels": r.boxes.cls.cpu().numpy().astype(int),
        }

    def extract_features(self, image) -> Dict[str, "np.ndarray"]:
        """Run a forward pass and return the captured FPN maps as ``{p3, p4, p5}``.

        Maps are ordered by decreasing resolution (P3 finest). Requires
        ``capture_features=True``.
        """
        if self._hook_handle is None:
            raise RuntimeError("CameraBranch was built with capture_features=False")
        self._features = None
        self.yolo.predict(image, device=self.device, verbose=False)
        if not self._features:
            raise RuntimeError("FPN features were not captured; check the Detect hook")
        feats = sorted(self._features, key=lambda f: f.shape[-1], reverse=True)
        return {name: f for name, f in zip(("p3", "p4", "p5"), feats)}

    def __call__(self, image, conf: float = 0.25) -> Tuple[Dict, Dict]:
        """Return ``(detections, fpn_features)`` from a single forward pass."""
        self._features = None
        results = self.yolo.predict(image, conf=conf, device=self.device, verbose=False)
        r = results[0]
        dets = (
            {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=int)}
            if r.boxes is None or len(r.boxes) == 0
            else {
                "boxes": r.boxes.xyxy.cpu().numpy(),
                "scores": r.boxes.conf.cpu().numpy(),
                "labels": r.boxes.cls.cpu().numpy().astype(int),
            }
        )
        feats = {}
        if self._features:
            ordered = sorted(self._features, key=lambda f: f.shape[-1], reverse=True)
            feats = {name: f for name, f in zip(("p3", "p4", "p5"), ordered)}
        return dets, feats

    def __del__(self):
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
