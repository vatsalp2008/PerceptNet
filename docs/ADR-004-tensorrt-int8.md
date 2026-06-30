# ADR-004: TensorRT INT8 strategy + ONNX subgraph boundaries

**Status:** Accepted

## Context
Naive PyTorch inference is too slow for the 10 Hz real-time target. NVIDIA TensorRT gives 3–5× speedups via FP16/INT8, but only ingests a static, supported op set — and some of our pipeline (dynamic point counts, scatter ops) doesn't export cleanly.

## Decision
1. **Keep pillarization/scatter OUTSIDE the ONNX graph.** Export only the dense **BEV pseudo-image → detection-head** subgraph (`perceptnet.optimization.export_onnx.BackboneHead`). Point→pillar grouping stays as Python/CUDA pre-processing. The camera branch exports via Ultralytics' own `yolo export`.
2. **Export camera and LiDAR backbones separately** rather than one mega-graph (the dynamic ROI count between them defeats single-graph export).
3. **Verify every exported subgraph against ONNX Runtime on CPU** before handing it to TensorRT (`verify_onnx`), so op-support breaks surface on the laptop, not the GPU box.
4. **Precision ladder:** FP32 baseline → FP16 (expected ~2× speedup, <1% mAP drop) → INT8 with calibration on a representative set of real BEV pseudo-images. Adopt INT8 only if the Car-Moderate mAP drop stays **< 1%**; otherwise ship FP16.

## Consequences
- **+** Robust, debuggable export; the brittle scatter ops never reach TensorRT.
- **+** CPU ORT parity check is runnable in CI without a GPU.
- **−** Pre/post-processing (pillarization, NMS, decode) is not TensorRT-accelerated — acceptable since the dense backbone dominates latency.
- **−** INT8 calibration data must be curated and stored; the accuracy/speed trade-off is measured per-release (numbers are TARGETS until the GPU run).
