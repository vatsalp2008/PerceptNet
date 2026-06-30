"""Inference optimization: ONNX export, TensorRT engine build, latency benchmarking.

``build_trt`` requires TensorRT (NVIDIA GPU only) and guards its import. Imported
explicitly by callers; nothing heavy is re-exported here.
"""
