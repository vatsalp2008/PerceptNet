"""Build a TensorRT engine from an ONNX model (FP16 / INT8).

**Cloud-only (NVIDIA GPU).** TensorRT and pycuda have no macOS wheels, so the import
is guarded: this module imports fine everywhere, but calling :func:`build_engine`
without TensorRT raises a clear error. Run it inside ``docker/Dockerfile.cuda``.

INT8 calibration uses a small representative set of real BEV pseudo-images; the
acceptable mAP-drop threshold and FP16-vs-INT8 trade-off are documented in ADR-004.
"""

from __future__ import annotations

from typing import Optional

try:
    import tensorrt as trt

    _HAS_TRT = True
except ImportError:  # pragma: no cover - expected off the GPU box
    trt = None
    _HAS_TRT = False


def tensorrt_available() -> bool:
    return _HAS_TRT


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace_gb: int = 4,
    int8_calibrator: Optional[object] = None,
) -> str:
    """Build and serialize a TensorRT engine from an ONNX file.

    Args:
        onnx_path: source ONNX model.
        engine_path: output ``.trt`` path.
        precision: ``"fp32"``, ``"fp16"``, or ``"int8"``.
        workspace_gb: builder workspace in GiB.
        int8_calibrator: a ``trt.IInt8Calibrator`` (required for ``"int8"``).

    Returns:
        ``engine_path`` on success.
    """
    if not _HAS_TRT:
        raise RuntimeError(
            "TensorRT is not installed. build_engine runs only on the CUDA box "
            "(see requirements-gpu.txt / docker/Dockerfile.cuda)."
        )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"failed to parse ONNX:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if int8_calibrator is None:
            raise ValueError("int8 precision requires an int8_calibrator")
        config.int8_calibrator = int8_calibrator
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path
