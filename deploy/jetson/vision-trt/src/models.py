"""TensorRT engine loading and management.

Handles ONNX → TensorRT conversion (cached) and engine lifecycle.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TRTEngine:
    """Wrapper around a TensorRT engine with numpy I/O."""

    def __init__(self, engine_path: str, onnx_path: str | None = None):
        self._engine_path = engine_path
        self._onnx_path = onnx_path
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = {}  # name → (device_ptr, host_array, shape)

    def build_or_load(self) -> bool:
        """Load cached engine or build from ONNX."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            logger.error(f"TensorRT/PyCUDA not available: {e}")
            return False

        trt_logger = trt.Logger(trt.Logger.WARNING)

        if os.path.exists(self._engine_path):
            logger.info(f"Loading cached TRT engine: {self._engine_path}")
            return self._load_engine(trt_logger)

        if not self._onnx_path or not os.path.exists(self._onnx_path):
            logger.error(f"No ONNX model found: {self._onnx_path}")
            return False

        logger.info(f"Building TRT engine from {self._onnx_path} (this may take 20-60s)...")
        return self._build_engine(trt_logger)

    def _load_engine(self, trt_logger) -> bool:
        import tensorrt as trt
        import pycuda.driver as cuda

        runtime = trt.Runtime(trt_logger)
        with open(self._engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            logger.error("Failed to deserialize TRT engine")
            return False

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._allocate_buffers()
        return True

    def _build_engine(self, trt_logger) -> bool:
        import tensorrt as trt
        import pycuda.driver as cuda

        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        with open(self._onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            logger.error("TRT engine build failed")
            return False

        # Cache the engine
        os.makedirs(os.path.dirname(self._engine_path), exist_ok=True)
        with open(self._engine_path, "wb") as f:
            f.write(engine_bytes)
        logger.info(f"TRT engine cached: {self._engine_path}")

        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._allocate_buffers()
        return True

    def _allocate_buffers(self):
        import pycuda.driver as cuda

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = np.float32  # assume FP32 I/O
            size = int(np.prod(shape))
            host = np.empty(size, dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self._bindings[name] = (device, host, tuple(shape))

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs."""
        import pycuda.driver as cuda

        for name, data in inputs.items():
            if name not in self._bindings:
                continue
            device, host, shape = self._bindings[name]
            np.copyto(host, data.ravel())
            cuda.memcpy_htod_async(device, host, self._stream)

        # Set tensor addresses
        for name, (device, _, _) in self._bindings.items():
            self._context.set_tensor_address(name, int(device))

        self._context.execute_async_v3(self._stream.handle)

        outputs = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode.name == "OUTPUT":
                device, host, shape = self._bindings[name]
                cuda.memcpy_dtoh_async(host, device, self._stream)
                outputs[name] = host.reshape(shape).copy()

        self._stream.synchronize()
        return outputs

    def close(self):
        """Release GPU resources."""
        self._bindings.clear()
        self._context = None
        self._engine = None


def load_engines(model_dir: str, engine_dir: str) -> dict[str, TRTEngine]:
    """Load all model engines (SCRFD, MobileFaceNet, HSEmotion)."""
    model_dir = Path(model_dir)
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    engines = {}
    model_specs = {
        "scrfd": "scrfd_2.5g_bnkps.onnx",
        "arcface": "w600k_mbf.onnx",
        "emotion": "enet_b0_8_best_afew.onnx",
    }

    for name, onnx_file in model_specs.items():
        onnx_path = model_dir / onnx_file
        engine_path = engine_dir / f"{name}.engine"

        engine = TRTEngine(str(engine_path), str(onnx_path))
        if engine.build_or_load():
            engines[name] = engine
            logger.info(f"Engine ready: {name}")
        else:
            logger.warning(f"Engine failed: {name} (service will run without it)")

    return engines
