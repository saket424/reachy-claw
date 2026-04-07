"""ONNX Runtime inference engine.

Replaces the previous TensorRT implementation. Uses onnxruntime (CPU) which
is available on l4t-base without needing TensorRT native libraries.
GPU execution via CUDAExecutionProvider is used if available.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OrtEngine:
    """Wrapper around an ONNX Runtime session with numpy I/O."""

    def __init__(self, onnx_path: str):
        self._onnx_path = onnx_path
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def build_or_load(self) -> bool:
        """Load the ONNX model via ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            logger.error(f"onnxruntime not available: {e}")
            return False

        if not os.path.exists(self._onnx_path):
            logger.error(f"ONNX model not found: {self._onnx_path}")
            return False

        # Prefer GPU if CUDAExecutionProvider is available
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"Loading {self._onnx_path} with CUDA provider")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info(f"Loading {self._onnx_path} with CPU provider")

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                self._onnx_path, sess_options=opts, providers=providers
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            logger.info(
                f"ORT session ready: {os.path.basename(self._onnx_path)} "
                f"inputs={self._input_names} outputs={self._output_names}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create ORT session for {self._onnx_path}: {e}")
            return False

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. inputs keyed by input tensor name."""
        if self._session is None:
            return {}
        feed = {k: v.astype(np.float32) for k, v in inputs.items() if k in self._input_names}
        results = self._session.run(self._output_names, feed)
        return dict(zip(self._output_names, results))

    def close(self):
        self._session = None


def load_engines(model_dir: str, engine_dir: str) -> dict[str, "OrtEngine"]:
    """Load all ONNX models (SCRFD, MobileFaceNet, HSEmotion).

    engine_dir is kept for API compatibility but unused — ORT has no separate
    build step.
    """
    model_dir = Path(model_dir)
    engines = {}

    model_specs = {
        # name: onnx_file
        "scrfd": "scrfd_10g_bnkps.onnx",
        "arcface": "w600k_mbf.onnx",
        # FER+ model: 64x64 grayscale input, 8-class output
        # From ONNX Model Zoo (public domain)
        "emotion": "emotion-ferplus-8.onnx",
    }

    for name, onnx_file in model_specs.items():
        onnx_path = model_dir / onnx_file
        engine = OrtEngine(str(onnx_path))
        if engine.build_or_load():
            engines[name] = engine
            logger.info(f"Engine ready: {name}")
        else:
            logger.warning(f"Engine failed: {name} (service will run without it)")

    return engines
