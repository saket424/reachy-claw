"""Voice Activity Detection backends for Reachy Mini.

All VAD runs locally. Backends differ in compute: CPU (energy/silero) vs NPU (hailo).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from clawd_reachy_mini.backend_registry import register_vad

logger = logging.getLogger(__name__)


class VADBackend(ABC):
    """Abstract base class for VAD backends."""

    @abstractmethod
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains speech."""

    def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""

    def preload(self) -> None:
        """Pre-load model to avoid delay on first call."""


@register_vad("silero")
class SileroVAD(VADBackend):
    """Silero VAD using ONNX runtime — lightweight, no torch.

    Downloads the ONNX model (~2MB) on first use. Runs on CPU via onnxruntime.
    """

    class Settings:
        threshold: float = 0.5

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._session = None
        self._state = None
        self._sr = None

    def preload(self) -> None:
        self._load_model()

    def _model_path(self) -> str:
        import urllib.request
        from pathlib import Path

        cache_dir = Path.home() / ".clawd-reachy-mini" / "cache" / "silero-vad"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "silero_vad.onnx"

        if not model_path.exists():
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            logger.info("Downloading Silero VAD ONNX model...")
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def _load_model(self):
        if self._session is not None:
            return

        import onnxruntime

        logger.info("Loading Silero VAD (onnxruntime)...")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            self._model_path(), sess_options=opts
        )
        self._reset_state()
        logger.info("Silero VAD ready")

    def _reset_state(self):
        # State shape: (2, 1, 128) for current Silero VAD ONNX model
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Process in 512-sample chunks (optimal for Silero VAD at 16kHz)
        chunk_size = 512
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            ort_inputs = {
                "input": chunk.reshape(1, -1),
                "state": self._state,
                "sr": self._sr,
            }
            ort_outputs = self._session.run(None, ort_inputs)
            prob = ort_outputs[0].item()
            self._state = ort_outputs[1]

            if prob > self._threshold:
                return True
        return False

    def reset(self) -> None:
        if self._session is not None:
            self._reset_state()

    @property
    def threshold(self) -> float:
        return self._threshold


@register_vad("energy")
class EnergyVAD(VADBackend):
    """Simple energy-based VAD — zero dependencies, works everywhere."""

    class Settings:
        threshold: float = 0.01

    def __init__(self, threshold: float = 0.01):
        self._threshold = threshold

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        return float(np.abs(audio).mean()) > self._threshold


def create_vad_backend(
    backend: str = "energy",
    config=None,
) -> VADBackend:
    """Create a VAD backend by name using the registry."""
    from clawd_reachy_mini.backend_registry import get_vad_info, get_vad_names

    name = backend.lower().strip()

    info = get_vad_info(name)
    if info is None:
        available = ", ".join(get_vad_names())
        raise ValueError(f"Unknown VAD backend: {backend!r}. Choose from: {available}")

    kwargs = {}
    if config:
        for field_name in info.settings_fields:
            config_key = f"{info.name}_{field_name}"
            if hasattr(config, config_key):
                kwargs[field_name] = getattr(config, config_key)

    import inspect

    sig = inspect.signature(info.cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    logger.info(f"Using VAD backend: {name}")
    return info.cls(**filtered)
