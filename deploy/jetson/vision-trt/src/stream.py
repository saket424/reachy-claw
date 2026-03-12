"""Video streaming — holds pre-encoded JPEG frames for MJPEG output.

The GstCameraCapture pipeline handles all encoding in hardware.
This module simply buffers the latest JPEG for HTTP streaming.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class VideoStreamer:
    """JPEG buffer for MJPEG HTTP streaming."""

    def __init__(self, port: int = 8632):
        self._port = port
        self._latest_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()
        self._has_clients = False

    def set_jpeg(self, data: bytes) -> None:
        """Store a pre-encoded JPEG frame."""
        with self._frame_lock:
            self._latest_jpeg = data

    def get_jpeg(self) -> bytes | None:
        """Get latest JPEG frame for HTTP streaming."""
        self._has_clients = True
        with self._frame_lock:
            return self._latest_jpeg

    def close(self):
        """No-op (capture pipeline owns lifecycle)."""
        pass
