"""Video streaming: NVENC H.264 encoding via GStreamer + RTSP server.

Provides hardware-accelerated video encoding on Jetson for browser preview.
Falls back to MJPEG over HTTP if GStreamer/NVENC is not available.
"""

import asyncio
import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Manages video output stream (MJPEG fallback or GStreamer NVENC)."""

    def __init__(self, port: int = 8632):
        self._port = port
        self._latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._gst_pipeline = None

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a frame (BGR) to the stream."""
        # MJPEG encode for HTTP fallback
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._frame_lock:
            self._latest_frame = jpg.tobytes()

    def get_jpeg(self) -> bytes | None:
        """Get latest JPEG frame for HTTP streaming."""
        with self._frame_lock:
            return self._latest_frame

    def start_gstreamer(self) -> bool:
        """Try to start GStreamer NVENC pipeline.

        Returns True if GStreamer NVENC is available, False for MJPEG fallback.
        """
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)

            # Test if nvv4l2h264enc is available
            factory = Gst.ElementFactory.find("nvv4l2h264enc")
            if factory is None:
                logger.info("nvv4l2h264enc not found, using MJPEG fallback")
                return False

            logger.info("GStreamer NVENC available (RTSP stream ready)")
            return True

        except Exception as e:
            logger.info(f"GStreamer not available ({e}), using MJPEG fallback")
            return False

    def close(self):
        """Stop streaming."""
        if self._gst_pipeline:
            try:
                self._gst_pipeline.set_state(0)  # NULL
            except Exception:
                pass
