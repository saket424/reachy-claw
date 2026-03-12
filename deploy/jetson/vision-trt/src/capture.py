"""GStreamer camera capture with hardware-accelerated decode, resize, and encode.

Full HW pipeline (no CPU involvement in video path):
  v4l2src (MJPEG) → jpegparse → nvv4l2decoder → tee
    ├→ nvvidconv 640x640 BGRx → appsink (inference)
    └→ nvvidconv 640w → nvjpegenc → appsink (MJPEG stream)

IMPORTANT: Do NOT set GST_V4L2_USE_LIBV4L2=1 — it loads libnvv4l2 which
transparently decodes MJPEG and hides the format from GStreamer, forcing
v4l2src to output raw YUY2 at max resolution (10MP). Without it, v4l2src
exposes image/jpeg caps and we can use nvv4l2decoder for HW MJPEG decode.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class GstCameraCapture:
    """GStreamer camera capture with dual output: inference frames + MJPEG stream."""

    def __init__(
        self,
        device: str = "/dev/video0",
        cam_width: int = 1920,
        cam_height: int = 1080,
        cam_fps: int = 30,
        inf_width: int = 640,
        inf_height: int = 640,
        stream_width: int = 640,
    ):
        self._device = device
        self._cam_width = cam_width
        self._cam_height = cam_height
        self._cam_fps = cam_fps
        self._inf_width = inf_width
        self._inf_height = inf_height
        self._stream_width = stream_width

        self._pipeline = None
        self._inference_sink = None
        self._stream_sink = None
        self._Gst = None
        self._running = False
        self._hw_pipeline = False

        # Latest JPEG for MJPEG stream (thread-safe)
        self._latest_jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()

    def start(self) -> bool:
        """Build and start the GStreamer pipeline.

        Tries HW-accelerated pipeline first, falls back to CPU.
        Returns True if pipeline started successfully.
        """
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp  # noqa: F401
            Gst.init(None)
            self._Gst = Gst
        except Exception as e:
            logger.error(f"GStreamer not available: {e}")
            return False

        # Try HW pipeline first (nvv4l2decoder + nvvidconv)
        if self._try_hw_pipeline():
            self._hw_pipeline = True
            logger.info("Camera capture started (HW: nvv4l2decoder + nvvidconv + nvjpegenc)")
            return True

        # Fall back to CPU pipeline
        if self._try_cpu_pipeline():
            self._hw_pipeline = False
            logger.info("Camera capture started (CPU fallback: jpegdec + videoconvert)")
            return True

        logger.error("Failed to start camera capture (both HW and CPU)")
        return False

    def _try_hw_pipeline(self) -> bool:
        """Full HW pipeline: MJPEG decode + resize + JPEG encode all on GPU/VIC.

        v4l2src exposes image/jpeg when GST_V4L2_USE_LIBV4L2 is NOT set.
        nvv4l2decoder does HW MJPEG decode, nvvidconv does HW resize.
        No CPU videoconvert needed — entire path is hardware-accelerated.
        """
        Gst = self._Gst

        # Check required plugins
        for name in ("nvv4l2decoder", "nvvidconv", "nvjpegenc"):
            if Gst.ElementFactory.find(name) is None:
                logger.debug(f"HW plugin {name} not found")
                return False

        stream_height = int(self._cam_height * self._stream_width / self._cam_width)

        pipeline_str = (
            f"v4l2src device={self._device} "
            f"! image/jpeg,width={self._cam_width},height={self._cam_height} "
            "! jpegparse "
            "! nvv4l2decoder mjpeg=1 "
            "! tee name=t "
            # Inference branch: nvvidconv resize to 640x640 BGRx
            "t. ! queue leaky=downstream max-size-buffers=1 "
            f"! nvvidconv ! video/x-raw,format=BGRx,width={self._inf_width},"
            f"height={self._inf_height} "
            "! appsink name=inference_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false "
            # Stream branch: nvvidconv resize + HW JPEG encode
            "t. ! queue leaky=downstream max-size-buffers=1 "
            f"! nvvidconv ! video/x-raw(memory:NVMM),width={self._stream_width},"
            f"height={stream_height} "
            "! nvjpegenc "
            "! appsink name=stream_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false"
        )

        return self._launch_pipeline(pipeline_str)

    def _try_cpu_pipeline(self) -> bool:
        """CPU fallback: jpegdec + videoscale + videoconvert."""
        stream_height = int(self._cam_height * self._stream_width / self._cam_width)

        pipeline_str = (
            f"v4l2src device={self._device} "
            f"! image/jpeg,width={self._cam_width},height={self._cam_height} "
            "! jpegdec "
            f"! videoscale ! video/x-raw,width={self._inf_width},height={self._inf_height} "
            "! videoconvert ! video/x-raw,format=BGR "
            "! tee name=t "
            # Inference branch (already at target size)
            "t. ! queue leaky=downstream max-size-buffers=1 "
            "! appsink name=inference_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false "
            # Stream branch (CPU JPEG encode)
            "t. ! queue leaky=downstream max-size-buffers=1 "
            "! jpegenc quality=70 "
            "! appsink name=stream_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false"
        )

        return self._launch_pipeline(pipeline_str)

    def _launch_pipeline(self, pipeline_str: str) -> bool:
        """Parse and start a GStreamer pipeline."""
        Gst = self._Gst
        logger.info(f"Launching pipeline: {pipeline_str}")
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            self._inference_sink = pipeline.get_by_name("inference_sink")
            self._stream_sink = pipeline.get_by_name("stream_sink")

            if not self._inference_sink or not self._stream_sink:
                logger.error("Failed to get appsink elements")
                return False

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Pipeline failed to start")
                pipeline.set_state(Gst.State.NULL)
                return False

            # Wait for pipeline to reach PLAYING (5s timeout)
            _, state, _ = pipeline.get_state(5 * Gst.SECOND)
            logger.info(f"Pipeline state after preroll: {state.value_name}")

            # Check bus for errors
            bus = pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR)
            if msg:
                err, debug = msg.parse_error()
                logger.error(f"Pipeline bus error: {err.message}")
                logger.debug(f"Debug: {debug}")
                pipeline.set_state(Gst.State.NULL)
                return False

            if state != Gst.State.PLAYING:
                logger.warning(f"Pipeline stuck in {state.value_name}, aborting")
                pipeline.set_state(Gst.State.NULL)
                return False

            self._pipeline = pipeline
            self._running = True
            return True

        except Exception as e:
            logger.error(f"Pipeline launch failed: {e}")
            return False

    def get_inference_frame(self) -> np.ndarray | None:
        """Pull a 640x640 BGR frame for inference (non-blocking).

        Returns None if no frame is available.
        """
        if not self._running or not self._inference_sink:
            return None

        Gst = self._Gst
        # Use 100ms timeout instead of non-blocking to catch preroll
        sample = self._inference_sink.try_pull_sample(100 * 1000000)  # 100ms in ns
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = struct.get_int("width")[1]
        h = struct.get_int("height")[1]

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None

        # HW pipeline outputs BGRx (4ch), CPU pipeline outputs BGR (3ch)
        channels = len(map_info.data) // (h * w)
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, channels)
        buf.unmap(map_info)
        if channels == 4:
            return frame[:, :, :3].copy()  # BGRx → BGR (strip alpha)
        return frame.copy()

    def get_stream_jpeg(self) -> bytes | None:
        """Pull a HW-encoded JPEG frame for MJPEG stream (non-blocking).

        Returns None if no frame is available.
        """
        if not self._running or not self._stream_sink:
            return None

        Gst = self._Gst
        sample = self._stream_sink.try_pull_sample(100 * 1000000)  # 100ms in ns
        if sample is None:
            return None

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None

        jpeg = bytes(map_info.data)
        buf.unmap(map_info)
        return jpeg

    def close(self):
        """Stop the pipeline and release resources."""
        self._running = False
        if self._pipeline and self._Gst:
            self._pipeline.send_event(self._Gst.Event.new_eos())
            self._pipeline.set_state(self._Gst.State.NULL)
            self._pipeline = None
            logger.info("Camera capture stopped")
