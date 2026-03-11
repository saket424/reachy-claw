"""VisionClientPlugin -- remote TensorRT vision service integration.

Captures frames from the SDK camera, writes them to shared memory for
the vision-trt container, and receives inference results via ZMQ SUB
to drive head tracking and emotion mapping.
"""

import asyncio
import logging
import struct
import time

import numpy as np

from ..motion.head_target import HeadTarget
from ..plugin import Plugin

logger = logging.getLogger(__name__)

# Shared memory frame header: width(u32) + height(u32) + channels(u32) + frame_id(u32)
_SHM_HEADER_FMT = "<IIII"
_SHM_HEADER_SIZE = struct.calcsize(_SHM_HEADER_FMT)

# HSEmotion output → EmotionMapper key
_EMOTION_REMAP = {
    "Anger": "angry",
    "Contempt": "neutral",
    "Disgust": "angry",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprised",
    # Lowercase variants (in case vision-trt sends lowercase)
    "anger": "angry",
    "contempt": "neutral",
    "disgust": "angry",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "angry": "angry",
}


class VisionClientPlugin(Plugin):
    """Remote vision service client: shm frame publisher + ZMQ result consumer."""

    name = "vision_client"

    def __init__(self, app):
        super().__init__(app)
        cfg = app.config
        self._shm_path = cfg.vision_shm_path
        self._zmq_url = cfg.vision_service_url
        self._max_yaw = cfg.vision_max_yaw
        self._max_pitch = cfg.vision_max_pitch
        self._smoothing_alpha = cfg.vision_smoothing_alpha
        self._deadzone = cfg.vision_deadzone
        self._face_lost_delay = cfg.vision_face_lost_delay
        self._emotion_threshold = cfg.vision_emotion_threshold
        self._emotion_cooldown = cfg.vision_emotion_cooldown

        # Frame publisher state
        self._frame_id = 0
        self._shm_fd = None
        self._shm_mmap = None
        self._shm_size = 0

        # Smoothing state (same as FaceTrackerPlugin)
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._last_face_time = 0.0
        self._face_lost_published = False

        # Emotion state
        self._last_emotion = "neutral"
        self._last_emotion_time = 0.0

        # Identity (shared with app for conversation context)
        self.current_identity = None

    def setup(self) -> bool:
        """Check SDK camera and ZMQ availability."""
        # Check SDK camera
        reachy = self.app.reachy
        if reachy is None:
            logger.warning("No robot connection, vision client skipped")
            return False

        for attempt in range(5):
            media = getattr(reachy, "media_manager", None)
            if media and getattr(media, "camera", None) is not None:
                logger.info("SDK camera available for vision client")
                break
            if attempt < 4:
                import time as _time
                _time.sleep(0.5)
        else:
            logger.warning("SDK camera not available, vision client skipped")
            return False

        # Check ZMQ
        try:
            import zmq  # noqa: F401
            import msgpack  # noqa: F401
        except ImportError as e:
            logger.warning(f"ZMQ/msgpack not installed ({e}), vision client skipped")
            return False

        return True

    def _init_shm(self, width: int, height: int, channels: int) -> None:
        """Initialize or resize shared memory for frame publishing."""
        needed = _SHM_HEADER_SIZE + (width * height * channels)
        if self._shm_mmap is not None and self._shm_size >= needed:
            return

        self._cleanup_shm()

        import mmap
        import os

        # Create or open the shm file
        fd = os.open(self._shm_path, os.O_RDWR | os.O_CREAT, 0o666)
        os.ftruncate(fd, needed)
        mm = mmap.mmap(fd, needed)
        os.close(fd)

        self._shm_mmap = mm
        self._shm_size = needed
        logger.info(
            f"SHM initialized: {self._shm_path} ({width}x{height}x{channels}, "
            f"{needed / 1024:.0f} KB)"
        )

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a BGR frame to shared memory with header."""
        h, w, c = frame.shape
        self._init_shm(w, h, c)

        self._frame_id += 1
        header = struct.pack(_SHM_HEADER_FMT, w, h, c, self._frame_id)

        mm = self._shm_mmap
        mm.seek(0)
        mm.write(header)
        mm.write(frame.tobytes())

    def _cleanup_shm(self) -> None:
        """Close shared memory mappings."""
        if self._shm_mmap is not None:
            try:
                self._shm_mmap.close()
            except Exception:
                pass
            self._shm_mmap = None
            self._shm_size = 0

    async def start(self):
        import zmq
        import zmq.asyncio
        import msgpack

        ctx = zmq.asyncio.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"vision")
        sub.setsockopt(zmq.CONFLATE, 1)  # only keep latest message
        sub.connect(self._zmq_url)

        logger.info(
            f"Vision client started (shm={self._shm_path}, zmq={self._zmq_url})"
        )

        frame_task = asyncio.create_task(self._frame_loop())
        result_task = asyncio.create_task(self._result_loop(sub, msgpack))

        try:
            await asyncio.gather(frame_task, result_task)
        finally:
            frame_task.cancel()
            result_task.cancel()
            sub.close()
            ctx.term()
            self._cleanup_shm()
            logger.info("Vision client stopped")

    async def _frame_loop(self):
        """Capture SDK frames and write to shared memory at ~25Hz."""
        consecutive_errors = 0
        while self._running:
            try:
                frame = await asyncio.to_thread(self.app.reachy.media.get_frame)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                    logger.warning(f"Frame capture error ({consecutive_errors}): {e}")
                await asyncio.sleep(1.0 if consecutive_errors >= 50 else 0.04)
                continue

            if frame is None:
                await asyncio.sleep(0.04)
                continue

            consecutive_errors = 0

            try:
                await asyncio.to_thread(self._write_frame, frame)
            except Exception as e:
                logger.error(f"SHM write error: {e}")

            await asyncio.sleep(0.04)  # ~25Hz

    async def _result_loop(self, sub, msgpack):
        """Receive inference results from vision-trt via ZMQ."""
        import zmq

        poller = zmq.asyncio.Poller()
        poller.register(sub, zmq.POLLIN)

        while self._running:
            events = dict(await poller.poll(timeout=100))
            if sub not in events:
                continue

            try:
                topic, data = await sub.recv_multipart()
                msg = msgpack.unpackb(data, raw=False)
            except Exception as e:
                logger.debug(f"ZMQ recv error: {e}")
                continue

            faces = msg.get("faces", [])
            now = time.monotonic()

            if not faces:
                if (now - self._last_face_time) > self._face_lost_delay:
                    if not self._face_lost_published:
                        self.app.head_targets.publish(
                            HeadTarget(
                                yaw=0.0, pitch=0.0, confidence=0.0,
                                source="face", timestamp=now,
                            )
                        )
                        self._face_lost_published = True
                        self.current_identity = None
                continue

            # Select primary face (largest bbox area)
            primary = max(
                faces,
                key=lambda f: (
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                    if "bbox" in f else 0
                ),
            )

            self._last_face_time = now
            self._face_lost_published = False

            # Head tracking (same logic as FaceTrackerPlugin)
            center = primary.get("center")
            if center:
                face_x, face_y = float(center[0]), float(center[1])

                if (
                    abs(face_x - self._smooth_x) >= self._deadzone
                    or abs(face_y - self._smooth_y) >= self._deadzone
                ):
                    self._smooth_x += self._smoothing_alpha * (face_x - self._smooth_x)
                    self._smooth_y += self._smoothing_alpha * (face_y - self._smooth_y)

                yaw = -self._smooth_x * self._max_yaw
                pitch = -self._smooth_y * self._max_pitch

                self.app.head_targets.publish(
                    HeadTarget(
                        yaw=yaw, pitch=pitch, confidence=0.9,
                        source="face", timestamp=now,
                    )
                )

            # Emotion mapping
            emotion = primary.get("emotion")
            emotion_conf = primary.get("emotion_confidence", 0.0)
            if (
                emotion
                and emotion_conf >= self._emotion_threshold
                and (now - self._last_emotion_time) >= self._emotion_cooldown
            ):
                mapped = _EMOTION_REMAP.get(emotion)
                if mapped and mapped != self._last_emotion:
                    self.app.emotions.queue_emotion(mapped)
                    self._last_emotion = mapped
                    self._last_emotion_time = now
                    logger.debug(
                        f"Vision emotion: {emotion} → {mapped} "
                        f"(conf={emotion_conf:.2f})"
                    )

            # Identity
            identity = primary.get("identity")
            if identity != self.current_identity:
                self.current_identity = identity
                if identity:
                    logger.info(f"Face identified: {identity}")

    async def stop(self):
        self._running = False
