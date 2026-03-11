"""Main server: shared memory reader + inference pipeline + outputs.

Reads frames from /dev/shm, runs TensorRT inference, and publishes:
1. ZMQ PUB → reachy-claw (structured data)
2. WebSocket → browser (JSON detections)
3. HTTP API → face DB CRUD + health + stats
4. Video stream → browser (MJPEG or NVENC)
"""

import asyncio
import json
import logging
import mmap
import os
import struct
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lazy imports for modules that need GPU
_zmq = None
_msgpack = None

# Frame header format (must match VisionClientPlugin)
_SHM_HEADER_FMT = "<IIII"
_SHM_HEADER_SIZE = struct.calcsize(_SHM_HEADER_FMT)


class VisionService:
    """Core vision inference service."""

    def __init__(self):
        from .config import config

        self.config = config
        self.pipeline = None
        self.face_db = None
        self.streamer = None
        self._zmq_pub = None
        self._zmq_ctx = None
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Stats
        self._fps = 0.0
        self._inference_ms = 0.0
        self._frame_count = 0
        self._last_stats_time = time.monotonic()
        self._last_frame_id = 0

    def init(self):
        """Initialize all components."""
        from .config import config
        from .face_database import FaceDatabase
        from .stream import VideoStreamer

        global _zmq, _msgpack
        import zmq
        import msgpack
        _zmq = zmq
        _msgpack = msgpack

        # Face database
        self.face_db = FaceDatabase(config.DATA_DIR)

        # Video streamer
        self.streamer = VideoStreamer(config.STREAM_PORT)
        self.streamer.start_gstreamer()

        # ZMQ publisher
        self._zmq_ctx = zmq.Context()
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(f"tcp://0.0.0.0:{config.ZMQ_PUB_PORT}")
        logger.info(f"ZMQ PUB bound on port {config.ZMQ_PUB_PORT}")

        # TRT engines (may take 20-60s on first boot)
        try:
            from .models import load_engines

            engines = load_engines(config.MODEL_DIR, config.ENGINE_DIR)
            if engines:
                from .inference import VisionPipeline

                self.pipeline = VisionPipeline(engines, self.face_db, config)
                logger.info(f"Vision pipeline ready ({len(engines)} engines)")
            else:
                logger.warning("No TRT engines loaded, running in passthrough mode")
        except Exception as e:
            logger.error(f"Failed to load TRT engines: {e}")
            logger.warning("Running in passthrough mode (no inference)")

    def read_shm_frame(self) -> tuple[np.ndarray | None, int]:
        """Read a frame from shared memory.

        Returns (frame_bgr, frame_id) or (None, 0).
        """
        path = self.config.SHM_FRAME_PATH
        if not os.path.exists(path):
            return None, 0

        try:
            fd = os.open(path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            if file_size < _SHM_HEADER_SIZE:
                os.close(fd)
                return None, 0

            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
            os.close(fd)

            header = mm[:_SHM_HEADER_SIZE]
            w, h, c, frame_id = struct.unpack(_SHM_HEADER_FMT, header)

            expected = _SHM_HEADER_SIZE + (w * h * c)
            if file_size < expected:
                mm.close()
                return None, 0

            data = mm[_SHM_HEADER_SIZE:expected]
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c).copy()
            mm.close()
            return frame, frame_id

        except Exception as e:
            logger.debug(f"SHM read error: {e}")
            return None, 0

    def process_and_publish(self, frame: np.ndarray, frame_id: int):
        """Run inference and publish results to all outputs."""
        t0 = time.monotonic()

        # Run inference
        results = []
        if self.pipeline:
            results = self.pipeline.process_frame(frame)

        t1 = time.monotonic()
        self._inference_ms = (t1 - t0) * 1000

        # Update stats
        self._frame_count += 1
        elapsed = t1 - self._last_stats_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_stats_time = t1

        # Build messages
        zmq_msg = {
            "timestamp": t0,
            "frame_id": frame_id,
            "faces": [
                {
                    "center": r.center,
                    "bbox": r.bbox,
                    "landmarks": r.landmarks,
                    "confidence": r.confidence,
                    "embedding": r.embedding,
                    "emotion": r.emotion,
                    "emotion_confidence": r.emotion_confidence,
                    "identity": r.identity,
                    "identity_distance": r.identity_distance,
                }
                for r in results
            ],
        }

        ws_msg = {
            "frame_id": frame_id,
            "faces": [
                {
                    "bbox": r.bbox,
                    "landmarks": r.landmarks,
                    "emotion": r.emotion,
                    "emotion_confidence": r.emotion_confidence,
                    "identity": r.identity,
                }
                for r in results
            ],
            "stats": {
                "fps": round(self._fps, 1),
                "inference_ms": round(self._inference_ms, 1),
            },
        }

        # ZMQ PUB
        try:
            self._zmq_pub.send_multipart([
                b"vision",
                _msgpack.packb(zmq_msg, use_bin_type=True),
            ])
        except Exception as e:
            logger.debug(f"ZMQ pub error: {e}")

        # WebSocket broadcast
        ws_json = json.dumps(ws_msg)
        with self._ws_lock:
            dead = set()
            for ws in self._ws_clients:
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(ws_json),
                        self._loop,
                    )
                except Exception:
                    dead.add(ws)
            self._ws_clients -= dead

        # Video stream
        if self.streamer:
            self.streamer.push_frame(frame)

    def run_loop(self):
        """Main inference loop (runs in thread)."""
        target_interval = 1.0 / self.config.TARGET_FPS
        logger.info(f"Inference loop started (target {self.config.TARGET_FPS} FPS)")

        while self._running:
            t0 = time.monotonic()

            frame, frame_id = self.read_shm_frame()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.005)
                continue

            self._last_frame_id = frame_id
            self.process_and_publish(frame, frame_id)

            elapsed = time.monotonic() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        """Cleanup."""
        self._running = False
        if self._zmq_pub:
            self._zmq_pub.close()
        if self._zmq_ctx:
            self._zmq_ctx.term()
        if self.streamer:
            self.streamer.close()


# Global service instance
service = VisionService()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: init service + start inference thread."""
    service.init()
    service._running = True
    service._loop = asyncio.get_event_loop()

    thread = threading.Thread(target=service.run_loop, daemon=True)
    thread.start()
    logger.info("Vision service started")

    yield

    service.close()
    logger.info("Vision service stopped")


# ── FastAPI app ──────────────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Vision TRT Service", lifespan=lifespan)

# Serve static files (frontend)
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": service.pipeline is not None,
        "fps": round(service._fps, 1),
    }


@app.get("/api/stats")
async def stats():
    return {
        "fps": round(service._fps, 1),
        "inference_ms": round(service._inference_ms, 1),
        "pipeline_ready": service.pipeline is not None,
        "faces_registered": len(service.face_db.list_faces()) if service.face_db else 0,
    }


@app.get("/api/faces")
async def list_faces():
    if not service.face_db:
        return {"faces": []}
    return {"faces": service.face_db.list_faces()}


@app.post("/api/faces/enroll")
async def enroll_face(name: str = Query(...)):
    if not service.face_db or not service.pipeline:
        return {"error": "Service not ready"}, 503

    frame, _ = service.read_shm_frame()
    if frame is None:
        return {"error": "No frame available"}, 400

    results = service.pipeline.process_frame(frame)
    if not results:
        return {"error": "No face detected"}, 400

    # Use the largest face
    primary = max(
        results,
        key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
    )
    if not primary.embedding:
        return {"error": "Face embedding extraction failed"}, 500

    embedding = np.array(primary.embedding, dtype=np.float32)
    service.face_db.enroll(name, embedding)
    return {"status": "enrolled", "name": name}


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if not service.face_db:
        return {"error": "Service not ready"}, 503

    if service.face_db.delete(name):
        return {"status": "deleted", "name": name}
    return {"error": "Face not found"}, 404


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with service._ws_lock:
        service._ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    finally:
        with service._ws_lock:
            service._ws_clients.discard(websocket)


@app.get("/stream")
async def mjpeg_stream():
    """MJPEG stream fallback for browsers."""

    async def generate():
        while True:
            jpg = service.streamer.get_jpeg() if service.streamer else None
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/")
async def index():
    """Serve the frontend page."""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Vision TRT Service</h1><p>Frontend not found.</p>")


if __name__ == "__main__":
    import uvicorn

    from .config import config

    uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT)
