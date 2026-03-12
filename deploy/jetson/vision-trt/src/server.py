"""Main server: GStreamer camera capture + inference pipeline + outputs.

Captures frames directly from the camera via GStreamer with hardware-
accelerated decode/resize/encode, runs TensorRT inference, and publishes:
1. ZMQ PUB → reachy-claw (structured data)
2. WebSocket → browser (JSON detections)
3. HTTP API → face DB CRUD + health + stats
4. Video stream → browser (MJPEG from HW encoder)
"""

import asyncio
import json
import logging
import os
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


class VisionService:
    """Core vision inference service."""

    def __init__(self):
        from .config import config

        self.config = config
        self.pipeline = None
        self.face_db = None
        self.streamer = None
        self.capture = None
        self._zmq_pub = None
        self._zmq_ctx = None
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Stats
        self._fps = 0.0
        self._inference_ms = 0.0
        self._frame_count = 0
        self._last_stats_time = time.monotonic()
        self._frame_id = 0

    def init(self):
        """Initialize all components."""
        from .capture import GstCameraCapture
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

        # Video streamer (simple JPEG buffer now)
        self.streamer = VideoStreamer(config.STREAM_PORT)

        # ZMQ publisher
        self._zmq_ctx = zmq.Context()
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(f"tcp://0.0.0.0:{config.ZMQ_PUB_PORT}")
        logger.info(f"ZMQ PUB bound on port {config.ZMQ_PUB_PORT}")

        # TRT engines first (may take 20-60s on first boot)
        # Must build before camera start — nvv4l2decoder takes GPU memory
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

        # Camera capture via GStreamer (after TRT engines to avoid GPU memory contention)
        self.capture = GstCameraCapture(
            device=config.CAMERA_DEVICE,
            cam_width=config.CAMERA_WIDTH,
            cam_height=config.CAMERA_HEIGHT,
            cam_fps=config.CAMERA_FPS,
            inf_width=config.INPUT_WIDTH,
            inf_height=config.INPUT_HEIGHT,
            stream_width=config.STREAM_WIDTH,
        )
        if not self.capture.start():
            logger.error("Camera capture failed to start")

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

        # Push HW-encoded JPEG to streamer
        if self.streamer and self.capture:
            jpeg = self.capture.get_stream_jpeg()
            if jpeg:
                self.streamer.set_jpeg(jpeg)

    def run_loop(self):
        """Main inference loop (runs in thread)."""
        target_interval = 1.0 / self.config.TARGET_FPS
        logger.info(f"Inference loop started (target {self.config.TARGET_FPS} FPS)")

        while self._running:
            t0 = time.monotonic()

            if not self.capture:
                time.sleep(0.1)
                continue

            frame = self.capture.get_inference_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self._frame_id += 1
            self.process_and_publish(frame, self._frame_id)

            elapsed = time.monotonic() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        """Cleanup."""
        self._running = False
        if self.capture:
            self.capture.close()
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, File, Form, UploadFile
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
        "capture": service.capture is not None and service.capture._running,
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

    if not service.capture:
        return {"error": "Camera not available"}, 503

    frame = service.capture.get_inference_frame()
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


@app.post("/api/faces/enroll-image")
async def enroll_face_from_image(name: str = Form(...), image: UploadFile = File(...)):
    """Register a face from an uploaded image file."""
    if not service.face_db or not service.pipeline:
        return {"error": "Service not ready"}, 503

    contents = await image.read()
    if not contents:
        return {"error": "Empty file"}, 400
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}, 400

    results = service.pipeline.process_frame(frame)
    if not results:
        return {"error": "No face detected in image"}, 400

    primary = max(
        results,
        key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
    )
    if not primary.embedding:
        return {"error": "Face embedding extraction failed"}, 500

    embedding = np.array(primary.embedding, dtype=np.float32)
    service.face_db.enroll(name, embedding)
    return {"status": "enrolled", "name": name}


@app.get("/api/faces/export")
async def export_faces():
    """Export all face data as a zip archive."""
    import io
    import zipfile

    if not service.face_db:
        return {"error": "Service not ready"}, 503

    buf = io.BytesIO()
    data_dir = service.face_db._dir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in data_dir.iterdir():
            if fpath.suffix in (".json", ".npy"):
                zf.write(fpath, fpath.name)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=faces.zip"},
    )


@app.post("/api/faces/import")
async def import_faces(file: UploadFile = File(...)):
    """Import face data from a zip archive (replaces existing)."""
    import io
    import zipfile

    if not service.face_db:
        return {"error": "Service not ready"}, 503

    contents = await file.read()
    buf = io.BytesIO(contents)
    if not zipfile.is_zipfile(buf):
        return {"error": "Invalid zip file"}, 400

    data_dir = service.face_db._dir
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            # Only extract .json and .npy files, no path traversal
            if name != os.path.basename(name):
                continue
            if not name.endswith((".json", ".npy")):
                continue
            zf.extract(name, data_dir)

    # Reload database
    service.face_db._faces.clear()
    service.face_db._load()
    return {"status": "imported", "faces": service.face_db.list_faces()}


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
    """MJPEG stream for browsers."""

    async def generate():
        try:
            while True:
                jpg = service.streamer.get_jpeg() if service.streamer else None
                if jpg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                await asyncio.sleep(0.1)  # ~10fps matches inference rate
        finally:
            if service.streamer and not service._ws_clients:
                service.streamer._has_clients = False

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
