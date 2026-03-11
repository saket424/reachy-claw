"""Environment-based configuration for vision-trt service."""

import os


class VisionConfig:
    """Configuration loaded from environment variables."""

    # Shared memory
    SHM_FRAME_PATH: str = os.getenv("SHM_FRAME_PATH", "/dev/shm/vision_frame")

    # ZMQ publisher
    ZMQ_PUB_PORT: int = int(os.getenv("ZMQ_PUB_PORT", "8631"))

    # HTTP/WebSocket server
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8630"))

    # RTSP/WebRTC video stream
    STREAM_PORT: int = int(os.getenv("STREAM_PORT", "8632"))

    # Model paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/app/models")
    ENGINE_DIR: str = os.getenv("ENGINE_DIR", "/app/engines")

    # Face database
    DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")

    # Inference settings
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
    RECOGNITION_THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.4"))
    INPUT_WIDTH: int = int(os.getenv("INPUT_WIDTH", "640"))
    INPUT_HEIGHT: int = int(os.getenv("INPUT_HEIGHT", "640"))

    # Emotion smoothing
    EMOTION_WINDOW_SIZE: int = int(os.getenv("EMOTION_WINDOW_SIZE", "5"))

    # Performance
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))


config = VisionConfig()
