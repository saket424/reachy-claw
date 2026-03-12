"""Environment-based configuration for vision-trt service."""

import os


class VisionConfig:
    """Configuration loaded from environment variables."""

    # Camera capture
    CAMERA_DEVICE: str = os.getenv("CAMERA_DEVICE", "/dev/video0")
    # Camera MJPEG resolution (set via V4L2 ioctl before GStreamer)
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    STREAM_WIDTH: int = int(os.getenv("STREAM_WIDTH", "640"))

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
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "10"))


config = VisionConfig()
