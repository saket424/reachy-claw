# System Architecture

## Overview

This project runs three independent services on the Jetson Orin Nano. Each can be started and tested in isolation.

```
┌─────────────────────────────────────────────────────────────────┐
│  Jetson Orin Nano (CUDA 12.6, 8GB RAM)                          │
│                                                                  │
│  ┌──────────────────────┐   ┌────────────────────────────────┐  │
│  │  jetson-local-voice  │   │        reachy-claw             │  │
│  │  (Docker :8000)      │   │  (uv Python process)           │  │
│  │                      │   │                                │  │
│  │  FastAPI             │   │  ConversationPlugin            │  │
│  │  ├── /health         │◄──│    STT: ParaformerStreamingSTT │  │
│  │  ├── WS /asr/stream  │   │    TTS: KokoroTTS              │  │
│  │  ├── POST /asr       │   │    VAD: SileroVAD              │  │
│  │  ├── POST /tts       │   │    LLM: OllamaBackend          │  │
│  │  └── POST /tts/stream│   │                                │  │
│  │                      │   │  FaceTrackerPlugin             │  │
│  │  sherpa-onnx (CUDA)  │   │    Vision: MediapipeTracker    │  │
│  │  ├── Zipformer ASR   │   │    Camera: OpenCV /dev/video0  │  │
│  │  ├── Kokoro TTS      │   │                                │  │
│  │  └── SenseVoice      │   │  DashboardPlugin :8640         │  │
│  └──────────────────────┘   └────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────┐   ┌────────────────────────────────┐  │
│  │  Ollama (REMOTE)     │   │  PulseAudio                    │  │
│  │  192.168.64.129:11434│   │  ├── sink: BT speakerphone     │  │
│  └──────────────────────┘   │  └── source: C920 mic / BT     │  │
│                              └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Hardware:
  /dev/video0  ─── Logitech C920 (1080p, USB)
  BT Radio     ─── Speakerphone (pair once, auto-reconnects)
```

## Data Flow: Conversation Loop

```
Microphone (BT/C920)
    │  raw PCM 16kHz
    ▼
VAD (Silero)
    │  speech detected
    ▼
ParaformerStreamingSTT (WebSocket → localhost:8000)
    │  transcribed text
    ▼
OllamaLLM (HTTP → localhost:11434)
    │  response text
    ▼
KokoroTTS (HTTP → localhost:8000)
    │  PCM audio chunks
    ▼
Speaker (BT/HDMI)
```

## Data Flow: Vision / Face Tracking

```
Logitech C920 (/dev/video0)
    │  BGR frames via OpenCV
    ▼
MediapipeTracker (CPU, in-process)
    │  face bounding box + landmarks
    ▼
FaceTrackerPlugin
    │  normalized (x, y) offset from center
    ▼
EventBus.emit("face_detected", {...})
    │
    ▼
[MotionPlugin — disabled without robot]
[DashboardPlugin — display face overlay]
```

## Conversation Pipeline (Internal Detail)

`ConversationPlugin` runs **4 concurrent async tasks** with queued handoffs:

```
_audio_loop                _sentence_accumulator        _tts_worker           _output_pipeline
───────────────            ─────────────────────        ───────────           ────────────────
Mic capture                Stream LLM response          TTS synthesis         Interruptible playback
  → VAD detect             Split on sentence end        Kokoro → audio        Barge-in detection
  → STT streaming          Buffer sentences             Yield PCM chunks      BT/speaker output
  → emit asr_partial/final → _sentence_queue ──────────► _audio_queue ───────►
  → send text to LLM ──────────────────────────────────────────────────────────────────────────►
```

The `_output_pipeline` monitors a barge-in detector in parallel. When speech is detected during
TTS playback (above `barge_in_energy_threshold`), it cancels the current audio and resumes listening.

## Component Responsibilities

### `jetson-local-voice` (Docker)

| Component | File | Role |
|-----------|------|------|
| FastAPI server | `app/main.py` | Routes, startup |
| Streaming ASR | `app/streaming_asr_service.py` | Zipformer/Paraformer WebSocket |
| Offline ASR | `app/asr_service.py` | SenseVoice HTTP |
| TTS | `app/tts_service.py` | Kokoro/Matcha batch + streaming |
| Model download | `app/model_downloader.py` | Auto-fetch on first start |

### `reachy-claw` (Python process)

| Component | File | Role |
|-----------|------|------|
| App lifecycle | `src/reachy_claw/app.py` | Plugin registry + asyncio gather |
| Config | `src/reachy_claw/config.py` | YAML → dataclass, env overrides |
| Plugin base | `src/reachy_claw/plugin.py` | setup/start/stop lifecycle |
| STT backends | `src/reachy_claw/stt.py` | Whisper, Paraformer, SenseVoice, OpenAI |
| TTS backends | `src/reachy_claw/tts.py` | Kokoro, Piper, ElevenLabs, Say, NoopTTS |
| Audio I/O | `src/reachy_claw/audio.py` | Record + playback via sounddevice/pyaudio |
| VAD | `src/reachy_claw/vad.py` | Silero + energy-based |
| LLM | `src/reachy_claw/llm.py` | Ollama + gateway (OpenClaw) |
| Event bus | `src/reachy_claw/event_bus.py` | Async pub/sub between plugins |
| Backend registry | `src/reachy_claw/backend_registry.py` | Decorator-based backend discovery |
| Vision trackers | `src/reachy_claw/vision/` | Mediapipe + GStreamer + remote |
| Conversation | `src/reachy_claw/plugins/conversation_plugin.py` | Full STT→LLM→TTS loop |
| Face tracker | `src/reachy_claw/plugins/face_tracker_plugin.py` | Vision → head control |
| Dashboard | `src/reachy_claw/plugins/dashboard_plugin.py` | WebSocket UI |

## Port Map

| Port | Service | Protocol | Status |
|------|---------|---------|--------|
| 8000 | jetson-local-voice ASR/TTS | HTTP + WebSocket | Phase 1 |
| 8630 | vision-trt HTTP API + MJPEG | HTTP | Phase 5+ |
| 8631 | vision-trt ZMQ PUB (face detections) | ZMQ | Phase 5+ |
| 8632 | vision-trt MJPEG stream | HTTP | Phase 5+ |
| 8640 | reachy-claw dashboard | HTTP + WebSocket | Phase 4 |
| 11434 | Ollama LLM | HTTP | Phase 3 |
| 38001 | reachy-daemon FastAPI | HTTP | (robot only) |

## Configuration Layers

Config is loaded in this priority order (highest wins):

```
1. Environment variables     (STT_BACKEND=whisper, TTS_BACKEND=none, ...)
2. runtime-overrides.yaml    (~/.reachy-claw/runtime-overrides.yaml)
3. reachy-claw.yaml          (this directory — our main config)
4. Code defaults             (Config dataclass in config.py)
```

## Extension Points

| Goal | Where to add |
|------|-------------|
| New TTS voice/backend | `reachy-claw/src/reachy_claw/tts.py` + `@register_tts("name")` |
| New STT engine | `reachy-claw/src/reachy_claw/stt.py` + `@register_stt("name")` |
| New behavior/feature | New `Plugin` subclass in `plugins/`, registered in `app.py` |
| New ASR model in service | `jetson-local-voice/app/streaming_asr_service.py` |
| New TTS model in service | `jetson-local-voice/app/tts_service.py` |
| New HTTP endpoint | `jetson-local-voice/app/main.py` |
