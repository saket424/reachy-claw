# CLAUDE.md — reachy-claw

Plugin-based conversational AI + vision + motion controller for Reachy Mini, running on a **Jetson Orin Nano** (JetPack R36.5.0, CUDA 12.6).

## Repository Layout

```
reachy-claw/
├── src/reachy_claw/         # Main package
│   ├── plugins/             # Plugin implementations
│   ├── tts.py / stt.py      # Backend registries
│   ├── audio.py             # Mic capture + BG reader
│   ├── llm.py               # Ollama / OpenClaw LLM client
│   └── app.py               # App lifecycle, plugin loader
├── deploy/jetson/           # Jetson Orin Nano deployment
│   ├── Dockerfile.reachy-claw           # aarch64 image (no silero-vad/mediapipe)
│   ├── docker-compose.standalone.yml   # Standalone (no robot) compose
│   ├── reachy-claw.standalone.yaml     # Config for Jetson standalone demo
│   └── scripts/                        # BT auto-connect, systemd service
├── docs/                    # Architecture, hardware, guides
└── scripts/                 # Dev/test scripts
```

## Hardware (Jetson Orin Nano — hostname: jorinn)

| Device | Details | Notes |
|--------|---------|-------|
| Jetson Orin Nano | JetPack R36.5.0, CUDA 12.6, 8GB | — |
| Logitech C920 webcam | USB, video + mic | `/dev/video0` |
| EMEET OfficeCore M0 Plus | BT speakerphone | MAC `64:31:39:07:DE:9B` |
| Ollama server | External, gemma4:26b | `http://192.168.64.129:11434` |

User `anand` is in groups: `video`, `audio`, `render`, `docker`.

## Standalone Deploy (No Robot)

```bash
# From the reachy-claw repo root:
docker compose -f deploy/jetson/docker-compose.standalone.yml up -d
docker compose -f deploy/jetson/docker-compose.standalone.yml logs -f reachy-claw
```

This starts:
- `jetson-voice` — Kokoro TTS + Zipformer ASR (CUDA, port 8000)
- `reachy-claw` — conversation + dashboard (port 8640)

Config is mounted from `deploy/jetson/reachy-claw.standalone.yaml`.

## Plugin System

Every feature is a `Plugin` subclass in `src/reachy_claw/plugins/`:

```python
class MyPlugin(Plugin):
    name = "my_plugin"
    def setup(self) -> bool: ...       # sync, return False to skip
    async def start(self) -> None: ... # main loop
    async def stop(self) -> None: ...  # cleanup
```

Active plugins (registered in `app.py`):
- `conversation_plugin` — STT → VAD → LLM → TTS loop
- `face_tracker_plugin` — face detection → head tracking (disabled on aarch64: no mediapipe wheel)
- `motion_plugin` — emotions, idle animations (skipped without robot)
- `dashboard_plugin` — web UI at `:8640`
- `vision_client_plugin` — connects to remote vision-trt service

## Backends (swappable via YAML config)

| Type | Jetson standalone | Alternatives |
|------|-------------------|-------------|
| STT | `paraformer-streaming` → port 8000 | `whisper`, `faster-whisper`, `openai` |
| TTS | `kokoro` → port 8000 | `elevenlabs`, `piper`, `none` |
| LLM | `ollama` (gemma4:26b, external) | `gateway` (OpenClaw) |
| VAD | `energy` (no deps) | `silero` (needs PyTorch — too large for aarch64 Docker) |
| Vision | `mediapipe` (disabled aarch64) | `remote` (vision-trt Docker) |

### Config priority (highest wins)
```
env vars > runtime-overrides.yaml > reachy-claw.yaml > defaults
```

## Key Design Patterns

1. **Backend registry** (`backend_registry.py`): `@register_tts("name")` / `@register_stt("name")` — auto-discovered at import.
2. **Event bus** (`event_bus.py`): `app.event_bus.emit(event, data)` for plugin-to-plugin communication.
3. **Reconnecting proxies**: `_ReconnectingTTS` / `_ReconnectingSTT` — fallback immediately, retry in background.
4. **Config `Settings` inner class**: Add to a backend class to declare YAML-configurable fields.

## aarch64 / Jetson Constraints

- **silero-vad**: pulls PyTorch (~2GB) → times out in Docker on aarch64. Use `vad.backend: energy` instead.
- **mediapipe**: no PyPI wheel for manylinux aarch64. Face tracking disabled until built from source.
- **TTS audio playback**: uses `sd.OutputStream` chunk-writer (not `sd.play(blocking=True)`) to avoid portaudio double-free when barge-in calls stop from a different thread.
- **PulseAudio stream watchdog**: BG reader auto-reopens if RMS stays near zero for 30s (device goes SUSPENDED after long idle).

## Common Tasks

### Start / stop
```bash
docker compose -f deploy/jetson/docker-compose.standalone.yml up -d
docker compose -f deploy/jetson/docker-compose.standalone.yml down
```

### Test voice service
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from the Jetson", "sid": 52, "speed": 1.0}' \
  --output /tmp/test.wav && aplay /tmp/test.wav
```

### Run without Docker (dev)
```bash
uv run reachy-claw --config deploy/jetson/reachy-claw.standalone.yaml -v
```

### Check GPU memory
```bash
nvidia-smi
```

### BT speakerphone auto-connect (one-time setup)
```bash
sudo cp deploy/jetson/scripts/bt-connect.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/bt-connect.sh
sudo cp deploy/jetson/scripts/bt-speakerphone.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now bt-speakerphone.service
```

## Adding New Features

- **New TTS backend**: Add class to `src/reachy_claw/tts.py` with `@register_tts("name")`, add `class Settings` for config fields.
- **New STT backend**: Same pattern in `stt.py`.
- **New plugin**: Create `plugins/my_plugin.py`, extend `Plugin`, register in `app.py`.

See `docs/PLUGIN_DEV.md` for full plugin development guide.
See `docs/ARCHITECTURE.md` for system-level design.

## Notes on Reachy Mini CM4 WiFi Incompatibility

The codebase targets Reachy Mini (Hugging Face / Pollen Robotics SDK). The CM4 WiFi variant uses a different control protocol. Until a compatible driver exists:
- `standalone_mode: false` + `behavior.play_emotions: false` + `plugins.motion: false`
- All voice + vision features work without the robot connected
