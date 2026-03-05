# Clawd Reachy Mini

Voice interface that connects a Reachy Mini robot to OpenClaw over WebSocket.

[![CI](https://github.com/ArturSkowronski/clawd-reachy-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/clawd-reachy-mini/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Reachy Mini](https://img.shields.io/badge/robot-Reachy%20Mini-orange.svg)](https://www.pollen-robotics.com/reachy-mini/)

<table style="border: none;">
<tr style="border: none;">
<td style="width: 42%; vertical-align: middle; border: none;">
  <img src="media/cover.png" alt="Reachy Mini" />
</td>
<td style="vertical-align: middle; border: none;">
<b>Quickstart</b>
<pre><code>uv sync --extra dev --extra audio
uv run clawd-reachy --gateway-host 127.0.0.1</code></pre>
<b>Standalone demo</b>
<pre><code>uv sync --extra dev --extra audio
uv run clawd-reachy --standalone</code></pre>
</td>
</tr>
</table>

This project runs a conversation loop on a machine connected to Reachy Mini:

1. capture microphone audio
2. transcribe speech (Whisper, Faster-Whisper, OpenAI, SenseVoice, ...)
3. send text to OpenClaw Gateway
4. receive AI response
5. speak response (ElevenLabs, Kokoro, Piper, macOS say, ...) and animate the robot

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  ClawdApp (app.py)                                              │
│  ├── MotionPlugin     — emotions, head tracking, idle anims     │
│  ├── FaceTrackerPlugin — MediaPipe face detection → HeadTarget  │
│  └── ConversationPlugin — STT → Gateway → TTS conversation loop │
│                                                                 │
│  Shared state:                                                  │
│  • HeadTargetBus  — fuses face/DOA/neutral head targets         │
│  • EmotionMapper  — 14 emotions, queue with debounce            │
│  • HeadWobbler    — speech-driven head micro-movements          │
└─────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   Reachy Mini SDK              OpenClaw Gateway
   (head, antennas)          (WebSocket, port 18790)
```

Data flow:

```text
Microphone → STT → Gateway (WebSocket) → AI response
AI response → sentence split → TTS → Speaker
            → EmotionMapper → MotionPlugin → Robot head/antennas
Camera → MediaPipe → HeadTarget → HeadTargetBus → Robot head
TTS audio → HeadWobbler → speech roll/pitch/yaw → Robot head
```

## Quickstart

```bash
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
uv sync --extra dev --extra audio
uv run clawd-reachy --gateway-host 127.0.0.1
```

Standalone mode (no gateway, echoes what it heard):

```bash
uv run clawd-reachy --standalone
```

Robot demo mode:

```bash
uv run clawd-reachy --demo
```

## Running as a Reachy Mini App

This project can run in two ways:

### Direct (development / standalone)

```bash
uv run clawd-reachy --gateway-host 192.168.1.100
```

The app manages the Reachy Mini connection itself.

### Via Reachy Mini Daemon (production)

The project registers as a Reachy Mini app via the `reachy_mini_apps` entry point. Install it into the daemon's environment:

```bash
pip install /path/to/clawd-reachy-mini
```

Then start via the daemon API:

```bash
# List available apps
curl http://localhost:8000/apps/list-available

# Start
curl http://localhost:8000/apps/start-app/clawd_reachy_mini

# Stop
curl http://localhost:8000/apps/stop-current-app
```

Or run directly as a Reachy Mini app (daemon must be running):

```bash
python -m clawd_reachy_mini.reachy_app
```

In daemon mode, the Reachy Mini connection is managed by the daemon and passed to the app.

## Configuration

Configuration is layered (highest priority wins):

**CLI args > Environment variables > YAML config file > Defaults**

### YAML config file

Copy the example and edit:

```bash
cp clawd.example.yaml clawd.yaml
```

The app auto-detects config files in this order:
1. `./clawd.yaml` or `./clawd.yml` (current directory)
2. `~/.clawd-reachy-mini/config.yaml`

Or specify explicitly:

```bash
clawd-reachy --config /path/to/config.yaml
# or
export CLAWD_CONFIG=/path/to/config.yaml
```

Example `clawd.yaml`:

```yaml
gateway:
  host: 192.168.1.100
  port: 18790

stt:
  backend: faster-whisper
  whisper_model: small

tts:
  backend: kokoro
  kokoro_speaker_id: 50
  kokoro_speed: 1.0

behavior:
  wake_word: hey reachy
  play_emotions: true

vision:
  tracker: mediapipe
  camera_index: 0
```

See `clawd.example.yaml` for the full list of options.

### Environment variables

| Variable | Description |
|---|---|
| `OPENCLAW_HOST` | Gateway host (default: `127.0.0.1`) |
| `OPENCLAW_PORT` | Gateway port (default: `18790`) |
| `OPENCLAW_TOKEN` | Gateway auth token |
| `OPENCLAW_PATH` | WebSocket path (default: `/desktop-robot`) |
| `STT_BACKEND` | STT backend (auto-detected from registry) |
| `WHISPER_MODEL` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `TTS_BACKEND` | TTS backend (auto-detected from registry) |
| `WAKE_WORD` | Wake word to activate listening |
| `SPEECH_SERVICE_URL` | Remote speech service URL |
| `OPENCLAW_OPENAI_TOKEN` / `OPENAI_API_KEY` | OpenAI API key (for `--stt openai`) |
| `CLAWD_CONFIG` | Path to YAML config file |

Backend-specific env vars are auto-generated from each backend's `Settings` class (e.g. `KOKORO_SPEAKER_ID`, `KOKORO_SPEED`, `SENSEVOICE_LANGUAGE`).

ElevenLabs TTS:
- `REACHY_ELEVENLABS_API_KEY` or `ELEVENLABS_API_KEY` (required)
- `REACHY_ELEVENLABS_VOICE_ID` or `ELEVENLABS_VOICE_ID` (optional)
- `REACHY_ELEVENLABS_MODEL_ID` or `ELEVENLABS_MODEL_ID` (optional)
- `REACHY_ELEVENLABS_OUTPUT_FORMAT` or `ELEVENLABS_OUTPUT_FORMAT` (optional)

### CLI options

```
-c, --config          Path to YAML config file
-v, --verbose         Debug logging
--gateway-host        OpenClaw host (default: 127.0.0.1)
--gateway-port        OpenClaw port (default: 18790)
--gateway-token       Auth token
--reachy-mode         auto | localhost_only | network
--stt                 (choices auto-detected from registry)
--whisper-model       tiny | base | small | medium | large
--tts                 (choices auto-detected from registry)
--tts-voice           Voice ID (backend-specific)
--tts-model           Model path (for Piper)
--speech-url          Remote speech service URL
--audio-device        Input device name
--wake-word           Wake phrase
--no-emotions         Disable emotion animations
--no-idle             Disable idle animations
--no-barge-in         Disable barge-in
--no-face-tracking    Disable face tracking
--tracker-type        mediapipe | none
--camera-index        Camera device index
--standalone          Run without gateway
--demo                Run robot movement demo and exit
```

## STT/TTS Backends

Backends are discovered automatically via the `@register_tts` / `@register_stt` decorators in `backend_registry.py`.

### Built-in STT backends

| Name | Type | Description |
|---|---|---|
| `whisper` | Local | OpenAI Whisper (default) |
| `faster-whisper` | Local | CTranslate2-optimized Whisper |
| `openai` | Cloud | OpenAI Whisper API |
| `sensevoice` | Remote | SenseVoice via speech service |

### Built-in TTS backends

| Name | Type | Description |
|---|---|---|
| `elevenlabs` | Cloud | ElevenLabs API (default) |
| `kokoro` | Remote | Kokoro TTS via speech service (sherpa-onnx) |
| `macos-say` | Local | macOS built-in `say` command |
| `piper` | Local | Piper neural TTS |
| `none` | — | Dummy (prints text, no audio) |

### Adding a new backend

Create a class in `tts.py` or `stt.py` with the decorator — that's it:

```python
@register_tts("my-backend")
class MyTTS(TTSBackend):
    """My custom TTS backend."""

    # Optional: declare backend-specific config fields
    class Settings:
        api_key: str = ""
        voice_type: str = "default"

    def __init__(self, base_url="http://localhost:8000", api_key="", voice_type="default"):
        self._base_url = base_url
        self._api_key = api_key
        self._voice_type = voice_type

    async def synthesize(self, text: str) -> str:
        # Call your API, return path to temp audio file
        ...
```

This automatically provides:
- `--tts my-backend` CLI option
- `tts.my_backend_api_key` / `tts.my_backend_voice_type` in YAML config
- `MY_BACKEND_API_KEY` / `MY_BACKEND_VOICE_TYPE` environment variables
- `config.my_backend_api_key` / `config.my_backend_voice_type` config attributes

## Installation

### Prerequisites

- Python 3.10+
- Reachy Mini SDK (`reachy-mini`)
- `ffmpeg` (required for mp3→wav conversion before Reachy playback)
- macOS `afplay` is used as local playback fallback

### Install the main app

```bash
uv sync
```

Development install:

```bash
uv sync --extra dev
```

### Optional extras

- local faster transcription: `uv sync --extra local-stt`
- OpenAI cloud transcription: `uv sync --extra cloud-stt`
- local mic deps: `uv sync --extra audio`
- Reachy vision extras: `uv sync --extra vision`
- MediaPipe face tracking: `uv sync --extra mediapipe-vision`

## OpenClaw Skill (`action-skill/`)

The action skill provides tool wrappers for robot control:

- connect/disconnect
- head movement
- antenna movement
- emotions and dance
- image capture
- robot speech
- status checks

Skill docs: `action-skill/SKILL.md`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv tool run ruff check .
```

Action skill tests:

```bash
cd action-skill
uv sync --extra dev
uv run pytest
```

## Key Files

| File | Purpose |
|---|---|
| `src/clawd_reachy_mini/main.py` | CLI entrypoint |
| `src/clawd_reachy_mini/app.py` | ClawdApp orchestrator |
| `src/clawd_reachy_mini/reachy_app.py` | Reachy Mini daemon app adapter |
| `src/clawd_reachy_mini/gateway.py` | OpenClaw WebSocket protocol |
| `src/clawd_reachy_mini/backend_registry.py` | Auto-discovery registry for STT/TTS backends |
| `src/clawd_reachy_mini/stt.py` | STT backend implementations |
| `src/clawd_reachy_mini/tts.py` | TTS backend implementations |
| `src/clawd_reachy_mini/plugins/` | Motion, conversation, face tracker plugins |
| `src/clawd_reachy_mini/motion/` | EmotionMapper, HeadTargetBus, HeadWobbler |
| `src/clawd_reachy_mini/vision/` | MediaPipe face tracker |
| `src/clawd_reachy_mini/config.py` | Configuration (YAML + env + defaults) |
| `clawd.example.yaml` | Example configuration file |
| `action-skill/` | OpenClaw skill package |
