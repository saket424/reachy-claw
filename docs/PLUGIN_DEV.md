# Plugin Development Guide

This guide explains how to add new features to `reachy-claw` using the plugin system.

## Plugin Lifecycle

Every plugin extends `Plugin` from `reachy_claw.plugin`:

```python
from reachy_claw.plugin import Plugin

class MyPlugin(Plugin):
    name = "my_plugin"          # must be unique

    def __init__(self, app):
        super().__init__(app)
        # store config references here

    def setup(self) -> bool:
        """Called synchronously before the asyncio loop starts.
        Return False to skip this plugin gracefully (e.g., missing hardware)."""
        try:
            import some_optional_dep
        except ImportError:
            return False         # skip — dependency not installed
        return True

    async def start(self) -> None:
        """Main async entry. Runs until stop() sets self._running = False."""
        self._running = True
        while self._running:
            # do something
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Called during shutdown. Clean up resources."""
        self._running = False
```

## Registering Your Plugin

In `src/reachy_claw/app.py`, add your plugin to the list:

```python
from reachy_claw.plugins.my_plugin import MyPlugin

class ReachyClawApp:
    def _build_plugins(self) -> list[Plugin]:
        plugins = [
            ConversationPlugin(self),
            FaceTrackerPlugin(self),
            MyPlugin(self),          # ← add here
            DashboardPlugin(self),
        ]
        return [p for p in plugins if p.setup()]
```

## Accessing App Context

The `app` object passed to `__init__` gives access to:

```python
self.app.config          # Config dataclass (all settings)
self.app.event_bus       # EventBus for pub/sub
self.app.tts             # TTSBackend instance
self.app.stt             # STTBackend instance
self.app.llm             # LLMBackend instance (if available)
```

## Event Bus

Plugins communicate via the event bus without direct coupling:

```python
# Emit an event (fire-and-forget)
await self.app.event_bus.emit("my_event", {"key": "value"})

# Subscribe to events
@self.app.event_bus.on("face_detected")
async def on_face(data: dict):
    x, y = data["x"], data["y"]
    print(f"Face at ({x:.2f}, {y:.2f})")
```

### Standard Events

| Event name | Emitter | Data |
|-----------|---------|------|
| `asr_partial` | `ConversationPlugin` | `{"text": str}` |
| `asr_final` | `ConversationPlugin` | `{"text": str}` |
| `llm_delta` | `ConversationPlugin` | `{"text": str}` |
| `llm_end` | `ConversationPlugin` | `{"text": str}` |
| `state_change` | `ConversationPlugin` | `{"state": str}` (listening/speaking/thinking) |
| `emotion` | `ConversationPlugin` / `VisionClientPlugin` | `{"emotion": str}` |
| `vision_faces` | `VisionClientPlugin` | `[{"bbox": ..., "identity": str, "emotion": str}]` |
| `smile_capture` | `VisionClientPlugin` | `{"name": str, "image": bytes}` |
| `observation` | `VisionClientPlugin` | `{"description": str}` |
| `barge_in` | `ConversationPlugin` | `{}` |

## Adding a New TTS Backend

Backends use a decorator registry — no central registration file needed:

```python
# In src/reachy_claw/tts.py
from reachy_claw.backend_registry import register_tts

@register_tts("my_tts")
class MyTTS(TTSBackend):

    class Settings:
        """Fields declared here become YAML-configurable automatically.
        
        In reachy-claw.yaml:
            tts:
              my_tts:
                my_param: value
        """
        my_param: str = "default"

    def __init__(self, my_param: str = "default"):
        self._my_param = my_param

    async def synthesize(self, text: str) -> str:
        # ... generate audio, write to temp WAV file
        return tmp_path   # caller deletes this file
```

Then in `reachy-claw.yaml`:
```yaml
tts:
  backend: my_tts
  my_tts:
    my_param: custom_value
```

## Adding a New STT Backend

Same pattern:

```python
from reachy_claw.backend_registry import register_stt

@register_stt("my_stt")
class MySTT(STTBackend):
    supports_streaming = True   # set if you implement start_stream/feed_chunk/finish_stream

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        ...

    def transcribe_file(self, path: Path) -> str:
        ...

    # Optional streaming interface:
    def start_stream(self, sample_rate: int = 16000) -> None: ...
    def feed_chunk(self, chunk: np.ndarray) -> PartialResult | None: ...
    def finish_stream(self) -> str: ...
```

## Adding a New LLM Backend

LLM backends are in `src/reachy_claw/llm.py`. See existing `OllamaBackend` and `GatewayBackend`.
The interface is:

```python
class LLMBackend(ABC):
    async def chat(self, message: str, history: list[dict]) -> str: ...
    async def stream_chat(self, message: str, history: list[dict]) -> AsyncIterator[str]: ...
```

## Testing Your Plugin

Use `tests/` directory with the existing pytest/conftest patterns:

```python
# tests/test_my_plugin.py
import pytest
from reachy_claw.plugins.my_plugin import MyPlugin

@pytest.fixture
def app(mock_app):   # mock_app from conftest.py
    return mock_app

@pytest.mark.asyncio
async def test_my_plugin_starts(app):
    plugin = MyPlugin(app)
    assert plugin.setup() is True
    # test your logic
```

Run tests:
```bash
cd reachy-claw
uv run pytest tests/test_my_plugin.py -v
```

## Example: Logging Plugin

A minimal real plugin that logs all events to a file:

```python
# src/reachy_claw/plugins/event_logger_plugin.py
import asyncio
import json
import logging
from pathlib import Path
from reachy_claw.plugin import Plugin

logger = logging.getLogger(__name__)

class EventLoggerPlugin(Plugin):
    name = "event_logger"

    def setup(self) -> bool:
        self._log_path = Path.home() / ".reachy-claw" / "events.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        return True

    async def start(self) -> None:
        self._running = True
        
        @self.app.event_bus.on("speech_ended")
        async def on_speech(data):
            self._write({"event": "speech_ended", "text": data.get("text", "")})

        @self.app.event_bus.on("llm_response")
        async def on_llm(data):
            self._write({"event": "llm_response", "text": data.get("text", "")})

        while self._running:
            await asyncio.sleep(1.0)

    def _write(self, data: dict) -> None:
        import time
        data["ts"] = time.time()
        with open(self._log_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    async def stop(self) -> None:
        self._running = False
        logger.info(f"Event log saved to {self._log_path}")
```

## Example: Vision-to-LLM Plugin (VLM)

See `config.py` — `enable_vlm` and `vlm_model` are already wired in. The VLM feature:
1. Captures a frame from the camera at conversation time
2. Sends it to Ollama vision model (e.g. `llava:7b`, `qwen2.5-vl:7b`)
3. Prepends the description to the LLM context

This is a "Phase 8" backlog item — the config keys exist, the plugin needs to be written.

## Config Discovery

To add a new top-level config section, add entries to `_YAML_FIELD_MAP` in `config.py`:

```python
# In config.py, _YAML_FIELD_MAP:
("my_section", "my_key"): "my_section_my_key",

# In Config dataclass:
my_section_my_key: str = "default"
```

Then in `reachy-claw.yaml`:
```yaml
my_section:
  my_key: custom_value
```
