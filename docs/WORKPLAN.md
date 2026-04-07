# Work Plan — Voice + Vision on Jetson Orin Nano

**Target**: Get local TTS, STT, and video face tracking running independently, without requiring the Reachy Mini robot.

**Last updated**: 2026-04-06  
**Status**: Phase 0 complete (documentation done). Starting Phase 1.

---

## Phase 1: Voice Service (jetson-local-voice Docker)

**Goal**: `curl http://localhost:8000/health` returns `{"asr": true, "tts": true, "streaming_asr": true}`

### 1.1 — Pull and start the Docker image (English mode)

```bash
docker run -d --name jetson-voice \
  --runtime nvidia --ipc host \
  -p 8000:8000 \
  -e LANGUAGE_MODE=en \
  -v jetson-voice-models:/opt/models \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-nvidia-libs:/host-libs:/host-cuda \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/jetson-voice:v3.0-slim
```

Wait ~60 seconds for model download, then:
```bash
curl http://localhost:8000/health
# expected: {"asr": false, "tts": true, "streaming_asr": true}
# (offline SenseVoice ASR takes longer to load — streaming ASR available immediately)
```

### 1.2 — Test TTS

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am your Jetson assistant", "sid": 52, "speed": 1.0}' \
  --output /tmp/test_tts.wav

# Play back (use HDMI or C920 built-in audio for now):
aplay /tmp/test_tts.wav
```

### 1.3 — Test Streaming ASR

```bash
# Record 5 seconds via C920 mic
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test_asr.wav

# Transcribe
curl -X POST http://localhost:8000/asr \
  -F "file=@/tmp/test_asr.wav" \
  -F "language=en"
```

### 1.4 — Lock GPU/CPU clocks for consistent latency (run once after boot)

```bash
cd /home/anand/claude-stuff/suharvest/jetson-local-voice
sudo bash setup-performance.sh
```

**Phase 1 complete when**: TTS returns audio and ASR transcribes recorded speech correctly.

---

## Phase 2: Bluetooth Speakerphone

**Goal**: Bluetooth speakerphone appears as PulseAudio default sink (speaker) and source (mic).

### 2.1 — Scan and pair

```bash
bluetoothctl
# Inside bluetoothctl shell:
power on
agent on
scan on
# Wait for your device to appear, note its MAC address (e.g. AA:BB:CC:DD:EE:FF)
scan off
pair AA:BB:CC:DD:EE:FF
trust AA:BB:CC:DD:EE:FF
connect AA:BB:CC:DD:EE:FF
quit
```

### 2.2 — Verify PulseAudio sees it

```bash
pactl list sinks short     # should show bt_sink
pactl list sources short   # should show bt_source

# Set as default
pactl set-default-sink   <bt-sink-name>
pactl set-default-source <bt-source-name>
```

### 2.3 — Test audio round-trip through BT

```bash
# Play TTS through BT speaker
aplay /tmp/test_tts.wav

# Record from BT mic (5 seconds)
arecord -d 5 /tmp/bt_test.wav
curl -X POST http://localhost:8000/asr -F "file=@/tmp/bt_test.wav"
```

### 2.4 — Make BT connection persist across reboots

See `docs/AUDIO_BLUETOOTH.md` for the systemd auto-connect service.

**Phase 2 complete when**: Speaking into BT mic and hearing TTS through BT speaker works end-to-end.

---

## Phase 3: Install Ollama + LLM

**Goal**: `curl http://localhost:11434/api/tags` returns a model list including `qwen3.5:2b`.

### 3.1 — Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3.2 — Pull a model

```bash
ollama pull qwen3.5:2b
# Test
ollama run qwen3.5:2b "Say hello in one sentence"
```

### 3.3 — Verify service is running

```bash
curl http://localhost:11434/api/tags
systemctl status ollama   # should show active (running)
```

**Phase 3 complete when**: `ollama run qwen3.5:2b` responds in the terminal.

---

## Phase 4: reachy-claw in Standalone Mode

**Goal**: reachy-claw starts, connects to jetson-local-voice, listens for speech, and responds via TTS.

### 4.1 — Install dependencies

```bash
cd /home/anand/claude-stuff/suharvest/reachy-claw
pip3 install uv   # if not already installed
uv sync           # installs all dependencies from pyproject.toml
```

### 4.2 — Install Silero VAD model (one-time)

```bash
# silero-vad downloads automatically on first run
# But pre-fetch to avoid delay:
python3 -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)"
```

### 4.3 — Run in standalone mode

```bash
cd /home/anand/claude-stuff/suharvest/reachy-claw
uv run python -m reachy_claw --config ../reachy-claw.yaml
```

Expected startup output:
```
INFO  Loading config from ../reachy-claw.yaml
INFO  Using STT backend: paraformer-streaming
INFO  Kokoro TTS: streaming endpoint available
INFO  Using TTS backend: kokoro
INFO  Starting ConversationPlugin
INFO  Starting FaceTrackerPlugin
INFO  Dashboard available at http://0.0.0.0:8640
```

### 4.4 — Open dashboard

```bash
# In browser: http://localhost:8640
# Shows: conversation history, audio level, face detection status
```

**Phase 4 complete when**: You can speak to the system and hear it respond in real time.

---

## Phase 5: Video Face Tracking

**Goal**: Camera detects faces and logs face position (even without robot head movement).

### 5.1 — Verify camera is accessible

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Opened:', cap.isOpened()); cap.release()"
```

### 5.2 — Test mediapipe standalone

```bash
python3 - <<'EOF'
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
cap = cv2.VideoCapture(0)
with mp_face.FaceDetection(min_detection_confidence=0.5) as detector:
    for _ in range(30):  # 30 frames
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if result.detections:
                print(f"Face detected: {len(result.detections)} face(s)")
cap.release()
EOF
```

### 5.3 — FaceTrackerPlugin in reachy-claw

The `face_tracker_plugin` runs automatically when `plugins.face_tracker: true` in `reachy-claw.yaml`.
It emits events on `event_bus` when a face is detected/lost.
The `motion_plugin` (disabled) would normally act on these events.
In standalone mode, you can add a custom plugin to log or display detections.

**Phase 5 complete when**: reachy-claw logs face detection events when a face appears in frame.

---

## Phase 6: Performance Tuning

### 6.1 — Measure end-to-end latency

```bash
# Time from end of speech to start of TTS audio:
# Rough target on Orin Nano:
#   STT:  ~50-80ms (streaming ASR endpoint detection)
#   LLM:  ~800ms-2s (qwen3.5:2b, first token)
#   TTS:  ~130ms (first audio chunk)
#   Total: ~1-2.5s
```

### 6.2 — Try a smaller/faster LLM

```bash
ollama pull qwen3.5:0.8b   # smallest, ~300ms response
# Update reachy-claw.yaml: llm.model: qwen3.5:0.8b
```

### 6.3 — Consider faster-whisper as STT fallback

If jetson-local-voice is unavailable and whisper fallback is slow, add faster-whisper:
```bash
pip3 install faster-whisper
# Update reachy-claw.yaml: stt.backend: faster-whisper
```

---

## Future Phases (Backlog)

| Phase | Goal |
|-------|------|
| 7 | Wake-word detection (porcupine or openwakeword) |
| 8 | VLM: visual context passed to LLM (describe what camera sees) |
| 9 | Knowledge base / RAG (ChromaDB or SQLite-VSS local) |
| 10 | Custom TTS voice (fine-tune or voice clone) |
| 11 | Multi-room audio (Snapcast or PulseAudio network) |
| 12 | Robot integration (once compatible hardware driver written) |
| 13 | Person re-identification across sessions (face database) |

---

## Troubleshooting Quick Reference

| Symptom | Fix |
|---------|-----|
| `curl http://localhost:8000/health` fails | `docker ps` — is jetson-voice running? `docker logs jetson-voice` |
| TTS returns silence / 0-byte WAV | Check `docker logs jetson-voice` for model load errors |
| STT transcribes garbage | Verify mic: `arecord -D plughw:2,0 -f S16_LE -r 16000 -d 3 /tmp/t.wav` then play it |
| `uv sync` fails | `pip3 install uv` first; or `pip3 install -r reachy-claw/requirements.txt` |
| Ollama not found | Re-run install script; `systemctl start ollama` |
| BT speaker not showing in PulseAudio | `pactl list sinks` — try `bluetoothctl connect <MAC>` again |
| Camera not opening | Check `ls -la /dev/video0` — user in `video` group? (`groups`) |
| GPU out of memory | `nvidia-smi` — kill other GPU processes; Kokoro + qwen3.5:2b together use ~3-4GB |
