# Hardware Inventory — Jetson Orin Nano

## Compute Platform

| Item | Value |
|------|-------|
| Board | NVIDIA Jetson Orin Nano (Engineering Reference Developer Kit Super) |
| JetPack | R36.5.0 (released 2026-01-16) |
| CUDA | 12.6 |
| Driver | 540.5.0 |
| Python | 3.10.12 (system) |
| Docker | 29.3.1 |
| Hostname | jorinn |

## Connected Devices

### USB Devices (lsusb output)

```
Bus 001 Device 006: ID 046d:082d Logitech, Inc. HD Pro Webcam C920
Bus 001 Device 003: ID 13d3:3549 IMC Networks Bluetooth Radio
Bus 002 Device 002: ID 0bda:0489 Realtek Semiconductor Corp. 4-Port USB 3.0 Hub
Bus 001 Device 002: ID 0bda:5489 Realtek Semiconductor Corp. 4-Port USB 2.0 Hub
```

### Logitech HD Pro Webcam C920

| Property | Value |
|----------|-------|
| Vendor:Product | `046d:082d` |
| Video device | `/dev/video0` (metadata on `/dev/video1`) |
| Resolution | Up to 1080p30 / 720p60 |
| USB audio | Card 2 — stereo mic, 32kHz |
| Group | `video` (anand already member) |

**Useful commands:**
```bash
# List supported video formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Capture test frame
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 /tmp/test_frame.jpg

# Record 5 seconds
ffmpeg -f v4l2 -i /dev/video0 -t 5 /tmp/test.mp4
```

### Bluetooth Radio (IMC Networks)

| Property | Value |
|----------|-------|
| Vendor:Product | `13d3:3549` |
| Controller MAC | `58:02:05:DD:F9:D8` |
| Status | Powered ON, not discoverable |
| Paired devices | None (as of 2026-04-06) |

**Pairing instructions**: See `docs/AUDIO_BLUETOOTH.md`

### Bluetooth Speakerphone — EMEET OfficeCore M0 Plus

| Property | Value |
|----------|-------|
| Model | EMEET OfficeCore M0 Plus |
| MAC | `64:31:39:07:DE:9B` |
| Status | Paired 2026-04-06 |
| PulseAudio sink | `bluez_sink.64_31_39_07_DE_9B.a2dp_sink` (when connected) |
| PulseAudio source | `bluez_source.64_31_39_07_DE_9B.headset_head_unit` (when connected) |

## Audio Cards (ALSA)

```
Card 0: NVIDIA Jetson Orin Nano HDA     — HDMI audio outputs (4 channels)
Card 1: NVIDIA Jetson Orin Nano APE     — Internal audio engine (ADMAIF)
Card 2: HD Pro Webcam C920              — USB Audio (mic only, 32kHz stereo)
[BT card will appear here when paired]
```

**PulseAudio sources** (current):
```
alsa_input.platform-sound.analog-stereo          (analog in, unused)
alsa_input.usb-046d_HD_Pro_Webcam_C920_...       (C920 mic — use this for now)
```

## User Permissions

User `anand` (uid=1000) is in groups:
- `video` — access to `/dev/video0`, `/dev/video1`
- `audio` — access to `/dev/snd/*`
- `render` — access to GPU rendering
- `docker` — run containers without sudo
- `gpio`, `i2c` — hardware I/O
- `sudo` — administrative commands

No additional permission setup needed.

## Software Stack

| Tool | Version | Location |
|------|---------|---------|
| Docker | 29.3.1 | `/usr/bin/docker` |
| nvidia-smi | 540.5.0 | `/usr/sbin/nvidia-smi` |
| Python | 3.10.12 | `/usr/bin/python3` |
| pip | 22.0.2 | system |
| uv | (install if needed) | `pip3 install uv` |
| Ollama | (not yet installed) | `curl -fsSL https://ollama.com/install.sh | sh` |

## GPIO / I2C Expansion (Future)

The Jetson has GPIO headers and I2C buses. The user `anand` is in the `gpio` and `i2c` groups. Future extensions:
- Servo control for camera pan/tilt
- LED status indicators
- Sensor integration (distance, temperature, etc.)

## Power Notes

For consistent AI inference latency, lock GPU/CPU clocks:
```bash
cd /home/anand/claude-stuff/suharvest/jetson-local-voice
sudo bash setup-performance.sh
```

This enables MAXN power mode and disables dynamic frequency scaling. Run once after each reboot (or add to `/etc/rc.local`).

## Incompatible Hardware Note

**Reachy Mini CM4 WiFi** — The reachy-claw software was designed for a different Reachy Mini variant using the Pollen Robotics SDK. The CM4 WiFi version uses a different control interface and is currently not supported. The `standalone_mode: true` flag bypasses all robot SDK calls so voice/vision can run without it.
