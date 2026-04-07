# Audio Setup — Bluetooth Speakerphone + Logitech C920

## Detected Audio Devices

| Card | Name | Notes |
|------|------|-------|
| Card 0 | NVIDIA Jetson Orin Nano HDA | HDMI audio outputs |
| Card 1 | NVIDIA Jetson Orin Nano APE | Internal audio engine |
| Card 2 | HD Pro Webcam C920 | USB audio — mic only (good for recording) |

PulseAudio sources currently:
- `alsa_output.platform-sound.analog-stereo` (analog out)
- `alsa_input.usb-046d_HD_Pro_Webcam_C920_7679427F-02.analog-stereo` (C920 mic)

## Bluetooth Device

**EMEET OfficeCore M0 Plus** — MAC: `64:31:39:07:DE:9B`

Already paired (2026-04-06). To reconnect manually: `bluetoothctl connect 64:31:39:07:DE:9B`

## Bluetooth Pairing (First-Time Setup — completed)

Your Jetson's BT controller: `58:02:05:DD:F9:D8` (hostname: `jorinn`), powered: yes.

### Step 1: Start interactive pairing

Run this in a terminal on the Jetson:

```bash
bluetoothctl
```

Inside the shell:

```
power on
agent on
default-agent
scan on
```

Wait for your speakerphone to appear in the scan output. It will show something like:
```
[NEW] Device 64:31:39:07:DE:9B JBL Flip / Jabra / [your device name]
```

```
scan off
pair 64:31:39:07:DE:9B
trust 64:31:39:07:DE:9B
connect 64:31:39:07:DE:9B
quit
```

Replace `64:31:39:07:DE:9B` with your device's actual MAC address.

### Step 2: Verify PulseAudio sees it

```bash
pactl list sinks short
# Should show a bluez sink like:
# bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink

pactl list sources short
# Should show:
# bluez_source.AA_BB_CC_DD_EE_FF.headset_head_unit
```

### Step 3: Set as default audio device

```bash
# Set speaker as default output
pactl set-default-sink bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink

# Set BT mic as default input
pactl set-default-source bluez_source.AA_BB_CC_DD_EE_FF.headset_head_unit
```

### Step 4: Test

```bash
# Play a WAV file through BT speaker
aplay /tmp/test_tts.wav

# Record 5s from BT mic and transcribe
arecord -d 5 /tmp/bt_test.wav
curl -X POST http://localhost:8000/asr -F "file=@/tmp/bt_test.wav"
```

## Auto-Reconnect on Boot

Create a small script to reconnect BT device after boot:

```bash
# /usr/local/bin/bt-connect.sh
#!/bin/bash
DEVICE_MAC="64:31:39:07:DE:9B"   # ← replace with your MAC
sleep 10   # wait for BT stack to initialize
bluetoothctl connect "$DEVICE_MAC"
```

```bash
sudo chmod +x /usr/local/bin/bt-connect.sh
```

Create a systemd service:

```ini
# /etc/systemd/system/bt-speakerphone.service
[Unit]
Description=Auto-connect Bluetooth speakerphone
After=bluetooth.service pulseaudio.service
Wants=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/bt-connect.sh
User=anand
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bt-speakerphone.service
sudo systemctl start bt-speakerphone.service
```

## Configuring reachy-claw to Use Bluetooth

In `reachy-claw.yaml`, leave `audio.device: null` to use PulseAudio's default device.
After setting the BT speaker/mic as default via `pactl`, reachy-claw will automatically use them.

To use a specific device by name:
```yaml
audio:
  device: "bluez_sink.AA_BB_CC_DD_EE_FF"   # replace with your device name
```

To find your device name:
```bash
python3 -c "import sounddevice as sd; print([d['name'] for d in sd.query_devices()])"
```

## Fallback: C920 Built-In Mic

If BT is unavailable, the Logitech C920 has a decent built-in stereo mic.

PulseAudio source name: `alsa_input.usb-046d_HD_Pro_Webcam_C920_7679427F-02.analog-stereo`

Set as default:
```bash
pactl set-default-source alsa_input.usb-046d_HD_Pro_Webcam_C920_7679427F-02.analog-stereo
```

ALSA direct access (bypassing PulseAudio, useful for testing):
```bash
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav
```

Note: PulseAudio and direct ALSA access conflict — use one or the other.

## Audio Format Requirements

The jetson-local-voice ASR service expects:
- **Format**: 16-bit signed PCM (S16_LE)
- **Sample rate**: 16000 Hz (16kHz)
- **Channels**: 1 (mono)

reachy-claw's audio.py handles format conversion automatically.

## Troubleshooting

| Issue | Command | Fix |
|-------|---------|-----|
| BT not connecting | `bluetoothctl info 64:31:39:07:DE:9B` | Is device trusted? Re-run `trust` + `connect` |
| PulseAudio crash | `systemctl --user restart pulseaudio` | Restart PA |
| No BT sink in PulseAudio | `pactl list cards` | Check if BT card loaded |
| Stuttering audio | `pactl set-sink-volume @DEFAULT_SINK@ 80%` | Lower volume to avoid clipping |
| C920 mic too quiet | `amixer -c 2 set Mic 80%` | Increase mic gain |
