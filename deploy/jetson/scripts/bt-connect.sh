#!/bin/bash
# Auto-connect EMEET OfficeCore M0 Plus on boot
# MAC: 64:31:39:07:DE:9B
# Install: sudo cp scripts/bt-connect.sh /usr/local/bin/bt-connect.sh
#          sudo chmod +x /usr/local/bin/bt-connect.sh
#          sudo cp scripts/bt-speakerphone.service /etc/systemd/system/
#          sudo systemctl daemon-reload && sudo systemctl enable --now bt-speakerphone.service

DEVICE_MAC="64:31:39:07:DE:9B"
DEVICE_NAME="EMEET OfficeCore M0 Plus"
MAX_ATTEMPTS=5

echo "Waiting for Bluetooth stack..."
sleep 8

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "Attempt $i/$MAX_ATTEMPTS: connecting $DEVICE_NAME..."
    if bluetoothctl connect "$DEVICE_MAC" 2>&1 | grep -q "Connection successful"; then
        echo "Connected to $DEVICE_NAME"
        # Set as default PulseAudio devices
        sleep 2
        SINK=$(pactl list sinks short | grep "${DEVICE_MAC//:/_}" | awk '{print $2}')
        SOURCE=$(pactl list sources short | grep "${DEVICE_MAC//:/_}" | awk '{print $2}')
        [ -n "$SINK" ]   && pactl set-default-sink   "$SINK"   && echo "Default sink: $SINK"
        [ -n "$SOURCE" ] && pactl set-default-source "$SOURCE" && echo "Default source: $SOURCE"
        exit 0
    fi
    sleep 5
done

echo "Could not connect to $DEVICE_NAME after $MAX_ATTEMPTS attempts"
exit 1
