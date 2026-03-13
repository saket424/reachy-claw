#!/usr/bin/env bash
# Install Reachy kiosk auto-start on a Jetson/Ubuntu desktop.
# Usage: bash deploy/jetson/kiosk/install.sh [user@host]
#   Without arguments: install locally
#   With argument:     install on remote host via SSH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${1:-}"

install_local() {
    cp "$SCRIPT_DIR/reachy-kiosk.sh" ~/reachy-kiosk.sh
    chmod +x ~/reachy-kiosk.sh
    mkdir -p ~/.config/autostart
    sed "s|/home/recomputer|$HOME|g" "$SCRIPT_DIR/reachy-kiosk.desktop" \
        > ~/.config/autostart/reachy-kiosk.desktop
    echo "Installed: ~/reachy-kiosk.sh + ~/.config/autostart/reachy-kiosk.desktop"
}

install_remote() {
    scp "$SCRIPT_DIR/reachy-kiosk.sh" "$REMOTE:~/reachy-kiosk.sh"
    ssh "$REMOTE" "chmod +x ~/reachy-kiosk.sh && mkdir -p ~/.config/autostart"
    # Get remote home dir and adjust .desktop file
    REMOTE_HOME=$(ssh "$REMOTE" 'echo $HOME')
    sed "s|/home/recomputer|$REMOTE_HOME|g" "$SCRIPT_DIR/reachy-kiosk.desktop" \
        | ssh "$REMOTE" "cat > ~/.config/autostart/reachy-kiosk.desktop"
    echo "Installed on $REMOTE"
}

if [ -n "$REMOTE" ]; then
    install_remote
else
    install_local
fi
