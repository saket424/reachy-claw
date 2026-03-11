#!/usr/bin/env bash
# Initialize OpenClaw config and install desktop-robot extension dependencies.
# Run once after first `docker compose up`, or after updating the extension.
set -euo pipefail

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

CONTAINER="${1:-openclaw-gateway}"

# ── Wait for gateway to be healthy ───────────────────────────────────
info "Waiting for $CONTAINER to be healthy..."
for i in $(seq 1 30); do
    if docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null | grep -q healthy; then
        break
    fi
    sleep 2
done

if ! docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null | grep -q healthy; then
    err "$CONTAINER is not healthy"
    exit 1
fi
ok "Gateway is healthy"

# ── Install / upgrade desktop-robot extension via npm ─────────────────
OPENCLAW_BIN=$(docker exec "$CONTAINER" sh -c 'command -v openclaw 2>/dev/null || find /app -name openclaw -path "*/node_modules/.bin/*" 2>/dev/null | head -1')
if [ -z "$OPENCLAW_BIN" ]; then
    err "Cannot find openclaw binary in container"
    exit 1
fi
info "Using openclaw at: $OPENCLAW_BIN"

# Remove old extension if present (supports upgrade)
if docker exec "$CONTAINER" test -d /home/node/.openclaw/extensions/desktop-robot 2>/dev/null; then
    info "Removing existing extension for upgrade..."
    docker exec -u root "$CONTAINER" rm -rf /home/node/.openclaw/extensions/desktop-robot
fi

info "Installing @seeed-studio/openclaw-reachy extension..."
docker exec "$CONTAINER" "$OPENCLAW_BIN" plugins install @seeed-studio/openclaw-reachy@latest 2>&1 | tail -10
ok "Extension installed"

# ── Initialize base config if needed ─────────────────────────────────
info "Checking openclaw.json..."
docker exec "$CONTAINER" node -e '
const fs = require("fs");
const p = "/home/node/.openclaw/openclaw.json";
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync(p, "utf8")); } catch {}

let changed = false;

// Ensure gateway settings
if (!cfg.gateway) cfg.gateway = {};
if (!cfg.gateway.controlUi) cfg.gateway.controlUi = {};
if (!cfg.gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback) {
    cfg.gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback = true;
    changed = true;
}

// Ensure tools config
const wantTools = {
    profile: "coding",
    allow: ["exec", "bash"],
    deny: [],
    exec: { host: "gateway", security: "full", ask: "off" }
};
if (JSON.stringify(cfg.tools) !== JSON.stringify(wantTools)) {
    cfg.tools = wantTools;
    changed = true;
}

// Ensure commands config
if (!cfg.commands) cfg.commands = {};
if (cfg.commands.native !== "auto") { cfg.commands.native = "auto"; changed = true; }
if (cfg.commands.nativeSkills !== "auto") { cfg.commands.nativeSkills = "auto"; changed = true; }

// Enable desktop-robot plugin
if (!cfg.plugins) cfg.plugins = {};
if (!cfg.plugins.entries) cfg.plugins.entries = {};
if (!cfg.plugins.entries["desktop-robot"]) {
    cfg.plugins.entries["desktop-robot"] = { enabled: true };
    changed = true;
}

if (changed) {
    fs.writeFileSync(p, JSON.stringify(cfg, null, 2));
    console.log("Config updated");
} else {
    console.log("Config already set");
}
'
ok "Config initialized"

# ── Restart to load extension ────────────────────────────────────────
info "Restarting gateway to load extension..."
docker restart "$CONTAINER"
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:18789/healthz >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
ok "Gateway restarted and healthy"

echo ""
echo "=== OpenClaw gateway ready ==="
echo "  Gateway:        http://localhost:18789"
echo "  Desktop-robot:  ws://localhost:18790/desktop-robot"
echo ""
echo "To configure LLM provider, run:"
echo "  ./configure-llm.sh <provider> <api-key>"
echo ""
