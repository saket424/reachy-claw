#!/usr/bin/env bash
# One-click deploy: reachy-claw stack to Jetson.
#
# Modes (combinable):
#   ./deploy.sh                    # Ollama mode (default, no OpenClaw)
#   ./deploy.sh --openclaw         # OpenClaw mode (starts gateway container)
#   ./deploy.sh --vision           # Add vision-trt service
#   ./deploy.sh --openclaw --vision  # Both
#
# Prerequisites:
#   - Speech service already running on Jetson (port 8621)
#   - SSH access to Jetson (key-based auth recommended)
#   - Ollama mode: GPU runtime (nvidia) available on Jetson
#   - OpenClaw mode: configure LLM after deploy with configure-llm.sh
#
# Options:
#   --openclaw          Deploy with OpenClaw gateway
#   --vision            Deploy with vision-trt service
#   --setup-openclaw    Run OpenClaw first-time setup (extension install + config)
#   JETSON_HOST=x       Custom SSH host (default: recomputer)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Config ────────────────────────────────────────────────────────────
JETSON_HOST="${JETSON_HOST:-recomputer}"
JETSON_USER="${JETSON_USER:-recomputer}"
DEPLOY_DIR="${DEPLOY_DIR:-~/reachy-deploy}"
USE_OPENCLAW=false
USE_VISION=false
SETUP_OPENCLAW=false

for arg in "$@"; do
    case "$arg" in
        --openclaw) USE_OPENCLAW=true ;;
        --vision) USE_VISION=true ;;
        --setup-openclaw) USE_OPENCLAW=true; SETUP_OPENCLAW=true ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
die()   { err "$@"; exit 1; }

ssh_cmd() { ssh "$JETSON_USER@$JETSON_HOST" "$@"; }

# Build COMPOSE_PROFILES from flags
# Default (no --openclaw) activates ollama profile
if [ "$USE_OPENCLAW" = true ]; then
    PROFILES="openclaw"
else
    PROFILES="ollama"
fi
if [ "$USE_VISION" = true ]; then
    PROFILES="$PROFILES,vision"
fi

# Build docker compose flags
compose_profile_flags() {
    local flags=""
    if [ "$USE_OPENCLAW" = true ]; then
        flags="$flags --profile openclaw"
    else
        flags="$flags --profile ollama"
    fi
    if [ "$USE_VISION" = true ]; then
        flags="$flags --profile vision"
    fi
    echo "$flags"
}

# ── Pre-flight ────────────────────────────────────────────────────────
info "Checking SSH connection to $JETSON_USER@$JETSON_HOST..."
ssh_cmd "echo ok" >/dev/null 2>&1 || die "Cannot SSH to $JETSON_HOST"
ok "SSH connected"

MODES="reachy"
[ "$USE_OPENCLAW" = true ] && MODES="$MODES + openclaw"
[ "$USE_VISION" = true ] && MODES="$MODES + vision-trt"
info "Mode: $MODES"

# ── Sync deploy files ─────────────────────────────────────────────────
info "Syncing deploy files..."
ssh_cmd "mkdir -p $DEPLOY_DIR"
rsync -az \
    "$SCRIPT_DIR/reachy/docker-compose.yml" \
    "$SCRIPT_DIR/reachy/reachy-claw.jetson.yaml" \
    "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/"

# Sync vision-trt source if --vision
if [ "$USE_VISION" = true ]; then
    info "Syncing vision-trt source..."
    rsync -az --delete \
        "$SCRIPT_DIR/vision-trt/" \
        "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/vision-trt/"
    ok "vision-trt synced"
fi

# Sync OpenClaw helper scripts (always — they're small and useful later)
rsync -az \
    "$SCRIPT_DIR/openclaw/setup.sh" \
    "$SCRIPT_DIR/openclaw/configure-llm.sh" \
    "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/"
ssh_cmd "chmod +x $DEPLOY_DIR/setup.sh $DEPLOY_DIR/configure-llm.sh"

# Generate .env with COMPOSE_PROFILES for systemd
if [ -n "$PROFILES" ]; then
    info "Writing .env (COMPOSE_PROFILES=$PROFILES)..."
    ssh_cmd "echo 'COMPOSE_PROFILES=$PROFILES' > $DEPLOY_DIR/.env"
else
    # Remove stale .env if no profiles needed
    ssh_cmd "rm -f $DEPLOY_DIR/.env"
fi

ok "Files synced"

# ── Install systemd service ──────────────────────────────────────────
info "Installing systemd service..."
rsync -az "$SCRIPT_DIR/reachy-claw.service" "$JETSON_USER@$JETSON_HOST:/tmp/reachy-claw.service"
ssh_cmd "sudo cp /tmp/reachy-claw.service /etc/systemd/system/reachy-claw.service && sudo systemctl daemon-reload && sudo systemctl enable reachy-claw.service"
ok "systemd service installed and enabled"

# ── Stop old containers ───────────────────────────────────────────────
info "Stopping old containers..."
PROFILE_FLAGS=$(compose_profile_flags)
ssh_cmd "cd $DEPLOY_DIR && docker compose --profile ollama --profile openclaw --profile vision down 2>/dev/null" || true

# Also clean up legacy separate openclaw compose if it exists
ssh_cmd "cd $DEPLOY_DIR/openclaw && docker compose down 2>/dev/null" || true

# ── Start stack ───────────────────────────────────────────────────────
info "Pulling and starting ($MODES)..."
ssh_cmd "cd $DEPLOY_DIR && docker compose $PROFILE_FLAGS pull && docker compose $PROFILE_FLAGS up -d"
ok "Containers started"

# ── Ollama: stop host service & pull model ──────────────────────────
if [ "$USE_OPENCLAW" != true ]; then
    info "Disabling host Ollama service (now containerized)..."
    ssh_cmd "sudo systemctl stop ollama 2>/dev/null; sudo systemctl disable ollama 2>/dev/null" || true

    info "Waiting for Ollama container to be ready..."
    for i in $(seq 1 12); do
        if ssh_cmd "curl -sf http://localhost:11434/api/tags" >/dev/null 2>&1; then
            break
        fi
        sleep 5
    done

    info "Pulling model qwen3.5:2b-q4_K_M (this may take a while on first run)..."
    ssh_cmd "docker exec ollama ollama pull qwen3.5:2b-q4_K_M"
    ok "Model ready"
fi

# ── OpenClaw first-time setup ─────────────────────────────────────────
if [ "$SETUP_OPENCLAW" = true ]; then
    info "Running OpenClaw first-time setup (extension + config)..."
    ssh_cmd "cd $DEPLOY_DIR && bash setup.sh"
    ok "OpenClaw setup complete"
fi

# ── Smoke tests ───────────────────────────────────────────────────────
info "=== Running smoke tests ==="

# Poll-based wait for reachy-daemon healthcheck (timeout 60s)
wait_for_service() {
    local name="$1" cmd="$2" timeout="${3:-60}"
    local elapsed=0
    info "Waiting for $name..."
    while [ $elapsed -lt $timeout ]; do
        if ssh_cmd "$cmd" >/dev/null 2>&1; then
            ok "$name"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    err "$name is not responding after ${timeout}s"
    return 1
}

check_service() {
    local name="$1" cmd="$2"
    if ssh_cmd "$cmd" >/dev/null 2>&1; then
        ok "$name"
    else
        err "$name is not responding"
    fi
}

check_service "Speech service (:8621)" "curl -sf http://localhost:8621/health"
wait_for_service "Reachy daemon (:38001)" "curl -sf http://localhost:38001/" 60
wait_for_service "Reachy claw (:8640)" "curl -sf http://localhost:8640/health" 60

if [ "$USE_OPENCLAW" = true ]; then
    wait_for_service "OpenClaw gateway (:18789)" "curl -sf http://localhost:18789/healthz" 60
else
    wait_for_service "Ollama (:11434)" "curl -sf http://localhost:11434/api/tags" 60
fi

if [ "$USE_VISION" = true ]; then
    wait_for_service "Vision TRT (:8630)" "curl -sf http://localhost:8630/health" 120
fi

echo ""
info "Container status:"
ssh_cmd "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'reachy|openclaw|ollama|speech|voice|vision'" || true

echo ""
ok "Deployment complete!"
echo ""

if [ "$USE_OPENCLAW" = true ]; then
    echo "Next steps (OpenClaw mode):"
    echo "  1. Configure LLM (if not done):"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && ./configure-llm.sh dashscope <api-key>'"
    echo ""
    echo "  2. View logs:"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && docker compose $(compose_profile_flags) logs -f'"
else
    echo "Next steps (Ollama mode):"
    echo "  1. View logs:"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && docker compose $(compose_profile_flags) logs -f'"
    echo ""
    echo "  To switch to OpenClaw mode later:"
    echo "     ./deploy.sh --setup-openclaw"
fi
echo ""
