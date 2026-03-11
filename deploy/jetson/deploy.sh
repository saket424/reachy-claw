#!/usr/bin/env bash
# One-click deploy: reachy-claw stack + OpenClaw gateway to Jetson.
#
# Prerequisites:
#   - Speech service already running on Jetson (port 8621)
#   - SSH access to Jetson (key-based auth recommended)
#
# Usage:
#   ./deploy.sh                           # deploy all
#   ./deploy.sh --reachy-only             # skip OpenClaw (user already has it)
#   JETSON_HOST=192.168.1.100 ./deploy.sh # custom host
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Config ────────────────────────────────────────────────────────────
JETSON_HOST="${JETSON_HOST:-recomputer}"
JETSON_USER="${JETSON_USER:-recomputer}"
DEPLOY_DIR="${DEPLOY_DIR:-~/reachy-deploy}"
SKIP_OPENCLAW=false

for arg in "$@"; do
    case "$arg" in
        --reachy-only) SKIP_OPENCLAW=true ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
die()   { err "$@"; exit 1; }

ssh_cmd() { ssh "$JETSON_USER@$JETSON_HOST" "$@"; }

# ── Pre-flight ────────────────────────────────────────────────────────
info "Checking SSH connection to $JETSON_USER@$JETSON_HOST..."
ssh_cmd "echo ok" >/dev/null 2>&1 || die "Cannot SSH to $JETSON_HOST"
ok "SSH connected"

# ── Deploy OpenClaw ──────────────────────────────────────────────────
if [ "$SKIP_OPENCLAW" = false ]; then
    info "=== Deploying OpenClaw gateway ==="

    # Sync openclaw compose + scripts
    info "Syncing OpenClaw deploy files..."
    ssh_cmd "mkdir -p $DEPLOY_DIR/openclaw"
    rsync -az \
        "$SCRIPT_DIR/openclaw/docker-compose.yml" \
        "$SCRIPT_DIR/openclaw/setup.sh" \
        "$SCRIPT_DIR/openclaw/configure-llm.sh" \
        "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/openclaw/"
    ssh_cmd "chmod +x $DEPLOY_DIR/openclaw/setup.sh $DEPLOY_DIR/openclaw/configure-llm.sh"

    # Stop old openclaw container if exists
    info "Stopping old OpenClaw container (if any)..."
    ssh_cmd "docker stop openclaw-gateway 2>/dev/null; docker rm openclaw-gateway 2>/dev/null" || true

    # Start OpenClaw
    info "Starting OpenClaw gateway..."
    ssh_cmd "cd $DEPLOY_DIR/openclaw && docker compose pull && docker compose up -d"

    # Run setup (install extension deps + init config)
    info "Running OpenClaw setup..."
    ssh_cmd "cd $DEPLOY_DIR/openclaw && bash setup.sh"
    ok "OpenClaw gateway deployed"
fi

# ── Deploy Reachy stack ──────────────────────────────────────────────
info "=== Deploying Reachy stack ==="

# Sync reachy compose + config
info "Syncing Reachy deploy files..."
ssh_cmd "mkdir -p $DEPLOY_DIR/reachy"
rsync -az \
    "$SCRIPT_DIR/reachy/docker-compose.yml" \
    "$SCRIPT_DIR/reachy/reachy-claw.jetson.yaml" \
    "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/reachy/"

# Stop old containers
info "Stopping old Reachy containers (if any)..."
ssh_cmd "cd $DEPLOY_DIR/reachy && docker compose down 2>/dev/null" || true

# Pull and start
info "Pulling and starting Reachy containers..."
ssh_cmd "cd $DEPLOY_DIR/reachy && docker compose pull && docker compose up -d"
ok "Reachy stack deployed"

# ── Smoke tests ──────────────────────────────────────────────────────
info "=== Running smoke tests ==="
sleep 5

check_service() {
    local name="$1" cmd="$2"
    if ssh_cmd "$cmd" >/dev/null 2>&1; then
        ok "$name"
    else
        err "$name is not responding"
    fi
}

check_service "Speech service (:8621)" "curl -sf http://localhost:8621/health"
check_service "Reachy daemon (:38001)" "curl -sf http://localhost:38001/"
if [ "$SKIP_OPENCLAW" = false ]; then
    check_service "OpenClaw gateway (:18789)" "curl -sf http://localhost:18789/healthz"
fi

echo ""
info "Container status:"
ssh_cmd "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'reachy|openclaw|speech|voice'"

echo ""
ok "Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Configure LLM (if not done):"
echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR/openclaw && ./configure-llm.sh dashscope <api-key>'"
echo ""
echo "  2. View logs:"
echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR/reachy && docker compose logs -f'"
echo ""
