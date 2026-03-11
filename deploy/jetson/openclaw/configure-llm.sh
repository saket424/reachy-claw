#!/usr/bin/env bash
# Configure LLM provider for OpenClaw (separate from main deploy flow).
#
# Usage:
#   ./configure-llm.sh dashscope <api-key>        # DashScope (Aliyun Qwen)
#   ./configure-llm.sh openai <api-key>            # OpenAI
#   ./configure-llm.sh <provider> <api-key> [model] [base-url]
#
# Preset providers:
#   dashscope  → Qwen3.5-Flash via https://dashscope.aliyuncs.com/compatible-mode/v1
#   openai     → gpt-4o via https://api.openai.com/v1
set -euo pipefail

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

PROVIDER="${1:-}"
API_KEY="${2:-}"
MODEL="${3:-}"
BASE_URL="${4:-}"

if [ -z "$PROVIDER" ] || [ -z "$API_KEY" ]; then
    echo "Usage: $0 <provider> <api-key> [model] [base-url]"
    echo ""
    echo "Presets:"
    echo "  dashscope <key>    → Qwen3.5-Flash (DashScope/Aliyun)"
    echo "  openai <key>       → gpt-4o (OpenAI)"
    exit 1
fi

# Presets
case "$PROVIDER" in
    dashscope)
        BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
        MODEL="${MODEL:-Qwen3.5-Flash}"
        ;;
    openai)
        BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
        MODEL="${MODEL:-gpt-4o}"
        ;;
    *)
        if [ -z "$MODEL" ] || [ -z "$BASE_URL" ]; then
            err "Custom provider requires: $0 $PROVIDER <api-key> <model> <base-url>"
            exit 1
        fi
        ;;
esac

CONTAINER="openclaw-gateway"

info "Configuring $PROVIDER / $MODEL..."
docker exec "$CONTAINER" node -e "
const fs = require('fs');
const p = '/home/node/.openclaw/openclaw.json';
const cfg = JSON.parse(fs.readFileSync(p, 'utf8'));

// Add provider
if (!cfg.models) cfg.models = {};
if (!cfg.models.providers) cfg.models.providers = {};
cfg.models.providers['${PROVIDER}'] = {
    baseUrl: '${BASE_URL}',
    apiKey: '${API_KEY}',
    api: 'openai-completions',
    models: [{ id: '${MODEL}', name: '${MODEL}', reasoning: false }]
};

// Set as default
if (!cfg.agents) cfg.agents = {};
if (!cfg.agents.defaults) cfg.agents.defaults = {};
cfg.agents.defaults.model = '${PROVIDER}/${MODEL}';

// Set desktop-robot channel model
if (!cfg.channels) cfg.channels = {};
if (!cfg.channels['desktop-robot']) cfg.channels['desktop-robot'] = {};
cfg.channels['desktop-robot'].responseModel = '${PROVIDER}/${MODEL}';

fs.writeFileSync(p, JSON.stringify(cfg, null, 2));
console.log('Provider configured: ${PROVIDER}/${MODEL}');
"
ok "LLM configured: $PROVIDER/$MODEL"

info "Restarting gateway..."
docker restart "$CONTAINER"
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:18789/healthz >/dev/null 2>&1; then break; fi
    sleep 2
done
ok "Gateway restarted"
