# LLM Provider Benchmark for Voice Assistant

> Tested: 2026-03-08 | Location: macOS → China API endpoints
> Use case: Desktop robot voice assistant (low-latency, short replies)
> All tests: streaming mode, thinking/reasoning disabled, system prompt + 1 user message

## Summary

| Provider | Model | Avg TTFB | Min | Max | Stability | Free Tier | Recommendation |
|----------|-------|----------|-----|-----|-----------|-----------|----------------|
| 阿里云 DashScope | **Qwen3.5-Flash** | **472ms** | 452ms | 499ms | ★★★★★ | 有 (限额) | **首选** |
| 阿里云 DashScope | Qwen3.5-35B-A3B | 604ms | 493ms | 865ms | ★★★★☆ | 有 (限额) | 备选 (更强能力) |
| 火山引擎 Ark | Doubao-Seed-2.0-mini | 561ms | 350ms | 963ms | ★★★☆☆ | 有 | 波动较大 |
| 火山引擎 Ark | Doubao-Seed-2.0-lite | 1302ms | 1091ms | 1740ms | ★★★★☆ | 有 | 过慢 |
| 智谱 ZAI | GLM-4.7-Flash | ~500ms* | 373ms | **30s+** | ★☆☆☆☆ | 免费 | 不推荐 (冷启动) |
| 智谱 ZAI | GLM-4.7-FlashX | ~700ms | 691ms | 11s+ | ★★☆☆☆ | 付费 | 不推荐 (冷启动) |
| DashScope CodingPlan | Kimi K2.5 | ~1000ms | 728ms | 1333ms | ★★★☆☆ | CodingPlan 额度 | 慢但稳定 |

\* GLM-4.7-Flash 正常时很快，但有严重的冷启动问题（偶发 10-30s 延迟）

## Detailed Results

### Qwen3.5-Flash (阿里云)

```
Provider:  dashscope.aliyuncs.com
Model ID:  qwen3.5-flash
API:       OpenAI-compatible
Thinking:  enable_thinking: false

Run1: TTFB=466ms  Total=599ms
Run2: TTFB=452ms  Total=691ms
Run3: TTFB=466ms  Total=616ms
Run4: TTFB=476ms  Total=652ms
Run5: TTFB=499ms  Total=747ms

Avg TTFB: 472ms | Std Dev: ~17ms
```

Extremely consistent. Best overall choice for voice assistant latency.

### Qwen3.5-35B-A3B (阿里云)

```
Provider:  dashscope.aliyuncs.com
Model ID:  qwen3.5-35b-a3b
API:       OpenAI-compatible
Thinking:  enable_thinking: false

Run1: TTFB=512ms  Total=763ms
Run2: TTFB=865ms  Total=1115ms
Run3: TTFB=556ms  Total=758ms
Run4: TTFB=493ms  Total=586ms
Run5: TTFB=594ms  Total=800ms

Avg TTFB: 604ms | Std Dev: ~142ms
```

MoE architecture (35B total, 3B active). Slightly slower but stronger reasoning. Good fallback.

### Doubao-Seed-2.0-mini (火山引擎)

```
Provider:  ark.cn-beijing.volces.com
Model ID:  doubao-seed-2-0-mini-260215
API:       OpenAI-compatible
Thinking:  thinking: {type: "disabled"}

Run1: TTFB=615ms  Total=1274ms
Run2: TTFB=350ms  Total=568ms
Run3: TTFB=398ms  Total=663ms
Run4: TTFB=963ms  Total=1137ms
Run5: TTFB=480ms  Total=752ms

Avg TTFB: 561ms | Std Dev: ~234ms
```

Can be very fast (350ms) but high variance. Occasional spikes near 1s.

### Doubao-Seed-2.0-lite (火山引擎)

```
Provider:  ark.cn-beijing.volces.com
Model ID:  doubao-seed-2-0-lite-260215
API:       OpenAI-compatible
Thinking:  thinking: {type: "disabled"}

Run1: TTFB=1239ms  Total=1584ms
Run2: TTFB=1740ms  Total=1871ms
Run3: TTFB=1212ms  Total=1534ms
Run4: TTFB=1228ms  Total=1444ms
Run5: TTFB=1091ms  Total=1326ms

Avg TTFB: 1302ms | Std Dev: ~234ms
```

Too slow for voice assistant use. "Lite" is misleading — it's a larger model optimized for quality, not speed.

### GLM-4.7-Flash (智谱)

```
Provider:  open.bigmodel.cn
Model ID:  glm-4.7-flash
API:       OpenAI-compatible
Thinking:  thinking: {type: "disabled"}

Run1: TTFB=1184ms
Run2: TTFB=373ms
Run3: TTFB=30206ms  ← cold start / rate limit

Direct API (earlier): 448ms, 642ms, 706ms
Via Gateway E2E: 4/10 tests timeout at 30s
```

Free tier is completely free but has severe cold-start issues and strict rate limits (1 concurrent request). Unusable for production.

### GLM-4.7-FlashX (智谱)

```
Provider:  open.bigmodel.cn
Model ID:  glm-4.7-flashx
API:       OpenAI-compatible (paid)
Thinking:  thinking: {type: "disabled"}

Direct API: 691-724ms typical, occasional 11s cold starts
```

Paid model, slightly better than Flash but still has cold-start problem.

### Kimi K2.5 (DashScope CodingPlan)

```
Provider:  coding.dashscope.aliyuncs.com
Model ID:  kimi-k2.5
API:       OpenAI-compatible

Direct API: 728ms, 1000ms, 1333ms
Via Gateway E2E: 10/10 tests pass, avg ~2000ms voice-to-voice
```

Stable but slow. Uses CodingPlan quota (separate billing).

## Thinking/Reasoning Disable Methods

| Provider | Parameter | Format |
|----------|-----------|--------|
| 阿里云 DashScope (Qwen) | `enable_thinking` | `"enable_thinking": false` |
| 智谱 ZAI (GLM) | `thinking` | `"thinking": {"type": "disabled"}` |
| 火山引擎 Ark (Doubao) | `thinking` | `"thinking": {"type": "disabled"}` |
| Moonshot (Kimi) | `thinking` | `"thinking": {"type": "disabled"}` |

All models default to thinking ON — must explicitly disable for voice assistant latency.

## API Endpoints

| Provider | Base URL | Auth |
|----------|----------|------|
| 阿里云 DashScope (通用) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Bearer token |
| 阿里云 DashScope (CodingPlan) | `https://coding.dashscope.aliyuncs.com/v1` | Bearer token |
| 智谱 ZAI | `https://open.bigmodel.cn/api/paas/v4` | Bearer token |
| 火山引擎 Ark | `https://ark.cn-beijing.volces.com/api/v3` | Bearer token |

## Gateway Overhead Profile

Profiled via `performance.now()` instrumentation in `desktop-robot/server.ts` (10 warm-session requests):

```
Phase                    Time        Notes
─────────────────────────────────────────────────────────
Gateway setup            0-1ms       State transition, session mgmt, config build
deps load (cold)         259ms       First request only (loadCoreAgentDeps)
deps load (warm)         0ms         Cached after first call
agent_prep               0-3ms       resolveDir, workspace, sessionFile
api_call                 422-694ms   runEmbeddedPiAgent → first stream delta
─────────────────────────────────────────────────────────
Total overhead           ~1ms        (excluding first-request cold start)
```

**Gateway framework adds ~1ms overhead.** 99%+ of TTFT is the LLM API call itself.
The only significant one-time cost is the first-request deps load (~260ms), cached after that.

TTFT variance (±200ms) comes entirely from the upstream LLM API, not the gateway.

## Conclusion

For the desktop robot voice assistant pipeline (target: <500ms TTFB):

1. **Qwen3.5-Flash** — Best choice. ~470ms avg, rock-solid stability, free tier available
2. **Qwen3.5-35B-A3B** — Good backup. ~600ms avg, stronger model, slight variance
3. **Doubao-Seed-2.0-mini** — Viable alternative. Fast floor (350ms) but unpredictable spikes
4. Others — Not recommended for latency-sensitive voice use
