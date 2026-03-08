"""E2E latency benchmark: measure TTFT and voice-to-voice through the full pipeline.

Measures:
  - Gateway TTFT: time from send_message to first stream_delta
  - Gateway total: time to stream_end
  - TTS batch latency: Kokoro full synthesis time
  - TTS streaming TTFB: time to first audio chunk
  - Voice-to-Voice estimate: Gateway TTFT + TTS streaming TTFB

Requires:
  1. OpenClaw desktop-robot extension on ws://127.0.0.1:18790/desktop-robot
  2. Jetson speech service on http://100.67.111.58:8000

Run with: uv run pytest tests/test_benchmark.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import urllib.request

import pytest
import websockets

SPEECH_URL = os.environ.get("SPEECH_SERVICE_URL", "http://100.67.111.58:8000")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "ws://127.0.0.1:18790/desktop-robot")


# ── Reachability ────────────────────────────────────────────────────────


def _speech_reachable() -> bool:
    try:
        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        return json.loads(resp.read().decode()).get("tts", False)
    except Exception:
        return False


def _gateway_reachable() -> bool:
    async def _check():
        try:
            ws = await asyncio.wait_for(websockets.connect(GATEWAY_URL), timeout=3)
            await ws.send(json.dumps({"type": "hello", "sessionId": "bench-probe"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            await ws.close()
            return msg.get("type") == "welcome"
        except Exception:
            return False

    return asyncio.run(_check())


_has_speech = _speech_reachable()
_has_gateway = _gateway_reachable()
_skip_no_gateway = pytest.mark.skipif(
    not _has_gateway, reason=f"Gateway not reachable at {GATEWAY_URL}"
)
_skip_no_speech = pytest.mark.skipif(
    not _has_speech, reason=f"Speech service not reachable at {SPEECH_URL}"
)


# ── Helpers ─────────────────────────────────────────────────────────────


async def _gateway_roundtrip(
    ws: websockets.WebSocketClientProtocol,
    prompt: str,
    timeout: float = 30.0,
) -> dict:
    """Send a message on an existing WS session, return timing dict."""
    # Notify speaking_done so session is in idle state
    await ws.send(json.dumps({"type": "state_change", "state": "speaking_done"}))
    await asyncio.sleep(0.2)

    t_send = time.monotonic()
    await ws.send(json.dumps({"type": "message", "text": prompt}))

    t_first_delta = None
    delta_count = 0
    full_text = ""

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        now = time.monotonic()

        if msg["type"] == "stream_delta":
            delta_count += 1
            if t_first_delta is None:
                t_first_delta = now
        elif msg["type"] == "stream_end":
            full_text = msg.get("fullText", "")
            t_end = now
            break
        elif msg["type"] == "error":
            pytest.fail(f"Gateway error: {msg}")

    ttft_ms = round((t_first_delta - t_send) * 1000) if t_first_delta else -1
    total_ms = round((t_end - t_send) * 1000)

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "delta_count": delta_count,
        "full_text": full_text,
    }


async def _tts_batch_latency(text: str) -> dict:
    """Measure Kokoro TTS batch synthesis latency."""
    import aiohttp

    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SPEECH_URL}/tts",
            json={"text": text, "speaker_id": 3, "speed": 1.0},
        ) as resp:
            data = await resp.read()
            t_done = time.monotonic()

    return {
        "tts_batch_ms": round((t_done - t0) * 1000),
        "audio_bytes": len(data),
    }


async def _tts_streaming_latency(text: str) -> dict:
    """Measure Kokoro streaming TTS TTFB and total time."""
    import aiohttp

    t0 = time.monotonic()
    t_first_chunk = None
    chunk_count = 0
    total_bytes = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SPEECH_URL}/tts/streaming",
            json={"text": text, "speaker_id": 3, "speed": 1.0},
        ) as resp:
            async for chunk in resp.content.iter_any():
                if t_first_chunk is None:
                    t_first_chunk = time.monotonic()
                chunk_count += 1
                total_bytes += len(chunk)

    t_done = time.monotonic()
    return {
        "tts_stream_ttfb_ms": round((t_first_chunk - t0) * 1000)
        if t_first_chunk
        else -1,
        "tts_stream_total_ms": round((t_done - t0) * 1000),
        "tts_chunks": chunk_count,
        "tts_bytes": total_bytes,
    }


def _print_summary(label: str, values: list[int | float], unit: str = "ms"):
    """Print min/avg/max/stdev for a list of values."""
    if not values:
        return
    avg = round(statistics.mean(values))
    mn = min(values)
    mx = max(values)
    sd = round(statistics.stdev(values)) if len(values) > 1 else 0
    print(f"  {label:<25} avg={avg}{unit}  min={mn}{unit}  max={mx}{unit}  stdev={sd}{unit}")


# ── Tests ───────────────────────────────────────────────────────────────


BENCH_PROMPTS = [
    ("short_en", "Say hello in one sentence."),
    ("short_zh", "用一句话跟我打个招呼。"),
    ("medium_en", "Tell me a fun fact about robots in two sentences."),
    ("medium_zh", "用两句话告诉我一个关于机器人的有趣事实。"),
    ("conversational", "What's your favorite color and why?"),
    ("followup", "Can you say that again but shorter?"),
]


@_skip_no_gateway
class TestGatewayTTFT:
    """Benchmark Gateway TTFT using a single warm session."""

    @pytest.mark.asyncio
    async def test_gateway_ttft_benchmark(self):
        ws = await asyncio.wait_for(websockets.connect(GATEWAY_URL), timeout=5)
        await ws.send(json.dumps({"type": "hello"}))
        welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert welcome["type"] == "welcome"

        results = []
        print("\n" + "=" * 76)
        print("GATEWAY TTFT BENCHMARK")
        print("=" * 76)
        print(f"{'#':>2} {'Prompt':<35} {'TTFT':>7} {'Total':>7} {'Chars':>5} {'Deltas':>6}")
        print("-" * 76)

        for i, (label, prompt) in enumerate(BENCH_PROMPTS):
            r = await _gateway_roundtrip(ws, prompt)
            results.append(r)
            print(
                f"{i + 1:>2} {prompt:<35} {r['ttft_ms']:>6}ms {r['total_ms']:>6}ms "
                f"{len(r['full_text']):>5} {r['delta_count']:>6}"
            )

        await ws.close()

        print("-" * 76)
        ttfts = [r["ttft_ms"] for r in results]
        totals = [r["total_ms"] for r in results]
        _print_summary("Gateway TTFT", ttfts)
        _print_summary("Gateway Total", totals)

        # Soft assertion: avg TTFT should be under 2s
        avg_ttft = statistics.mean(ttfts)
        assert avg_ttft < 2000, f"Avg TTFT {avg_ttft}ms > 2000ms threshold"


@_skip_no_gateway
@_skip_no_speech
class TestFullPipelineBenchmark:
    """Benchmark full voice-to-voice pipeline: Gateway + TTS."""

    @pytest.mark.asyncio
    async def test_voice_to_voice_benchmark(self):
        ws = await asyncio.wait_for(websockets.connect(GATEWAY_URL), timeout=5)
        await ws.send(json.dumps({"type": "hello"}))
        welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert welcome["type"] == "welcome"

        results = []
        print("\n" + "=" * 76)
        print("FULL PIPELINE BENCHMARK (Gateway + Jetson TTS)")
        print("=" * 76)

        for i, (label, prompt) in enumerate(BENCH_PROMPTS):
            gw = await _gateway_roundtrip(ws, prompt)
            text = gw["full_text"]

            tts_batch = await _tts_batch_latency(text)
            tts_stream = await _tts_streaming_latency(text)

            v2v = gw["ttft_ms"] + tts_stream["tts_stream_ttfb_ms"]

            row = {
                "label": label,
                "prompt": prompt,
                **gw,
                **tts_batch,
                **tts_stream,
                "v2v_ms": v2v,
            }
            results.append(row)

            print(f"\n--- [{i + 1}] {label}: {prompt}")
            print(f"  Gateway TTFT:      {gw['ttft_ms']}ms")
            print(f"  Gateway Total:     {gw['total_ms']}ms  ({gw['delta_count']} deltas, {len(text)} chars)")
            print(f"  TTS batch:         {tts_batch['tts_batch_ms']}ms")
            print(f"  TTS stream TTFB:   {tts_stream['tts_stream_ttfb_ms']}ms")
            print(f"  TTS stream total:  {tts_stream['tts_stream_total_ms']}ms")
            print(f"  Voice-to-Voice:    {v2v}ms")
            print(f"  Reply: {text[:80]}")

        await ws.close()

        print("\n" + "=" * 76)
        print("SUMMARY")
        print("=" * 76)
        _print_summary("Gateway TTFT", [r["ttft_ms"] for r in results])
        _print_summary("TTS stream TTFB", [r["tts_stream_ttfb_ms"] for r in results])
        _print_summary("Voice-to-Voice", [r["v2v_ms"] for r in results])
        print()
        print("  Voice-to-Voice = Gateway TTFT + TTS streaming TTFB")
        print("  (excludes ASR; add ~200-300ms for Paraformer STT)")

        avg_v2v = statistics.mean(r["v2v_ms"] for r in results)
        assert avg_v2v < 3000, f"Avg voice-to-voice {avg_v2v}ms > 3000ms threshold"


@_skip_no_gateway
class TestDirectApiTTFB:
    """Compare direct API TTFB vs gateway TTFB (measures gateway overhead)."""

    @pytest.mark.asyncio
    async def test_gateway_overhead(self):
        """Estimate gateway overhead by comparing direct curl-equivalent vs gateway."""
        import aiohttp

        # Direct API call (OpenAI-compatible, same as gateway backend)
        # Read current provider from env or use defaults
        api_url = os.environ.get(
            "LLM_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        api_key = os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

        if not api_key:
            pytest.skip("Set LLM_API_KEY env var to run direct API comparison")

        prompt = "Say hello in one sentence."
        messages = [
            {"role": "system", "content": "You are a voice assistant. Keep replies to 1-2 short sentences."},
            {"role": "user", "content": prompt},
        ]

        # Direct API TTFB (3 runs)
        direct_ttfbs = []
        async with aiohttp.ClientSession() as session:
            for _ in range(3):
                t0 = time.monotonic()
                async with session.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "stream": True,
                        "enable_thinking": False,
                        "messages": messages,
                    },
                ) as resp:
                    async for _ in resp.content:
                        direct_ttfbs.append(round((time.monotonic() - t0) * 1000))
                        break

        # Gateway TTFB (3 runs)
        ws = await asyncio.wait_for(websockets.connect(GATEWAY_URL), timeout=5)
        await ws.send(json.dumps({"type": "hello"}))
        await asyncio.wait_for(ws.recv(), timeout=5)

        gw_ttfbs = []
        for _ in range(3):
            r = await _gateway_roundtrip(ws, prompt)
            gw_ttfbs.append(r["ttft_ms"])

        await ws.close()

        direct_avg = round(statistics.mean(direct_ttfbs))
        gw_avg = round(statistics.mean(gw_ttfbs))
        overhead = gw_avg - direct_avg

        print("\n" + "=" * 76)
        print("GATEWAY OVERHEAD ANALYSIS")
        print("=" * 76)
        print(f"  Direct API TTFB:   {direct_ttfbs} → avg={direct_avg}ms")
        print(f"  Gateway TTFB:      {gw_ttfbs} → avg={gw_avg}ms")
        print(f"  Gateway overhead:  ~{overhead}ms")
        print(f"  Model: {model} via {api_url}")
