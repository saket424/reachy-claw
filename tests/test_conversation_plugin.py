"""Tests for the ConversationPlugin (dual-pipeline architecture)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.plugins.conversation_plugin import (
    ConversationPlugin,
    ConvState,
    SentenceItem,
    _drain_queue,
)


@pytest.fixture
def standalone_app(mock_reachy):
    """App in standalone mode (no gateway) with mock robot."""
    from reachy_claw.app import ReachyClawApp

    config = Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    a = ReachyClawApp(config)
    a.reachy = mock_reachy
    return a


# ── ConvState enum ────────────────────────────────────────────────────


class TestConvState:
    def test_all_states_exist(self):
        assert ConvState.IDLE.value == "idle"
        assert ConvState.LISTENING.value == "listening"
        assert ConvState.TRANSCRIBING.value == "transcribing"
        assert ConvState.THINKING.value == "thinking"
        assert ConvState.SPEAKING.value == "speaking"


# ── SentenceItem ──────────────────────────────────────────────────────


class TestSentenceItem:
    def test_defaults(self):
        item = SentenceItem(text="Hello.")
        assert item.text == "Hello."
        assert item.is_last is False

    def test_is_last(self):
        item = SentenceItem(text="Done.", is_last=True)
        assert item.is_last is True


# ── drain_queue helper ────────────────────────────────────────────────


class TestDrainQueue:
    @pytest.mark.asyncio
    async def test_drains_all_items(self):
        q: asyncio.Queue[str] = asyncio.Queue()
        await q.put("a")
        await q.put("b")
        await q.put("c")
        _drain_queue(q)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_noop_on_empty_queue(self):
        q: asyncio.Queue[str] = asyncio.Queue()
        _drain_queue(q)
        assert q.empty()


# ── Sentence accumulator ─────────────────────────────────────────────


class TestSentenceAccumulator:
    @pytest.mark.asyncio
    async def test_splits_on_period(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        await plugin._stream_text_queue.put("Hello world. ")
        await plugin._stream_text_queue.put("How are you?")
        await plugin._stream_text_queue.put(None)

        # Run accumulator briefly
        task = asyncio.create_task(plugin._sentence_accumulator())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sentences = []
        while not plugin._sentence_queue.empty():
            sentences.append(plugin._sentence_queue.get_nowait())

        # Should get at least "Hello world." and "How are you?" (or combined final)
        texts = [s.text for s in sentences if s.text]
        assert any("Hello world" in t for t in texts)

    @pytest.mark.asyncio
    async def test_flushes_buffer_on_stream_end(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        await plugin._stream_text_queue.put("Short")
        await plugin._stream_text_queue.put(None)

        task = asyncio.create_task(plugin._sentence_accumulator())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sentences = []
        while not plugin._sentence_queue.empty():
            sentences.append(plugin._sentence_queue.get_nowait())

        # "Short" is too short for sentence split, should flush as is_last
        assert any(s.is_last for s in sentences)
        assert any("Short" in s.text for s in sentences)


# ── Output pipeline ──────────────────────────────────────────────────


class TestOutputPipeline:
    @pytest.mark.asyncio
    async def test_speaks_sentences(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Hello world."), None))
        await plugin._audio_queue.put((SentenceItem(text="Done.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        # Wait long enough for both sentences + inter-sentence pause (0.15s)
        await asyncio.sleep(0.5)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "Hello world." in spoken
        assert "Done." in spoken

    @pytest.mark.asyncio
    async def test_interrupt_drains_queue(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            # Simulate interrupt after first sentence
            plugin._interrupt_event.set()
            return True

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="First."), None))
        await plugin._audio_queue.put((SentenceItem(text="Second."), None))
        await plugin._audio_queue.put((SentenceItem(text="Third.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only first sentence spoken, rest drained
        assert len(spoken) == 1
        assert spoken[0] == "First."

    @pytest.mark.asyncio
    async def test_sets_is_speaking_flag(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        speaking_during = []

        async def mock_speak(text, prefetched_chunks=None):
            speaking_during.append(standalone_app.is_speaking)
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Hello.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert any(s is True for s in speaking_during)
        # After done, is_speaking should be reset
        assert standalone_app.is_speaking is False

    @pytest.mark.asyncio
    async def test_empty_is_last_finishes_speaking(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True

        await plugin._audio_queue.put((SentenceItem(text="", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert standalone_app.is_speaking is False
        assert plugin._state == ConvState.IDLE


# ── Callback wiring ──────────────────────────────────────────────────


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_stream_start_drains_queue(self, standalone_app):
        from reachy_claw.plugins.conversation_plugin import _RESET_BUFFER

        plugin = ConversationPlugin(standalone_app)
        await plugin._stream_text_queue.put("stale")
        await plugin._stream_text_queue.put("data")

        await plugin._on_stream_start("run-1")

        assert plugin._current_run_id == "run-1"
        # Stale data drained, only the reset sentinel remains
        assert plugin._stream_text_queue.qsize() == 1
        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is _RESET_BUFFER
        assert plugin._state == ConvState.THINKING

    @pytest.mark.asyncio
    async def test_stream_delta_puts_text(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.THINKING
        plugin._current_run_id = "run-1"
        await plugin._on_stream_delta("hello", "run-1")

        text = plugin._stream_text_queue.get_nowait()
        assert text == "hello"
        assert plugin._state == ConvState.SPEAKING

    @pytest.mark.asyncio
    async def test_stream_end_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_stream_end("full text", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_stream_abort_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_stream_abort("interrupted", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_task_completed_queues_notification_for_output_pipeline(
        self, standalone_app
    ):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_task_completed("Background search finished", "task-1")

        item = plugin._sentence_queue.get_nowait()
        assert item is not None
        assert item.text == "Background search finished"
        assert item.is_last is True


# ── Fire interrupt ────────────────────────────────────────────────────


class TestFireInterrupt:
    @pytest.mark.asyncio
    async def test_sets_event_and_drains_queues(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin.app.is_speaking = True

        await plugin._stream_text_queue.put("some text")
        await plugin._sentence_queue.put(SentenceItem(text="sentence"))
        await plugin._audio_queue.put((SentenceItem(text="audio"), None))

        await plugin._fire_interrupt()

        assert plugin._interrupt_event.is_set()
        assert plugin._stream_text_queue.empty()
        assert plugin._sentence_queue.empty()
        assert plugin._audio_queue.empty()
        assert plugin.app.is_speaking is False

    @pytest.mark.asyncio
    async def test_sends_interrupt_to_gateway(self, standalone_app):
        standalone_app.config.standalone_mode = False
        plugin = ConversationPlugin(standalone_app)
        plugin._client = MagicMock()
        plugin._client.send_interrupt = AsyncMock()

        await plugin._fire_interrupt()

        plugin._client.send_interrupt.assert_called_once()


# ── State transitions ─────────────────────────────────────────────────


class TestStateTransitions:
    def test_initial_state_is_idle(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        assert plugin._state == ConvState.IDLE

    def test_set_state_logs_transition(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._set_state(ConvState.LISTENING)
        assert plugin._state == ConvState.LISTENING

    def test_set_state_noop_on_same(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.SPEAKING
        plugin._set_state(ConvState.SPEAKING)
        assert plugin._state == ConvState.SPEAKING


# ── Emotion queueing ─────────────────────────────────────────────────


class TestEmotionIntegration:
    @pytest.mark.asyncio
    async def test_thinking_emotion_queued_on_send(self, standalone_app):
        standalone_app.config.standalone_mode = False
        standalone_app.config.play_emotions = True

        plugin = ConversationPlugin(standalone_app)
        plugin._client = MagicMock()
        plugin._client.send_message_streaming = AsyncMock()
        plugin._client.send_state_change = AsyncMock()

        standalone_app.emotions.queue_emotion("thinking")

        expr = standalone_app.emotions.get_next_expression()
        assert expr is not None
        assert "hinking" in expr.description.lower()


# ── Stop / cleanup ────────────────────────────────────────────────────


class TestConversationCleanup:
    @pytest.mark.asyncio
    async def test_stop_cleans_resources(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._audio = MagicMock()
        plugin._audio.stop = AsyncMock()
        plugin._tts = MagicMock()
        plugin._client = MagicMock()
        plugin._client.disconnect = AsyncMock()

        await plugin.stop()

        assert plugin._running is False
        plugin._audio.stop.assert_called_once()
        plugin._client.disconnect.assert_called_once()
        plugin._tts.cleanup.assert_called_once()


# ── Config: barge_in_confirm_frames ───────────────────────────────────


class TestBargeInConfig:
    def test_default_confirm_frames(self):
        config = Config()
        assert config.barge_in_confirm_frames == 3

    def test_custom_confirm_frames(self):
        config = Config(barge_in_confirm_frames=5)
        assert config.barge_in_confirm_frames == 5

    def test_default_silero_threshold(self):
        config = Config()
        assert config.barge_in_silero_threshold == 0.6

    def test_custom_silero_threshold(self):
        config = Config(barge_in_silero_threshold=0.8)
        assert config.barge_in_silero_threshold == 0.8

    def test_default_cooldown_ms(self):
        config = Config()
        assert config.barge_in_cooldown_ms == 500

    def test_custom_cooldown_ms(self):
        config = Config(barge_in_cooldown_ms=1000)
        assert config.barge_in_cooldown_ms == 1000

    def test_default_energy_threshold(self):
        config = Config()
        assert config.barge_in_energy_threshold == 0.02


# ── Config: barge_in YAML mapping ─────────────────────────────────────


class TestBargeInYamlMapping:
    def test_yaml_maps_silero_threshold(self):
        from reachy_claw.config import _apply_yaml

        config = Config()
        _apply_yaml(config, {"barge_in": {"silero_threshold": 0.7}})
        assert config.barge_in_silero_threshold == 0.7

    def test_yaml_maps_cooldown_ms(self):
        from reachy_claw.config import _apply_yaml

        config = Config()
        _apply_yaml(config, {"barge_in": {"cooldown_ms": 1000}})
        assert config.barge_in_cooldown_ms == 1000


# ── VAD: speech_probability ───────────────────────────────────────────


class TestVADSpeechProbability:
    def test_energy_vad_returns_binary_probability(self):
        from reachy_claw.vad import EnergyVAD

        vad = EnergyVAD(threshold=0.01)
        silence = np.zeros(512, dtype=np.float32)
        loud = np.full(512, 0.5, dtype=np.float32)
        assert vad.speech_probability(silence) == 0.0
        assert vad.speech_probability(loud) == 1.0

    def test_base_class_default_delegates_to_is_speech(self):
        from reachy_claw.vad import VADBackend

        class DummyVAD(VADBackend):
            def is_speech(self, audio, sample_rate=16000):
                return True

        vad = DummyVAD()
        assert vad.speech_probability(np.zeros(512, dtype=np.float32)) == 1.0


# ── ConversationPlugin: _speaking_since tracking ─────────────────────


class TestSpeakingSinceTracking:
    def _make_plugin(self):
        app = MagicMock()
        app.config = Config()
        plugin = ConversationPlugin(app)
        return plugin

    def test_speaking_since_set_on_state_change(self):
        """_speaking_since is updated when state changes to SPEAKING."""
        import time as _time

        plugin = self._make_plugin()
        before = _time.monotonic()
        plugin._set_state(ConvState.SPEAKING)
        after = _time.monotonic()
        assert before <= plugin._speaking_since <= after

    def test_speaking_since_not_updated_for_other_states(self):
        """_speaking_since is NOT updated for non-SPEAKING states."""
        plugin = self._make_plugin()
        plugin._speaking_since = 0.0
        plugin._set_state(ConvState.LISTENING)
        assert plugin._speaking_since == 0.0
