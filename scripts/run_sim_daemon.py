#!/usr/bin/env python3
"""Launch MuJoCo sim daemon on macOS.

macOS requires `mjpython` for MuJoCo GUI, and the mujoco backend
imports GStreamer modules that aren't available on macOS.

Usage:
    .venv/bin/mjpython scripts/run_sim_daemon.py [--sim] [--localhost-only] [--deactivate-audio]
"""

import sys
import types
from unittest.mock import MagicMock

# ── Shim: stub out GStreamer modules not available on macOS ──────────────

# gi / gi.repository
gi_mod = types.ModuleType("gi")
gi_mod.require_version = lambda *a, **kw: None  # type: ignore[attr-defined]

gi_repo = types.ModuleType("gi.repository")
gi_repo.Gst = MagicMock()  # type: ignore[attr-defined]
gi_repo.GLib = MagicMock()  # type: ignore[attr-defined]

gi_mod.repository = gi_repo  # type: ignore[attr-defined]

sys.modules.setdefault("gi", gi_mod)
sys.modules.setdefault("gi.repository", gi_repo)
sys.modules.setdefault("gi.repository.Gst", gi_repo.Gst)
sys.modules.setdefault("gi.repository.GLib", gi_repo.GLib)

# GStreamer media modules used by mujoco backend
for mod_name in [
    "reachy_mini.media.gstreamer_udp_camera",
    "reachy_mini.media.gstreamer_audio",
]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.GStreamerUDPCamera = MagicMock  # type: ignore[attr-defined]
        stub.GStreamerAudio = MagicMock  # type: ignore[attr-defined]
        sys.modules[mod_name] = stub

# ── Delegate to daemon ───────────────────────────────────────────────────

from reachy_mini.daemon.app.main import main  # noqa: E402

main()
