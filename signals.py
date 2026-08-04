"""Signal bus for the visualizer. Plain files in the project root.

Every write is best-effort: a stray exception here must never take down
the voice line. The self-heal rule (see write_waveform) is what makes the
bus resilient to any other process stomping .voice_state mid-speech.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STATE_FILE = ROOT / ".voice_state"
WAVEFORM_FILE = ROOT / ".voice_waveform"
LOADING_PID_FILE = ROOT / ".voice_loading_pid"
# .voice_alert is intentionally never written here — it belongs to some
# other process on the machine that wants the visualizer's attention.

_WAVEFORM_MIN_INTERVAL = 1.0 / 15.0  # at most 15 writes/sec
_last_waveform_write = 0.0


def set_state(state: str) -> None:
    """state is one of: idle, listening, thinking, speaking."""
    try:
        STATE_FILE.write_text(state, encoding="utf-8")
    except OSError:
        pass


def _downsample_to_64(pcm_int16) -> list[float]:
    n = len(pcm_int16)
    if n == 0:
        return [0.0] * 64
    step = max(1, n // 64)
    points = []
    for i in range(64):
        start = i * step
        end = min(start + step, n)
        if start >= n:
            points.append(0.0)
            continue
        chunk = pcm_int16[start:end] if end > start else pcm_int16[start : start + 1]
        points.append(float(max(abs(int(x)) for x in chunk)))
    return points


def write_waveform(pcm_int16, force: bool = False) -> None:
    """Downsample a PCM int16 block to 64 points and write it, throttled
    to ~15Hz. Also re-asserts state=speaking every time (self-heal rule):
    this only runs while audio is audibly playing, so any stray process
    that stomps .voice_state gets corrected within about one throttle
    window (~70ms).
    """
    global _last_waveform_write
    now = time.monotonic()
    if not force and (now - _last_waveform_write) < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        samples = _downsample_to_64(pcm_int16)
        WAVEFORM_FILE.write_text(
            json.dumps({"ts": time.time(), "samples": samples}), encoding="utf-8"
        )
    except OSError:
        pass
    set_state("speaking")


def start_loading_indicator() -> None:
    try:
        LOADING_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def stop_loading_indicator() -> None:
    try:
        LOADING_PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass
