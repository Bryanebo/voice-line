"""Optional Spotify ducking via pycaw (Windows Core Audio session API).

pycaw controls Spotify's own per-app volume slider directly -- not the
system volume, not another app's. We never launch Spotify; if it isn't
running (no matching audio session), every call here is a silent no-op.
"""

from __future__ import annotations

import threading

from pycaw.pycaw import AudioUtilities

DUCK_THRESHOLD_PCT = 30.0
DUCK_FACTOR = 0.6
RESTORE_DEBOUNCE_S = 1.2


def _find_spotify_session():
    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception:
        return None
    for session in sessions:
        try:
            if session.Process and session.Process.name().lower() == "spotify.exe":
                return session
        except Exception:
            continue
    return None


class Ducking:
    def __init__(self) -> None:
        self._original_volume: float | None = None
        self._ducked = False
        self._restore_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def duck(self) -> None:
        """Call when the assistant starts speaking a chunk."""
        with self._lock:
            if self._restore_timer:
                self._restore_timer.cancel()
                self._restore_timer = None

            session = _find_spotify_session()
            if session is None:
                return
            try:
                volume_iface = session.SimpleAudioVolume
                current = volume_iface.GetMasterVolume()  # 0.0-1.0
            except Exception:
                return
            current_pct = current * 100.0
            if current_pct <= DUCK_THRESHOLD_PCT:
                return

            if not self._ducked:
                self._original_volume = current
                self._ducked = True

            target_pct = max(DUCK_THRESHOLD_PCT, current_pct * DUCK_FACTOR)
            try:
                volume_iface.SetMasterVolume(target_pct / 100.0, None)
            except Exception:
                pass

    def release(self) -> None:
        """Call when the assistant stops speaking a chunk. Restores after
        a debounce so back-to-back sentence chunks don't yo-yo the volume
        -- a fresh duck() before the debounce fires cancels this.
        """
        with self._lock:
            if not self._ducked:
                return
            if self._restore_timer:
                self._restore_timer.cancel()
            timer = threading.Timer(RESTORE_DEBOUNCE_S, self._do_restore)
            self._restore_timer = timer
        timer.start()

    def _do_restore(self) -> None:
        with self._lock:
            original = self._original_volume
            session = _find_spotify_session()
            if session is not None and original is not None:
                try:
                    session.SimpleAudioVolume.SetMasterVolume(original, None)
                except Exception:
                    pass
            self._ducked = False
            self._original_volume = None
            self._restore_timer = None
