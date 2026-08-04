"""Global hold-to-talk key listener.

pynput's global listener works out of the box on a normal Windows console
process (no special OS permission gate) -- just make sure microphone access
is allowed for desktop apps under Windows Settings > Privacy & security >
Microphone.

CRITICAL: Windows fires OS key-repeat on_press events continuously while a
key is held down. Without the _held flag below, every repeat looks like a
fresh press and kills the in-flight turn before it ever speaks.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

from pynput import keyboard

TAP_MIN_S = 0.25
RELEASE_TAIL_S = 0.18


class PushToTalk:
    def __init__(self, key: keyboard.Key = keyboard.Key.f9):
        self.key = key
        self._held = False
        self._press_time = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_interrupt: Callable[[], None] | None = None
        self._on_open: Callable[[], None] | None = None
        self._on_close: Callable[[bool], None] | None = None
        self._listener: keyboard.Listener | None = None
        self._release_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def attach(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        on_interrupt: Callable[[], None],
        on_open: Callable[[], None],
        on_close: Callable[[bool], None],
    ) -> None:
        """on_interrupt: fires immediately on any real press (barge-in).
        on_open: fires immediately on any real press (mic should open).
        on_close(keep): fires RELEASE_TAIL_S after a real release that was
        held >= TAP_MIN_S (keep=True), or immediately for a too-short tap
        (keep=False, discard whatever was captured).
        """
        self._loop = loop
        self._on_interrupt = on_interrupt
        self._on_open = on_open
        self._on_close = on_close

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._raw_press, on_release=self._raw_release
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        with self._lock:
            if self._release_timer:
                self._release_timer.cancel()
                self._release_timer = None

    def _raw_press(self, key: keyboard.Key) -> None:
        if key != self.key:
            return
        with self._lock:
            if self._held:
                return  # OS key-repeat while held -- not a fresh press
            self._held = True
            self._press_time = time.monotonic()
            if self._release_timer:
                self._release_timer.cancel()
                self._release_timer = None
        if self._loop is None:
            return
        if self._on_interrupt:
            self._loop.call_soon_threadsafe(self._on_interrupt)
        if self._on_open:
            self._loop.call_soon_threadsafe(self._on_open)

    def _raw_release(self, key: keyboard.Key) -> None:
        if key != self.key:
            return
        with self._lock:
            if not self._held:
                return
            self._held = False
            held_duration = time.monotonic() - self._press_time

        if self._loop is None or self._on_close is None:
            return

        if held_duration < TAP_MIN_S:
            self._loop.call_soon_threadsafe(self._on_close, False)
            return

        def _fire_close() -> None:
            assert self._loop is not None and self._on_close is not None
            self._loop.call_soon_threadsafe(self._on_close, True)

        timer = threading.Timer(RELEASE_TAIL_S, _fire_close)
        with self._lock:
            self._release_timer = timer
        timer.start()
