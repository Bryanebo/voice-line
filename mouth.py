"""TTS queue and playback.

CRITICAL pipeline shape (proven on Windows): synthesis and playback run
as a two-stage pipeline on separate execution contexts (an asyncio task
for synthesis, a dedicated OS thread for playback) connected by a
thread-safe queue, so synthesis of the NEXT sentence overlaps playback of
the CURRENT one. A naive single-queue synthesize-then-play-per-sentence
loop produces a real, audible dead-air gap at every sentence boundary.

The playback thread opens ONE sounddevice OutputStream for the whole
session (not one per sentence) so there's no device-reopen latency at
sentence boundaries either.
"""

from __future__ import annotations

import asyncio
import queue
import threading

import httpx
import numpy as np
import sounddevice as sd

import signals

KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"
SAMPLE_RATE = 24000
CHANNELS = 1
PLAYBACK_BLOCK = 480  # 20ms at 24kHz -- how finely interrupt() can cut in


class Mouth:
    def __init__(self, voice: str = "bm_lewis", ducking=None) -> None:
        self.voice = voice
        self.ducking = ducking
        self._client = httpx.AsyncClient()
        self._text_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._play_queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
            maxsize=8
        )
        self._cancel = threading.Event()
        self._turn_id = 0
        self._turn_lock = threading.Lock()

        self._stream: sd.OutputStream | None = None
        self._playback_thread: threading.Thread | None = None
        self._synth_task: asyncio.Task | None = None
        self._pending_sentences = 0  # bumped on queue_sentence, dropped when played/discarded
        self._pending_lock = threading.Lock()
        self._drain_event = threading.Event()
        self._drain_event.set()

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
        )
        self._stream.start()
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()
        self._synth_task = asyncio.create_task(self._synth_worker())

    async def stop(self) -> None:
        if self._synth_task:
            self._synth_task.cancel()
        self._play_queue.put(None)
        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
        if self._stream:
            self._stream.stop()
            self._stream.close()
        await self._client.aclose()

    def current_turn_id(self) -> int:
        with self._turn_lock:
            return self._turn_id

    def is_speaking(self) -> bool:
        """True while any sentence for the current turn is queued,
        synthesizing, or playing. Used for the half-duplex mic gate in
        open-mic mode and to decide whether a keystroke should interrupt.
        """
        return not self._drain_event.is_set()

    async def queue_sentence(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        with self._pending_lock:
            self._pending_sentences += 1
            self._drain_event.clear()
        await self._text_queue.put((self.current_turn_id(), sentence))

    async def wait_until_silent(self) -> None:
        """Await until every queued sentence has been synthesized AND
        finished playing. Call this after the brain has finished
        streaming a turn, before signaling state=idle.
        """
        while not self._drain_event.is_set():
            await asyncio.sleep(0.02)

    def interrupt(self) -> None:
        """Cancel = clear queue + stop playback now."""
        with self._turn_lock:
            self._turn_id += 1
        self._cancel.set()
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        drained = 0
        while True:
            try:
                self._play_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        with self._pending_lock:
            self._pending_sentences = 0
            self._drain_event.set()
        signals.set_state("idle")

    async def _synth_worker(self) -> None:
        while True:
            turn_id, sentence = await self._text_queue.get()
            if turn_id != self.current_turn_id():
                self._mark_one_done()
                continue
            try:
                pcm_bytes = await self._synthesize(sentence)
            except Exception:
                self._mark_one_done()
                continue
            if turn_id != self.current_turn_id():
                self._mark_one_done()
                continue
            audio = np.frombuffer(pcm_bytes, dtype=np.int16)
            await asyncio.to_thread(self._play_queue.put, (turn_id, audio))

    async def _synthesize(self, sentence: str) -> bytes:
        resp = await self._client.post(
            KOKORO_URL,
            json={
                "model": "kokoro",
                "input": sentence,
                "voice": self.voice,
                "response_format": "pcm",
                "stream": False,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.content

    def _mark_one_done(self) -> None:
        with self._pending_lock:
            self._pending_sentences = max(0, self._pending_sentences - 1)
            if self._pending_sentences == 0:
                self._drain_event.set()

    def _playback_loop(self) -> None:
        while True:
            item = self._play_queue.get()
            if item is None:
                return
            turn_id, audio = item
            if turn_id != self.current_turn_id():
                self._mark_one_done()
                continue
            self._cancel.clear()
            if self.ducking:
                self.ducking.duck()
            for i in range(0, len(audio), PLAYBACK_BLOCK):
                if self._cancel.is_set() or turn_id != self.current_turn_id():
                    break
                chunk = audio[i : i + PLAYBACK_BLOCK]
                assert self._stream is not None
                self._stream.write(chunk.reshape(-1, 1))
                signals.write_waveform(chunk)
            if self.ducking:
                self.ducking.release()
            self._mark_one_done()
