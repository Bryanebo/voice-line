"""Mic capture and transcription.

Hold-to-talk (default): buffer raw frames between start_capture() and
stop_capture(); the tail before stop_capture() is called is owned by
ptt.py (RELEASE_TAIL_S), not here -- by the time stop_capture(keep=True)
fires, the last word has already been captured.

Open-mic (--open-mic): webrtcvad endpointing on a background thread.

GOTCHA (from the spec): whisper.cpp server builds differ in which route
they expose. This build only answers on /inference, not the OpenAI-style
/v1/audio/transcriptions -- confirmed with curl before writing this
client. Do not "fix" this to the OpenAI route without re-testing.
"""

from __future__ import annotations

import io
import queue
import re
import threading
import wave
from collections.abc import Callable

import httpx
import numpy as np
import sounddevice as sd
import webrtcvad

WHISPER_URL = "http://127.0.0.1:2022/inference"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

_BRACKETED_MARKER_RE = re.compile(r"\[[^\]]*\]")


def _strip_non_speech(text: str) -> str:
    text = _BRACKETED_MARKER_RE.sub("", text)
    return " ".join(text.split())


def _frames_to_wav_bytes(frames: list[np.ndarray]) -> bytes | None:
    if not frames:
        return None
    audio = np.concatenate(frames)
    if len(audio) == 0:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


async def transcribe(wav_bytes: bytes, client: httpx.AsyncClient) -> str:
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"response_format": "json"}
    resp = await client.post(WHISPER_URL, files=files, data=data, timeout=30.0)
    resp.raise_for_status()
    text = resp.json().get("text", "")
    return _strip_non_speech(text)


class Ears:
    """Hold-to-talk capture. One InputStream opened per hold, fully closed
    on release -- room audio and music never leak into the transcriber
    between holds.
    """

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._frames.append(indata.copy().reshape(-1))

    def start_capture(self) -> None:
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()

    def stop_capture(self, keep: bool) -> bytes | None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            frames = self._frames
            self._frames = []
        if not keep:
            return None
        return _frames_to_wav_bytes(frames)


# ---------------------------------------------------------------------------
# Open-mic mode (--open-mic): continuous listening with VAD endpointing.
# ---------------------------------------------------------------------------

_VAD_FRAME_MS = 30
_VAD_FRAME_SAMPLES = SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 480 samples
_TRAILING_SILENCE_MS = 600
_MIN_SPEECH_MS = 240


def run_open_mic(
    loop,
    on_utterance: Callable[[bytes], None],
    is_speaking: Callable[[], bool],
    vad_aggressiveness: int = 2,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocking; run this on its own thread. Half-duplex: while
    is_speaking() is true, frames are dropped and any in-progress
    utterance is abandoned, so the mic never hears the speakers.
    """
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:
        frame_q.put(indata.copy().reshape(-1))

    stop_event = stop_event or threading.Event()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=_VAD_FRAME_SAMPLES,
        callback=callback,
    ):
        utterance_frames: list[np.ndarray] = []
        speech_ms = 0
        trailing_silence_ms = 0
        speech_started = False

        while not stop_event.is_set():
            try:
                frame = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if is_speaking():
                # Half-duplex gate: never hear our own speakers.
                utterance_frames = []
                speech_ms = 0
                trailing_silence_ms = 0
                speech_started = False
                continue

            if len(frame) != _VAD_FRAME_SAMPLES:
                continue

            is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)

            if is_speech:
                utterance_frames.append(frame)
                speech_ms += _VAD_FRAME_MS
                trailing_silence_ms = 0
                speech_started = True
                continue

            if not speech_started:
                continue  # pre-speech silence, don't buffer indefinitely

            utterance_frames.append(frame)
            trailing_silence_ms += _VAD_FRAME_MS
            if trailing_silence_ms < _TRAILING_SILENCE_MS:
                continue

            # Utterance ended.
            if speech_ms >= _MIN_SPEECH_MS:
                wav_bytes = _frames_to_wav_bytes(utterance_frames)
                if wav_bytes is not None:
                    loop.call_soon_threadsafe(on_utterance, wav_bytes)
            utterance_frames = []
            speech_ms = 0
            trailing_silence_ms = 0
            speech_started = False
