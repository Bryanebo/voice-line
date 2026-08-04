"""Entry point: the turn loop and hold-to-talk wiring.

THE crash to avoid on Windows: the default ProactorEventLoop (required
here for the Claude Agent SDK's subprocess handling) does not implement
asyncio.add_reader(), so any input code built on it raises
NotImplementedError at startup. Nothing in this file reads stdin or
keystrokes through add_reader -- typed input is read on a background
thread via msvcrt and fed into the loop with call_soon_threadsafe (see
TypedInputReader below), exactly like ptt.py does for the key listener.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import msvcrt
import re
import sys
import threading

import httpx
from pynput import keyboard

import brain as brain_mod
import ducking as ducking_mod
import ears
import mouth as mouth_mod
import ptt as ptt_mod
import signals

DEFAULT_ASSISTANT_CWD = r"C:\Users\bryan\Documents\ADAM"
GREETING = "Adam reporting for duty. What are we working on?"

STD_INPUT_HANDLE = -10
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
PASTE_ECHO_THRESHOLD = 60


# ---------------------------------------------------------------------------
# Console setup: enable VT input so Windows Terminal sends bracketed-paste
# markers, and bracketed paste mode itself.
# ---------------------------------------------------------------------------


def _enable_vt_input() -> None:
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_uint32()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_INPUT)
    except Exception:
        pass


def _set_bracketed_paste(enabled: bool) -> None:
    try:
        sys.stdout.write("\x1b[?2004h" if enabled else "\x1b[?2004l")
        sys.stdout.flush()
    except Exception:
        pass


_GUTTER_RE = re.compile(r"^[>|]\s?")


def _scrub_pasted_text(raw: str) -> str:
    cleaned = []
    for line in raw.splitlines():
        line = _GUTTER_RE.sub("", line.strip())
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


class TypedInputReader:
    """Typing is a first-class turn, not a side channel: completed lines
    go into the same asyncio.Queue the turn loop reads from, so a typed
    message is spoken aloud exactly like a held-key utterance, and typing
    while the assistant talks interrupts it just the same.
    """

    def __init__(
        self, loop: asyncio.AbstractEventLoop, line_queue: asyncio.Queue, on_key: callable
    ) -> None:
        self._loop = loop
        self._queue = line_queue
        self._on_key = on_key  # called on any keystroke -- drives interrupt-while-speaking
        self._line_buffer: list[str] = []
        self._in_paste = False
        self._paste_buffer: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _submit_line(self) -> None:
        text = "".join(self._line_buffer).strip()
        self._line_buffer = []
        sys.stdout.write("\n")
        sys.stdout.flush()
        if text:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    def _read_escape_sequence(self) -> str | None:
        ch = msvcrt.getwch()
        if ch != "[":
            return None
        digits = ""
        while True:
            ch = msvcrt.getwch()
            if ch.isdigit():
                digits += ch
            elif ch == "~":
                if digits == "200":
                    return "paste_start"
                if digits == "201":
                    return "paste_end"
                return None
            else:
                return None

    def _run(self) -> None:
        while True:
            ch = msvcrt.getwch()
            self._loop.call_soon_threadsafe(self._on_key)

            if ch == "\x1b":
                seq = self._read_escape_sequence()
                if seq == "paste_start":
                    self._in_paste = True
                    self._paste_buffer = []
                elif seq == "paste_end":
                    self._in_paste = False
                    raw = "".join(self._paste_buffer)
                    self._paste_buffer = []
                    cleaned = _scrub_pasted_text(raw)
                    if len(cleaned) > PASTE_ECHO_THRESHOLD:
                        sys.stdout.write(f"[pasted {len(cleaned)} chars]")
                    else:
                        sys.stdout.write(cleaned)
                    sys.stdout.flush()
                    self._line_buffer.extend(cleaned)
                continue

            if self._in_paste:
                self._paste_buffer.append(ch)
                continue

            if ch in ("\r", "\n"):
                self._submit_line()
                continue

            if ch in ("\x08", "\x7f"):
                if self._line_buffer:
                    self._line_buffer.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch == "\x03":
                continue  # Ctrl+C: default console handler raises KeyboardInterrupt

            self._line_buffer.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


class TurnState:
    def __init__(self) -> None:
        self.interrupted = False


def _key_from_name(name: str) -> keyboard.Key | keyboard.KeyCode:
    key = getattr(keyboard.Key, name, None)
    if key is not None:
        return key
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unrecognized key name: {name!r}")


async def main_async(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    _enable_vt_input()
    _set_bracketed_paste(True)

    ducking = ducking_mod.Ducking()
    mouth = mouth_mod.Mouth(voice=args.voice, ducking=ducking)
    brain = brain_mod.Brain(cwd=args.assistant_cwd)
    transcribe_client = httpx.AsyncClient()
    turn_state = TurnState()

    signals.set_state("idle")
    mouth.start()
    await brain.start()

    print("Adam is warming up...")
    warmup_task = asyncio.create_task(brain.warmup())
    await mouth.queue_sentence(GREETING)
    await mouth.wait_until_silent()
    await warmup_task
    signals.set_state("idle")
    print(f"Ready. Hold {args.key} to talk, or type a message. Say 'goodbye' to quit.")

    turn_queue: asyncio.Queue[str] = asyncio.Queue()

    def on_key_activity() -> None:
        # Any typed keystroke interrupts playback, same as a fresh PTT press.
        if mouth.is_speaking():
            turn_state.interrupted = True
            mouth.interrupt()
            asyncio.ensure_future(brain.interrupt())

    typed_reader = TypedInputReader(loop, turn_queue, on_key_activity)
    typed_reader.start()

    ears_obj = ears.Ears()
    open_mic_stop = threading.Event()
    open_mic_thread: threading.Thread | None = None

    def on_interrupt() -> None:
        turn_state.interrupted = True
        mouth.interrupt()
        asyncio.ensure_future(brain.interrupt())

    def on_open() -> None:
        signals.set_state("listening")
        ears_obj.start_capture()

    def on_close(keep: bool) -> None:
        wav_bytes = ears_obj.stop_capture(keep)
        if wav_bytes is None:
            signals.set_state("idle")
            return
        signals.set_state("thinking")
        asyncio.ensure_future(_transcribe_and_enqueue(wav_bytes))

    async def _transcribe_and_enqueue(wav_bytes: bytes) -> None:
        try:
            text = await ears.transcribe(wav_bytes, transcribe_client)
        except Exception as exc:
            print(f"[transcription error: {exc}]")
            signals.set_state("idle")
            return
        if text.strip():
            await turn_queue.put(text)
        else:
            signals.set_state("idle")

    def on_open_mic_utterance(wav_bytes: bytes) -> None:
        signals.set_state("thinking")
        asyncio.ensure_future(_transcribe_and_enqueue(wav_bytes))

    ptt = None
    if args.open_mic:
        open_mic_thread = threading.Thread(
            target=ears.run_open_mic,
            args=(loop, on_open_mic_utterance, mouth.is_speaking),
            kwargs={"stop_event": open_mic_stop},
            daemon=True,
        )
        open_mic_thread.start()
        print("Open-mic mode: listening continuously (webrtcvad endpointing).")
    else:
        ptt = ptt_mod.PushToTalk(key=_key_from_name(args.key))
        ptt.attach(loop, on_interrupt=on_interrupt, on_open=on_open, on_close=on_close)
        ptt.start()

    async def handle_user_text(text: str) -> bool:
        print(f"you> {text}")
        if brain_mod.is_quit_phrase(text):
            signals.set_state("thinking")
            await mouth.queue_sentence("Goodbye.")
            await mouth.wait_until_silent()
            signals.set_state("idle")
            return True

        turn_state.interrupted = False
        signals.set_state("thinking")

        async def on_sentence(sentence: str) -> None:
            print(f"adam> {sentence}")
            await mouth.queue_sentence(sentence)

        try:
            await brain.send_and_speak(
                text, on_sentence, should_stop=lambda: turn_state.interrupted
            )
        except Exception as exc:
            print(f"[brain error: {exc}]")
        await mouth.wait_until_silent()
        if not turn_state.interrupted:
            signals.set_state("idle")
        return False

    pending_turn = asyncio.ensure_future(turn_queue.get())
    try:
        while True:
            done, _ = await asyncio.wait({pending_turn}, return_when=asyncio.FIRST_COMPLETED)
            if pending_turn in done:
                text = pending_turn.result()
                pending_turn = asyncio.ensure_future(turn_queue.get())
                should_quit = await handle_user_text(text)
                if should_quit:
                    break
    finally:
        pending_turn.cancel()
        if args.open_mic:
            open_mic_stop.set()
            if open_mic_thread:
                open_mic_thread.join(timeout=2.0)
        elif ptt is not None:
            ptt.stop()
        _set_bracketed_paste(False)
        await mouth.stop()
        await brain.stop()
        await transcribe_client.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="voice-line: talk to Adam out loud")
    parser.add_argument("--key", default="f9", help="hold-to-talk key name (default: f9)")
    parser.add_argument("--voice", default="bm_lewis", help="Kokoro voice id (default: bm_lewis)")
    parser.add_argument(
        "--assistant-cwd",
        default=DEFAULT_ASSISTANT_CWD,
        help="project folder whose CLAUDE.md defines Adam's identity",
    )
    parser.add_argument(
        "--open-mic",
        action="store_true",
        help="legacy always-listening mode with VAD endpointing, instead of hold-to-talk",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")


if __name__ == "__main__":
    main()
