# voice-line

Talk to Adam out loud, hands on the keyboard, half-duplex, Windows-native.

## What this is

A hold-to-talk voice interface for talking to Claude Code out loud instead of typing — hold a key anywhere on Windows, speak, release, and hear a spoken reply back through your speakers, first audio landing in 1–2 seconds on a warm turn. Built from a detailed technical spec covering the real failure modes of building this on Windows: the default asyncio event loop can't read keyboard input the normal way, TTS needs a two-stage pipeline or every sentence has an audible gap, and a GPU-accelerated dependency can silently fall back to CPU with nothing telling you.

**Stack:** Python (`asyncio`, `sounddevice`, `pynput`), a local whisper.cpp server (CUDA) for transcription, the Claude Agent SDK for a warm streaming session, and a local Kokoro-FastAPI server (CUDA) for sentence-chunked, cancellable text-to-speech. Both backing servers run as Windows services (NSSM) — auto-start, auto-restart on crash.

**What it demonstrates:** end-to-end systems integration across four independent processes (mic capture, ASR, LLM, TTS) talking over local HTTP; diagnosing and fixing a real silent-GPU-fallback bug (`torch.cuda.is_available()` returning `False` despite a "GPU install" script) by tracing it to a dependency-resolution quirk and pinning the correct CUDA wheel directly; Windows service administration (NSSM, `LocalSystem` account quirks, elevated installs); and building for a documented spec's hard constraints rather than the easy path.

```
mic -> ears (sounddevice capture)
    -> local whisper.cpp server on port 2022
    -> warm Claude Agent SDK session (streaming, one client per session)
    -> mouth (sentence-chunked Kokoro TTS on port 8880, cancellable playback)
    -> speakers
```

## Prerequisites (once per machine)

Two local servers must be running before you launch voice-line, and both are installed as Windows services (NSSM) — auto-start on boot, auto-restart on crash, no terminals to babysit:

- **WhisperCpp** service — `C:\Users\bryan\whisper-cpp`, CUDA build, `ggml-small.en.bin` model, serving `/inference` on port 2022. Launched via `run-service.bat` in that folder.
- **KokoroFastAPI** service — `C:\Users\bryan\kokoro-fastapi`, CUDA torch, serving `/v1/audio/speech` on port 8880. Launched via `run-service.bat` in that folder, which calls the venv's `python.exe -m uvicorn` directly rather than `uv run` (LocalSystem doesn't inherit the user PATH `uv` lives on) and skips `start-gpu.ps1`'s `uv pip install -e ".[gpu]"` (that command silently re-resolves to CPU-only torch every time it runs — see Gotchas).

Manage them like any Windows service:

```powershell
Get-Service WhisperCpp, KokoroFastAPI
Restart-Service WhisperCpp
Restart-Service KokoroFastAPI
```

Logs land in `<project>\logs\stdout.log` / `stderr.log` in each folder — check there first if a service won't come up. After any change to Kokoro's torch install, re-verify GPU is actually active (silent CPU fallback, nothing tells you otherwise):

```powershell
cd C:\Users\bryan\kokoro-fastapi
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False` or a version without `+cu126`, re-pin the wheel:

```powershell
uv pip install "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

voice-line itself always stays a foreground app you launch on demand; it is intentionally NOT a service — nobody wants a 24/7 open mic.

## Launch

```
C:\Users\bryan\voice-line\run-voice-line.bat
```

Optional flags: `--key f9` (hold-to-talk key), `--voice bm_lewis` (Kokoro voice id), `--open-mic` (legacy always-listening mode instead of hold-to-talk), `--assistant-cwd <path>` (which CLAUDE.md defines Adam's identity).

## Controls

| Action | How |
|---|---|
| Talk | Hold **F9**, speak, release. Reply plays back automatically. |
| Interrupt | Press F9 again (or start typing) while Adam is talking — playback stops immediately. |
| Type instead | Just type a line and hit Enter — same turn loop, spoken reply either way. |
| Paste | Paste normally; long pastes echo as `[pasted N chars]` instead of dumping the text. |
| End session | Say or type "goodbye", "end voice mode", or "hang up". Ctrl+C also works. |
| Quiet tap | Holding F9 for less than 250ms is ignored (no accidental triggers). |

**Notes:**
- Half-duplex: the mic is gated while Adam is speaking, so it never hears its own voice through the speakers.
- Nothing is captured between holds — releasing F9 fully closes the mic stream.
- If Spotify is playing above 30% volume, it ducks automatically while Adam talks and restores after a short debounce.

## Gotchas (hard-won, don't relitigate these)

- Windows' ProactorEventLoop (required for the Agent SDK's subprocess handling) has no `asyncio.add_reader()`. All keyboard/typed input is read on background threads (`msvcrt` for typed lines, `pynput` for the global hold key) and fed into the loop with `call_soon_threadsafe`.
- OS key-repeat fires `on_press` continuously while a key is held — `ptt.py` filters this with a held-state flag.
- whisper.cpp's server here only answers `/inference`, not the OpenAI-style `/v1/audio/transcriptions` — confirmed with curl, don't "fix" this without re-testing.
- TTS synthesis and playback run as a two-stage pipeline (separate queues) so the next sentence synthesizes while the current one plays — a naive single-queue loop produces audible dead air between sentences.
- `content_block_stop` always flushes the sentence buffer, punctuation or not, so pre-tool filler ("checking now...") doesn't sit silent through a tool call.
- Kokoro's own `start-gpu.ps1` (and even a plain `uv sync --extra gpu`) can silently resolve CPU-only torch — always verify `torch.cuda.is_available()` after starting it, never assume.
