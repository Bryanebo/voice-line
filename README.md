# voice-line

Talk to Adam out loud, hands on the keyboard, half-duplex, Windows-native.

```
mic -> ears (sounddevice capture)
    -> local whisper.cpp server on port 2022
    -> warm Claude Agent SDK session (streaming, one client per session)
    -> mouth (sentence-chunked Kokoro TTS on port 8880, cancellable playback)
    -> speakers
```

## Prerequisites (once per machine)

Two local servers must be running before you launch voice-line:

- **whisper.cpp** (`C:\Users\bryan\whisper-cpp`) — CUDA build, `ggml-small.en.bin` model, serving `/inference` on port 2022.
- **Kokoro-FastAPI** (`C:\Users\bryan\kokoro-fastapi`) — CUDA torch, serving `/v1/audio/speech` on port 8880.

Start them (each in its own terminal, left running):

```powershell
# Whisper
cd C:\Users\bryan\whisper-cpp\bin\Release
.\whisper-server.exe -m ..\..\models\ggml-small.en.bin --port 2022 --host 127.0.0.1

# Kokoro
cd C:\Users\bryan\kokoro-fastapi
.\start-gpu.ps1
```

`start-gpu.ps1` re-runs `uv pip install -e ".[gpu]"` on every launch, which silently resolves to CPU-only torch on this machine (known uv/pyproject quirk — see Gotchas). After it starts, always confirm GPU is active before trusting it:

```powershell
cd C:\Users\bryan\kokoro-fastapi
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False` or a version without `+cu126`, re-pin the wheel:

```powershell
uv pip install "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

Want these to survive reboots instead of babysitting two terminals? Install them as NSSM services (see spec — Administrator required, one-time setup). voice-line itself always stays a foreground app you launch on demand; never run it as a service.

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
