"""The warm Claude Agent SDK session -- the same assistant as the terminal
session, reached through one persistent ClaudeSDKClient.

Sentence chunking watches the raw Anthropic stream events (via
include_partial_messages=True): text_delta events grow a buffer, which is
split into complete sentences and handed to the mouth immediately. After
the first sentence (shipped alone for low latency), sentences ship in
two-sentence breaths -- lone short sentences sound flat on TTS.

CRITICAL: content_block_stop always flushes whatever's left in the
buffer, punctuation or not. Without this, pre-tool filler like "On it,
checking now" sits silent through the whole tool run and then plays
glued to the answer.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, StreamEvent

SPOKEN_DISCIPLINE = """You are in a live spoken voice conversation, not a text chat. \
Speak in short, natural, conversational sentences -- write for the ear, not the eye. \
Never use markdown, code blocks, bullet points, numbered lists, or headers; if you'd \
normally show a list, say it as a flowing sentence instead. TTS performs your \
punctuation, so use periods, commas, and question marks to shape the delivery -- flat, \
unpunctuated text sounds flat out loud. Keep replies tight; this is a back-and-forth, \
not a report."""

QUIT_PHRASES = {"goodbye", "end voice mode", "hang up"}

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def is_quit_phrase(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?")
    return t in QUIT_PHRASES


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off complete sentences, returning (sentences, remainder)."""
    parts = _SENTENCE_END_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    return [s for s in complete if s.strip()], remainder


class Brain:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self._client: ClaudeSDKClient | None = None

    async def start(self) -> None:
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": SPOKEN_DISCIPLINE,
            },
            include_partial_messages=True,
        )
        self._client = ClaudeSDKClient(options)
        await self._client.connect()

    async def stop(self) -> None:
        if self._client:
            await self._client.disconnect()

    async def warmup(self) -> None:
        """Fire-and-drain a throwaway query so the prompt-cache toll (a
        few seconds on a cold session) happens while a spoken greeting is
        playing, not while the user is waiting on their real first turn.
        """
        assert self._client is not None
        await self._client.query(
            "(session warmup -- no reply needed, just get ready)"
        )
        async for _ in self._client.receive_response():
            pass

    async def interrupt(self) -> None:
        if self._client is not None:
            try:
                await self._client.interrupt()
            except Exception:
                pass

    async def send_and_speak(
        self,
        user_text: str,
        on_sentence: Callable[[str], Awaitable[None]],
        should_stop: Callable[[], bool],
    ) -> None:
        """Send user_text, stream the reply, and call on_sentence(text)
        for each chunk ready to speak -- first sentence alone, then
        two-sentence breaths, with an unconditional flush whenever a
        content block stops.
        """
        assert self._client is not None
        await self._client.query(user_text)

        buffer = ""
        carry = ""
        sentence_count = 0

        async def ship(s: str) -> None:
            nonlocal sentence_count, carry
            s = s.strip()
            if not s:
                return
            sentence_count += 1
            if sentence_count == 1:
                await on_sentence(s)
            elif carry:
                await on_sentence(carry + " " + s)
                carry = ""
            else:
                carry = s

        async def flush_carry() -> None:
            nonlocal carry
            if carry:
                await on_sentence(carry)
                carry = ""

        async for message in self._client.receive_response():
            if should_stop():
                await self.interrupt()
                break

            if isinstance(message, StreamEvent):
                event = message.event
                etype = event.get("type")

                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        buffer += delta.get("text", "")
                        sentences, buffer = _split_complete_sentences(buffer)
                        for s in sentences:
                            await ship(s)

                elif etype == "content_block_stop":
                    # Unconditional flush -- catches pre-tool filler that
                    # has no terminal punctuation yet.
                    if buffer.strip():
                        await ship(buffer)
                        buffer = ""
                    await flush_carry()

            elif isinstance(message, ResultMessage):
                if buffer.strip():
                    await ship(buffer)
                    buffer = ""
                await flush_carry()
                break
