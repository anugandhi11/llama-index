#!/usr/bin/env python3
"""Companion AI — conversational loop.

Ties together the cloud LLM (the brain), the local GPU voice engine (the
voice), the personality engine (the vibe), and long/short-term memory (the
continuity) into one companion you can talk to by voice or text.

Quick start (run from the repository root, one level up from this folder):
    pip install -r companion_ai/requirements.txt
    cp companion_ai/.env.example companion_ai/.env   # fill in a free API key
    python -m companion_ai.main                      # voice mode
    python -m companion_ai.main --text-only           # type instead of talking

See companion_ai/README.md for full setup instructions (CUDA/PyTorch, model
downloads, mic/speaker setup, and getting a free API key).
"""

from __future__ import annotations

import argparse
import sys

from companion_ai.config import Config, ConfigError, get_config
from companion_ai.llm_client import LLMError, generate_reply
from companion_ai.memory import ConversationHistory, MemoryStore
from companion_ai.personality import EmotionalState, build_system_prompt, debotify
from companion_ai.voice_engine import (
    AudioPlayer,
    MicrophoneRecorder,
    SpeechToText,
    TextToSpeech,
    VoiceEngineError,
)

_EXIT_WORDS = {"quit", "exit", "bye", "goodbye", "good night", "night"}


class Companion:
    """Wires the brain, voice, memory, and personality together for one session."""

    def __init__(self, cfg: Config, *, voice_in: bool, voice_out: bool) -> None:
        self.cfg = cfg
        self.voice_in = voice_in
        self.voice_out = voice_out

        self.history = ConversationHistory(max_turns=cfg.max_history_turns)
        self.memory = MemoryStore(cfg)
        self.emotional_state = EmotionalState()

        self._stt: SpeechToText | None = SpeechToText(cfg) if voice_in else None
        self._recorder: MicrophoneRecorder | None = MicrophoneRecorder(cfg) if voice_in else None
        self._tts: TextToSpeech | None = TextToSpeech(cfg) if voice_out else None
        self._player: AudioPlayer | None = AudioPlayer(cfg) if voice_out else None

    def greet(self) -> str:
        greeting = f"Hey {self.cfg.user_name}, {self.cfg.companion_name} here. What's going on?"
        self._emit(greeting)
        return greeting

    def get_user_input(self) -> str:
        if self.voice_in and self._stt and self._recorder:
            print("[listening...]")
            try:
                text = self._stt.listen_and_transcribe(self._recorder)
            except VoiceEngineError as exc:
                print(
                    f"[voice_engine] {exc}\nFalling back to typed input for this turn.",
                    file=sys.stderr,
                )
                text = input(f"{self.cfg.user_name}> ").strip()
            if not text:
                print("[didn't catch that]")
            return text
        return input(f"{self.cfg.user_name}> ").strip()

    def step(self, user_text: str) -> str:
        """Run one conversational turn and return the companion's reply."""
        self.emotional_state.observe(user_text)
        self.memory.remember_if_notable(user_text)

        relevant_memories = self.memory.retrieve_relevant(user_text)
        memory_context = "\n".join(f"- {m}" for m in relevant_memories)
        system_prompt = build_system_prompt(self.cfg, self.emotional_state, memory_context)

        self.history.add("user", user_text)
        messages = [{"role": "system", "content": system_prompt}, *self.history.as_messages()]

        try:
            raw_reply = generate_reply(self.cfg, messages)
        except LLMError as exc:
            print(f"[llm_client] {exc}", file=sys.stderr)
            reply = (
                "Hit a snag talking to the cloud brain just now — give me a "
                "second and try that again?"
            )
            self._emit(reply)
            return reply

        reply = debotify(raw_reply)
        self.history.add("assistant", reply)
        if self.emotional_state.should_check_in():
            self.emotional_state.mark_checked_in()

        self._emit(reply)
        return reply

    def _emit(self, text: str) -> None:
        print(f"{self.cfg.companion_name}> {text}")
        if self.voice_out and self._tts and self._player:
            try:
                self._tts.speak(text, self._player)
            except VoiceEngineError as exc:
                print(f"[voice_engine] Couldn't speak that aloud: {exc}", file=sys.stderr)

    def run_forever(self) -> None:
        self.greet()
        while True:
            try:
                user_text = self.get_user_input()
            except (EOFError, KeyboardInterrupt):
                self._emit(f"Catch you later, {self.cfg.user_name}.")
                break

            if not user_text:
                continue
            if user_text.strip().lower() in _EXIT_WORDS:
                self._emit(f"Catch you later, {self.cfg.user_name}.")
                break

            self.step(user_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Companion AI conversational loop.")
    parser.add_argument(
        "--text-only", action="store_true", help="Disable mic input and speaker output; type instead."
    )
    parser.add_argument(
        "--text-in", action="store_true", help="Type input, but still speak replies aloud."
    )
    parser.add_argument(
        "--text-out", action="store_true", help="Use the mic, but print replies instead of speaking."
    )
    parser.add_argument(
        "--list-audio-devices", action="store_true", help="List available audio devices and exit."
    )
    return parser.parse_args(argv)


def _list_audio_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_audio_devices:
        _list_audio_devices()
        return 0

    try:
        cfg = get_config()
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 1

    voice_in = not (args.text_only or args.text_in)
    voice_out = not (args.text_only or args.text_out)

    companion = Companion(cfg, voice_in=voice_in, voice_out=voice_out)
    print(
        f"--- {cfg.companion_name} online | provider={cfg.llm_provider.value} | "
        f"device={cfg.device} | voice_in={voice_in} voice_out={voice_out} ---"
    )
    try:
        companion.run_forever()
    except Exception as exc:  # last-resort guard so the loop never crashes silently
        print(f"Unexpected error, shutting down: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
