"""Local, GPU-accelerated voice engine: speech-to-text and text-to-speech.

Runs entirely on local hardware (tuned for an NVIDIA RTX 5070 Ti) so voice
turns don't pay cloud STT/TTS latency or cost — only the text reasoning step
goes to the cloud LLM. STT is faster-whisper (CTranslate2); TTS is
Kokoro-82M, a small, fast, Apache-2.0 neural TTS model that runs comfortably
on a single consumer GPU. Both fall back to CPU automatically if CUDA isn't
available or a GPU load fails (e.g. out-of-memory).
"""

from __future__ import annotations

import queue
import sys

import numpy as np

from companion_ai.config import Config


class VoiceEngineError(RuntimeError):
    """Raised for unrecoverable voice pipeline failures."""


class SpeechToText:
    """Local speech-to-text using faster-whisper."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self._cfg.whisper_model_size,
                device=self._cfg.device,
                compute_type=self._cfg.whisper_compute_type,
            )
        except Exception as exc:  # e.g. CUDA OOM or missing cuDNN
            if self._cfg.device == "cuda":
                print(
                    f"[voice_engine] Whisper failed to load on CUDA ({exc}); "
                    "falling back to CPU/int8.",
                    file=sys.stderr,
                )
                self._model = WhisperModel(self._cfg.whisper_model_size, device="cpu", compute_type="int8")
            else:
                raise VoiceEngineError(f"Failed to load Whisper model: {exc}") from exc

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 PCM buffer at `sample_rate` Hz."""
        self._ensure_loaded()
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        try:
            segments, _info = self._model.transcribe(audio, language="en", vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise VoiceEngineError(f"Speech-to-text failed: {exc}") from exc

    def listen_and_transcribe(self, recorder: "MicrophoneRecorder") -> str:
        audio = recorder.record_utterance()
        if audio.size == 0:
            return ""
        return self.transcribe(audio, recorder.sample_rate)


class TextToSpeech:
    """Local text-to-speech using Kokoro-82M, GPU-accelerated when available."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from kokoro import KPipeline

        try:
            self._pipeline = KPipeline(lang_code="a", device=self._cfg.device)
        except Exception as exc:
            if self._cfg.device == "cuda":
                print(
                    f"[voice_engine] Kokoro failed to load on CUDA ({exc}); "
                    "falling back to CPU.",
                    file=sys.stderr,
                )
                self._pipeline = KPipeline(lang_code="a", device="cpu")
            else:
                raise VoiceEngineError(f"Failed to load Kokoro TTS model: {exc}") from exc

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 PCM audio at cfg.sample_rate for `text`."""
        self._ensure_loaded()
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        try:
            chunks = [
                np.asarray(audio, dtype=np.float32)
                for _graphemes, _phonemes, audio in self._pipeline(
                    text, voice=self._cfg.tts_voice, speed=self._cfg.tts_speed
                )
            ]
        except Exception as exc:
            raise VoiceEngineError(f"Text-to-speech synthesis failed: {exc}") from exc

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def speak(self, text: str, player: "AudioPlayer") -> None:
        audio = self.synthesize(text)
        if audio.size:
            player.play(audio, self._cfg.sample_rate)


class MicrophoneRecorder:
    """Hands-free microphone capture with energy-based silence detection."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.sample_rate = cfg.mic_sample_rate

    def record_utterance(self) -> np.ndarray:
        """Record from the mic until the user pauses, then return the audio."""
        import sounddevice as sd

        cfg = self._cfg
        block_seconds = 0.1
        block_size = int(cfg.mic_sample_rate * block_seconds)
        silence_blocks_needed = max(1, int(cfg.silence_duration_s / block_seconds))
        max_blocks = max(1, int(cfg.max_utterance_s / block_seconds))

        audio_queue: queue.Queue = queue.Queue()

        def _callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"[voice_engine] mic status: {status}", file=sys.stderr)
            audio_queue.put(indata[:, 0].copy())

        frames: list[np.ndarray] = []
        silence_run = 0
        heard_speech = False

        try:
            with sd.InputStream(
                samplerate=cfg.mic_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=cfg.input_device,
                callback=_callback,
            ):
                for _ in range(max_blocks):
                    block = audio_queue.get()
                    frames.append(block)
                    rms = float(np.sqrt(np.mean(np.square(block)) + 1e-12))
                    if rms >= cfg.silence_threshold:
                        heard_speech = True
                        silence_run = 0
                    elif heard_speech:
                        silence_run += 1
                        if silence_run >= silence_blocks_needed:
                            break
        except Exception as exc:
            raise VoiceEngineError(f"Microphone capture failed: {exc}") from exc

        if not heard_speech:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32)


class AudioPlayer:
    """Plays synthesized speech through the default (or configured) output device."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        try:
            sd.play(audio, samplerate=sample_rate, device=self._cfg.output_device)
            sd.wait()
        except Exception as exc:
            raise VoiceEngineError(f"Audio playback failed: {exc}") from exc


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Minimal dependency-free linear resampler (good enough for speech)."""
    if orig_sr == target_sr or audio.size == 0:
        return audio
    duration = audio.shape[0] / orig_sr
    target_len = int(round(duration * target_sr))
    orig_x = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
    target_x = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, orig_x, audio).astype(np.float32)
