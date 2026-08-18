"""Configuration management for the companion AI app.

Centralizes environment variables, free-tier cloud LLM credentials, and
local GPU device flags used by the voice engine. Call `get_config()` to
obtain a validated, cached `Config` instance; everything else in the app
reads settings from there instead of touching `os.environ` directly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI_COMPATIBLE = "openai_compatible"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _detect_device(preference: str) -> str:
    """Resolve 'auto'/'cuda'/'cpu' against actual hardware availability."""
    preference = (preference or "auto").strip().lower()
    if preference == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if preference == "cuda":
            print(
                "[config] WARNING: DEVICE=cuda requested but PyTorch is not "
                "installed. Falling back to CPU. Install a CUDA build of "
                "PyTorch to use the RTX GPU for STT/TTS.",
                file=sys.stderr,
            )
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if preference == "cuda":
        print(
            "[config] WARNING: DEVICE=cuda requested but no CUDA GPU was "
            "detected. Falling back to CPU (voice pipeline will be slower).",
            file=sys.stderr,
        )
    return "cpu"


@dataclass
class Config:
    # --- Identity -------------------------------------------------------
    companion_name: str = "Atlas"
    user_name: str = "you"

    # --- Cloud LLM (the brain) ------------------------------------------
    llm_provider: LLMProvider = LLMProvider.GEMINI
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_model: str | None = None
    llm_temperature: float = 0.9
    llm_max_output_tokens: int = 512
    llm_request_timeout: float = 30.0

    # --- Local GPU voice engine ------------------------------------------
    device: str = "cpu"  # resolved from DEVICE env var at load time
    whisper_model_size: str = "small.en"
    whisper_compute_type: str = "float16"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    sample_rate: int = 24000  # Kokoro TTS output rate
    mic_sample_rate: int = 16000  # Whisper's native input rate
    input_device: int | None = None
    output_device: int | None = None
    silence_threshold: float = 0.012
    silence_duration_s: float = 1.1
    max_utterance_s: float = 30.0

    # --- Memory -----------------------------------------------------------
    memory_dir: Path = field(default_factory=lambda: Path.home() / ".companion_ai" / "memory")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_history_turns: int = 12
    memory_top_k: int = 4
    memory_relevance_threshold: float = 0.35

    def validate(self) -> None:
        errors: list[str] = []

        if self.llm_provider is LLMProvider.GEMINI and not self.gemini_api_key:
            errors.append(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey, or switch "
                "LLM_PROVIDER to 'groq'."
            )
        elif self.llm_provider is LLMProvider.GROQ and not self.groq_api_key:
            errors.append(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys, or switch LLM_PROVIDER to "
                "'gemini'."
            )
        elif self.llm_provider is LLMProvider.OPENAI_COMPATIBLE:
            if not self.openai_compatible_base_url:
                errors.append(
                    "OPENAI_COMPATIBLE_BASE_URL is required when "
                    "LLM_PROVIDER=openai_compatible."
                )
            if not self.openai_compatible_model:
                errors.append(
                    "OPENAI_COMPATIBLE_MODEL is required when "
                    "LLM_PROVIDER=openai_compatible."
                )

        if not 0.0 <= self.llm_temperature <= 2.0:
            errors.append("LLM_TEMPERATURE must be between 0 and 2.")

        if self.device not in ("cuda", "cpu"):
            errors.append(f"Unsupported resolved device '{self.device}'.")

        if errors:
            raise ConfigError("Invalid configuration:\n- " + "\n- ".join(errors))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load, validate, and cache configuration from the environment.

    Cached because model/device detection and directory setup only need to
    happen once per process. Call `get_config.cache_clear()` first if you
    need to reload after mutating the environment (e.g. in tests).
    """
    if _DOTENV_AVAILABLE:
        load_dotenv(override=False)

    provider_raw = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    try:
        provider = LLMProvider(provider_raw)
    except ValueError as exc:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{provider_raw}'. Valid options: {valid}"
        ) from exc

    device = _detect_device(os.getenv("DEVICE", "auto"))
    whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    if device == "cpu" and whisper_compute_type == "float16":
        # float16 CTranslate2 kernels require CUDA; int8 is the fast CPU path.
        whisper_compute_type = "int8"

    cfg = Config(
        companion_name=os.getenv("COMPANION_NAME", "Atlas"),
        user_name=os.getenv("USER_NAME", "you"),
        llm_provider=provider,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        openai_compatible_model=os.getenv("OPENAI_COMPATIBLE_MODEL"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.9")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512")),
        llm_request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "30")),
        device=device,
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small.en"),
        whisper_compute_type=whisper_compute_type,
        tts_voice=os.getenv("TTS_VOICE", "af_heart"),
        tts_speed=float(os.getenv("TTS_SPEED", "1.0")),
        sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
        mic_sample_rate=int(os.getenv("MIC_SAMPLE_RATE", "16000")),
        input_device=_optional_int(os.getenv("INPUT_DEVICE")),
        output_device=_optional_int(os.getenv("OUTPUT_DEVICE")),
        silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.012")),
        silence_duration_s=float(os.getenv("SILENCE_DURATION_S", "1.1")),
        max_utterance_s=float(os.getenv("MAX_UTTERANCE_S", "30")),
        memory_dir=Path(
            os.getenv("MEMORY_DIR", str(Path.home() / ".companion_ai" / "memory"))
        ).expanduser(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "12")),
        memory_top_k=int(os.getenv("MEMORY_TOP_K", "4")),
        memory_relevance_threshold=float(
            os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.35")
        ),
    )

    cfg.validate()
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    return cfg
