"""Cloud LLM client — the companion's "brain".

Supports three free-tier-friendly backends behind one interface:
- Google AI Studio (Gemini) via the official `google-generativeai` SDK
- Groq via its OpenAI-compatible endpoint (ultra-low latency, free tier)
- Any other OpenAI-compatible endpoint (LM Studio, OpenRouter, vLLM, etc.)

`generate_reply()` is the only entry point main.py needs to know about.
"""

from __future__ import annotations

import time

from companion_ai.config import Config, LLMProvider


class LLMError(RuntimeError):
    """Raised when the cloud LLM call ultimately fails after retries."""


def generate_reply(cfg: Config, messages: list[dict[str, str]], retries: int = 2) -> str:
    """Send `messages` (OpenAI-style role/content dicts) and return the reply text."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if cfg.llm_provider is LLMProvider.GEMINI:
                return _call_gemini(cfg, messages)
            if cfg.llm_provider is LLMProvider.GROQ:
                return _call_openai_compatible(
                    messages,
                    api_key=cfg.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    model=cfg.groq_model,
                    cfg=cfg,
                )
            if cfg.llm_provider is LLMProvider.OPENAI_COMPATIBLE:
                return _call_openai_compatible(
                    messages,
                    api_key=cfg.openai_compatible_api_key,
                    base_url=cfg.openai_compatible_base_url,
                    model=cfg.openai_compatible_model,
                    cfg=cfg,
                )
            raise LLMError(f"Unsupported LLM provider: {cfg.llm_provider}")
        except Exception as exc:  # re-raised as LLMError below once retries are exhausted
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue

    raise LLMError(f"LLM call failed after {retries + 1} attempt(s): {last_error}") from last_error


def _call_gemini(cfg: Config, messages: list[dict[str, str]]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=cfg.gemini_api_key)

    system_prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    history = _to_gemini_history(m for m in messages if m["role"] != "system")
    if not history:
        raise LLMError("No user message to send to Gemini.")

    model = genai.GenerativeModel(
        model_name=cfg.gemini_model, system_instruction=system_prompt or None
    )
    generation_config = genai.types.GenerationConfig(
        temperature=cfg.llm_temperature,
        max_output_tokens=cfg.llm_max_output_tokens,
    )

    *prior_turns, last_turn = history
    chat = model.start_chat(history=prior_turns)
    response = chat.send_message(
        last_turn["parts"],
        generation_config=generation_config,
        request_options={"timeout": cfg.llm_request_timeout},
    )
    text = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text


def _to_gemini_history(messages) -> list[dict]:
    role_map = {"user": "user", "assistant": "model"}
    history = []
    for message in messages:
        role = role_map.get(message["role"])
        if role is None:
            continue
        history.append({"role": role, "parts": [message["content"]]})
    return history


def _call_openai_compatible(
    messages: list[dict[str, str]],
    *,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    cfg: Config,
) -> str:
    from openai import OpenAI

    if not api_key:
        raise LLMError(f"No API key configured for endpoint {base_url}.")
    if not base_url or not model:
        raise LLMError("base_url and model must both be set for an OpenAI-compatible endpoint.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=cfg.llm_request_timeout)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_output_tokens,
    )
    text = response.choices[0].message.content
    if not text:
        raise LLMError("LLM endpoint returned an empty response.")
    return text
