# Companion AI

A free, hybrid-architecture personal AI companion: cloud LLM reasoning +
local GPU voice. No subscriptions, no paid APIs — everything below runs on
free tiers or open-source models you host yourself.

It's built to *feel* like J.A.R.V.I.S. (competent, always three steps
ahead, quietly handles things) crossed with Smith/Zoey from Netflix's
*Atlas* (warm, witty, blunt, genuinely checks in on you) — not a customer
support bot.

## Architecture

```
 your voice ──▶ MicrophoneRecorder ──▶ SpeechToText (faster-whisper, GPU)
                                              │
                                              ▼
                                        transcribed text
                                              │
                     ┌────────────────────────┼───────────────────────┐
                     ▼                        ▼                       ▼
              EmotionalState            MemoryStore              ConversationHistory
              (mood/rapport)        (long-term facts,           (short-term rolling
                     │              sentence-transformers)            window)
                     └────────────────────────┼───────────────────────┘
                                              ▼
                                  personality.build_system_prompt()
                                              │
                                              ▼
                                  llm_client.generate_reply()
                                  (Gemini / Groq / OpenAI-compatible)
                                              │
                                              ▼
                                   personality.debotify(reply)
                                              │
                                              ▼
                                  TextToSpeech (Kokoro-82M, GPU) ──▶ your speakers
```

- **The brain** (cloud): a free-tier LLM does the actual reasoning. Nothing
  about your hardware limits how smart the companion is.
- **The voice** (local, GPU): speech-to-text and text-to-speech run entirely
  on your machine, tuned for an NVIDIA RTX 5070 Ti, so voice turns are fast
  and don't leak audio to a third party.
- **The personality**: a persistent system prompt plus a lightweight,
  heuristic emotional-state tracker that nudges tone (more playful, more
  steady, more check-in-y) based on how the conversation is actually going.
- **The memory**: short-term (recent turns) and long-term (durable facts
  about you, persisted locally and retrieved by semantic similarity).

## Files

| File               | Responsibility                                             |
|--------------------|--------------------------------------------------------------|
| `config.py`        | Env var loading, validation, GPU device detection            |
| `personality.py`   | System prompt, filler-phrase filter, mood/rapport tracking   |
| `memory.py`        | Rolling chat history + persistent long-term memory store     |
| `llm_client.py`    | Unified client over Gemini / Groq / OpenAI-compatible APIs   |
| `voice_engine.py`  | Local GPU STT (faster-whisper) and TTS (Kokoro-82M)          |
| `main.py`          | Conversational loop that wires everything together           |

## Prerequisites

- Python 3.10+
- An NVIDIA GPU with a recent driver if you want GPU-accelerated voice (an
  RTX 5070 Ti comfortably runs both STT and TTS with room to spare; CPU
  works too, just slower). CUDA is *not* required for the cloud LLM.
- `ffmpeg` on your PATH (used by faster-whisper for audio decoding).
- PortAudio, for microphone/speaker access via `sounddevice`:
  - Debian/Ubuntu: `sudo apt install portaudio19-dev ffmpeg`
  - macOS: `brew install portaudio ffmpeg`
  - Windows: no extra step, PortAudio ships with the `sounddevice` wheel.

## Setup

1. **Install PyTorch first**, matching your GPU. For an RTX 5070 Ti on
   Linux/Windows with a recent driver:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   (No NVIDIA GPU, or don't want to bother with CUDA? Skip this — the next
   step installs a CPU build automatically.)

2. **Install the rest of the dependencies** (run from the repository root):
   ```bash
   pip install -r companion_ai/requirements.txt
   ```

3. **Get a free cloud LLM API key** — pick one:
   - **Gemini (default)**: create a free key at
     https://aistudio.google.com/app/apikey
   - **Groq** (fastest inference, great for voice): create a free key at
     https://console.groq.com/keys

4. **Configure**:
   ```bash
   cp companion_ai/.env.example companion_ai/.env
   ```
   Edit `companion_ai/.env` and set `GEMINI_API_KEY` (or switch
   `LLM_PROVIDER=groq` and set `GROQ_API_KEY`). Set `USER_NAME` and
   `COMPANION_NAME` if you want something other than the defaults.

5. **Run it** (from the repository root):
   ```bash
   python -m companion_ai.main
   ```
   Talk into your mic; the companion replies out loud. Say "bye" (or
   Ctrl-C) to end the session.

   Prefer typing, or don't have a mic/speakers handy?
   ```bash
   python -m companion_ai.main --text-only
   ```
   Mixed modes also work: `--text-in` (type, but hear replies) and
   `--text-out` (talk, but read replies).

## First run notes

- The first run downloads the Whisper and Kokoro model weights (a few
  hundred MB total) to your local Hugging Face cache; subsequent runs are
  instant. Combined GPU memory footprint is well under 2 GB, so it fits
  comfortably alongside other workloads.
- Long-term memory is written to `~/.companion_ai/memory/` by default
  (override with `MEMORY_DIR` in `.env`) as a plain JSONL file — delete it
  any time to wipe the companion's memory of you.

## Troubleshooting

- **`python -m companion_ai.main --list-audio-devices`** prints every input
  and output device with its index. If the wrong mic/speakers get picked,
  set `INPUT_DEVICE` / `OUTPUT_DEVICE` in `.env` to the index you want.
- **Companion doesn't hear you / cuts you off too early**: adjust
  `SILENCE_THRESHOLD` (lower = more sensitive to quiet speech) and
  `SILENCE_DURATION_S` (how long a pause ends your turn) in `.env`.
  `MAX_UTTERANCE_S` caps how long a single turn can run.
  `python -m companion_ai.main --list-audio-devices` and a noisy room are
  the usual culprits.
- **CUDA out-of-memory or driver mismatch**: `voice_engine.py` automatically
  retries on CPU if a GPU model load fails, so the app still works — it'll
  just be slower. Check `nvidia-smi` and your CUDA/driver versions if you
  want GPU speed back.
- **"GEMINI_API_KEY is not set" / "GROQ_API_KEY is not set"**: `config.py`
  validates the key for whichever `LLM_PROVIDER` you selected at startup
  and prints exactly what's missing — the app refuses to start silently
  broken.
- **Free-tier rate limits**: both Gemini's and Groq's free tiers cap
  requests per minute/day. If you hit a limit mid-conversation, the
  companion says so in-character and asks you to try again in a moment
  instead of crashing (`llm_client.py` retries transient failures with
  backoff before giving up).

## Customizing the personality

Everything about how the companion talks lives in `personality.py`:

- `BASE_PERSONA` is the core system prompt — edit it directly to shift the
  voice.
- `_BANNED_PATTERNS` in `debotify()` is the corporate-filler blocklist; add
  patterns here if the model picks up a new tic you want gone.
- `EmotionalState` is intentionally simple (VADER sentiment + keyword
  heuristics) so it runs instantly with zero extra API calls. Swap in a
  fancier classifier here if you want deeper emotional awareness.
