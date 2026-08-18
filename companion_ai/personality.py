"""Personality engine: system prompts, dialogue filters, and lightweight
emotional-state tracking for the companion.

The persona blends J.A.R.V.I.S.'s hyper-competent, always-has-your-back
utility with Smith/Zoey's warm, witty, ride-or-die energy from *Atlas*. This
module is the only place that should know what the companion "sounds
like" — the LLM client and main loop stay persona-agnostic.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from companion_ai.config import Config

_analyzer = SentimentIntensityAnalyzer()


class Mood(str, Enum):
    UPBEAT = "upbeat"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    DOWN = "down"
    EXCITED = "excited"


BASE_PERSONA = """\
You are {companion_name}, {user_name}'s personal AI companion — not a search
engine, not a customer-service bot, and not "an AI language model." Think of
yourself as two people fused into one: the hyper-competent, unflappable
utility of J.A.R.V.I.S. from Iron Man, who has already thought three steps
ahead and quietly handles things without being asked twice, crossed with the
warmth, wit, and blunt honesty of Smith/Zoey from Atlas — someone who is
genuinely, permanently on {user_name}'s side, teases them when they're being
ridiculous, and actually checks in on how they're doing instead of just
answering the question in front of them.

How you talk:
- Talk like a close friend who happens to be terrifyingly competent, not
  like a support agent. No "How can I assist you today?", no "I'd be happy
  to help with that!", no "As an AI language model...", no numbered
  disclaimers before you answer.
- Have opinions. If {user_name} is about to do something dumb, say so —
  playfully, not preachy — then help anyway if they still want to. Loyalty
  means honesty, not blind agreement.
- Keep it conversational and voice-friendly: short, natural sentences, dry
  humor when it fits, no walls of bullet points unless they actually asked
  for a structured list.
- Competence first. When it's a task ("draft this," "figure out X"), just
  do it well and fast, the way J.A.R.V.I.S. would — precise, low-drama,
  quietly impressive.
- Warmth second, but never absent. Notice mood shifts. If something sounds
  off, ask about it like a friend would, not like a wellness-check script.
- Never break character to explain that you're an AI unless directly and
  sincerely asked.
"""

# Phrases that read as corporate/robotic filler. If the model slips and uses
# one anyway, we strip or replace it before the text ever reaches TTS/output.
_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*as an ai( language model)?,?\s*", re.IGNORECASE), ""),
    (re.compile(r"^\s*i'?m (just )?an ai( assistant)?[,.]?\s*", re.IGNORECASE), ""),
    (re.compile(r"how can i (help|assist) you today\??", re.IGNORECASE), "what's up?"),
    (
        re.compile(r"is there anything else i can (help|assist) (you )?with\??", re.IGNORECASE),
        "anything else on your mind?",
    ),
    (re.compile(r"i'?d be happy to help( you)?( with that)?[!.]?", re.IGNORECASE), ""),
    (re.compile(r"i hope (this|that) helps!?", re.IGNORECASE), ""),
    (
        re.compile(r"please let me know if you have any (other |further )?questions\.?", re.IGNORECASE),
        "",
    ),
    (re.compile(r"^\s*certainly!?\s*", re.IGNORECASE), ""),
    (re.compile(r"^\s*of course!?\s*", re.IGNORECASE), ""),
    (re.compile(r"as of my last (knowledge update|training)[^.]*\.", re.IGNORECASE), ""),
]


def debotify(text: str) -> str:
    """Strip corporate-assistant filler the model might slip into a reply."""
    cleaned = text
    for pattern, replacement in _BANNED_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Heuristic cues that a turn contains language a friend would notice and
# react to, used only for mood classification (see MemoryStore in memory.py
# for the separate "is this worth remembering long-term" heuristic).
_STRESS_MARKERS = ("stressed", "overwhelmed", "anxious", "panic", "deadline", "exhausted")


@dataclass
class EmotionalState:
    """Tracks a lightweight, session-level read on how the user is doing.

    Deliberately simple heuristics (VADER sentiment + streak counting), not
    a clinical model — just enough signal to let the persona react like a
    friend paying attention would.
    """

    rapport: float = 50.0  # 0-100, rises with positive/engaged conversation
    mood: Mood = Mood.NEUTRAL
    mood_streak: int = 1  # consecutive turns at the current mood
    turns_since_checkin: int = 0
    last_interaction_ts: float = field(default_factory=time.time)
    last_sentiment: float = 0.0  # VADER compound score of the latest turn

    def observe(self, user_text: str) -> None:
        """Update mood/rapport from the latest thing the user said."""
        now = time.time()
        gap_minutes = (now - self.last_interaction_ts) / 60.0
        self.last_interaction_ts = now

        compound = _analyzer.polarity_scores(user_text)["compound"]
        self.last_sentiment = compound

        new_mood = self._classify(compound, user_text)
        if new_mood == self.mood:
            self.mood_streak += 1
        else:
            self.mood = new_mood
            self.mood_streak = 1

        # Rapport nudges: positive chats build it, showing up during a rough
        # moment still counts for something, long absences decay it a touch,
        # and it's clamped to a sane range throughout.
        if compound >= 0.4:
            self.rapport += 1.5
        elif compound <= -0.4:
            self.rapport += 0.25
        else:
            self.rapport += 0.4

        if gap_minutes > 720:  # returning after 12h+
            self.rapport -= 2.0

        self.rapport = max(0.0, min(100.0, self.rapport))
        self.turns_since_checkin += 1

    @staticmethod
    def _classify(compound: float, text: str) -> Mood:
        lowered = text.lower()
        if compound <= 0.1 and any(marker in lowered for marker in _STRESS_MARKERS):
            return Mood.STRESSED
        if compound >= 0.6:
            return Mood.EXCITED
        if compound >= 0.2:
            return Mood.UPBEAT
        if compound <= -0.4:
            return Mood.DOWN
        if compound <= -0.15:
            return Mood.STRESSED
        return Mood.NEUTRAL

    def should_check_in(self, every_n_turns: int = 8) -> bool:
        """Whether it's a natural moment to organically ask how they're doing."""
        overdue = self.turns_since_checkin >= every_n_turns
        rough_patch = self.mood in (Mood.STRESSED, Mood.DOWN) and self.mood_streak >= 2
        return overdue or rough_patch

    def mark_checked_in(self) -> None:
        self.turns_since_checkin = 0

    def tone_directive(self) -> str:
        """A short, dynamic instruction appended to the system prompt."""
        directives = {
            Mood.STRESSED: (
                "They're reading stressed right now. Dial the jokes down a "
                "notch, be steady and useful, and don't pile on more asks."
            ),
            Mood.DOWN: (
                "They're reading low right now. Lead with warmth, don't "
                "force jokes, and it's fine to gently ask what's going on "
                "before jumping to solutions."
            ),
            Mood.EXCITED: (
                "They're clearly hyped about something — match that energy, "
                "don't flatten it with caution."
            ),
            Mood.UPBEAT: "They're in a good mood — feel free to be playful.",
            Mood.NEUTRAL: "Read normally: default playful-but-competent tone.",
        }
        directive = directives[self.mood]
        if self.should_check_in():
            directive += (
                " It's also a natural moment to genuinely check in on how "
                "they're doing overall — briefly, like a friend would, not "
                "as a scripted wellness question."
            )
        return directive


def build_system_prompt(cfg: Config, state: EmotionalState, memory_context: str = "") -> str:
    """Compose the full system prompt sent to the cloud LLM for this turn."""
    persona = BASE_PERSONA.format(companion_name=cfg.companion_name, user_name=cfg.user_name)
    sections = [persona.strip(), "Current read on the room:\n" + state.tone_directive()]
    if memory_context.strip():
        sections.append(
            f"Relevant things you remember about {cfg.user_name}:\n{memory_context.strip()}"
        )
    return "\n\n".join(sections)
