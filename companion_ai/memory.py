"""Short- and long-term memory for the companion.

Short-term memory is a rolling window of the current conversation. Long-term
memory is a small local embedding store (sentence-transformers + cosine
similarity over numpy arrays) that persists facts about the user across
sessions, so the companion actually remembers you tomorrow without needing
any paid vector DB.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from companion_ai.config import Config


@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list[float]
    created_at: float
    importance: float = 1.0


class ConversationHistory:
    """Rolling short-term memory: the last N turns, ready for the LLM."""

    def __init__(self, max_turns: int) -> None:
        self._messages: deque = deque(maxlen=max(1, max_turns) * 2)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def as_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


# Heuristic cues that a user utterance likely contains a durable, worth-
# remembering fact rather than small talk. Keeps long-term memory useful
# without needing an extra LLM call just to decide what to store.
_FACT_CUES = re.compile(
    r"\b("
    r"my name is|i'?m called|i live in|i work (at|as)|i'?m (a|an)\s\w+|"
    r"i like|i love|i hate|i can'?t stand|my favorite|i'?m allergic to|"
    r"remember that|remember this|don'?t forget|my birthday|i'?m married|"
    r"my (wife|husband|partner|kids?|son|daughter|dog|cat)|"
    r"i'?m (studying|majoring in)|my goal is|i'?m trying to"
    r")\b",
    re.IGNORECASE,
)


def looks_memorable(text: str) -> bool:
    """Cheap heuristic: does this sound like a durable fact worth keeping?"""
    if len(text.strip()) < 6:
        return False
    return bool(_FACT_CUES.search(text))


class MemoryStore:
    """Local, persistent long-term memory backed by sentence embeddings."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._path = cfg.memory_dir / "long_term_memory.jsonl"
        self._items: list[MemoryItem] = []
        self._model = None  # lazy-loaded; embedding models aren't free to import
        self._load()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._cfg.embedding_model, device=self._cfg.device)
        return self._model

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._items.append(MemoryItem(**data))

    def _append_to_disk(self, item: MemoryItem) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.__dict__) + "\n")

    def add(self, text: str, importance: float = 1.0) -> MemoryItem:
        embedding = self.model.encode(text, normalize_embeddings=True).tolist()
        item = MemoryItem(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            created_at=time.time(),
            importance=importance,
        )
        self._items.append(item)
        self._append_to_disk(item)
        return item

    def remember_if_notable(self, user_text: str) -> MemoryItem | None:
        """Store the utterance if it heuristically looks like a durable fact."""
        if looks_memorable(user_text):
            return self.add(user_text)
        return None

    def retrieve_relevant(
        self, query: str, top_k: int | None = None, threshold: float | None = None
    ) -> list[str]:
        if not self._items:
            return []
        top_k = top_k or self._cfg.memory_top_k
        threshold = threshold if threshold is not None else self._cfg.memory_relevance_threshold

        query_vec = np.array(self.model.encode(query, normalize_embeddings=True))
        matrix = np.array([item.embedding for item in self._items])
        similarities = matrix @ query_vec  # both sides unit-normalized -> cosine sim

        ranked = sorted(
            zip(similarities.tolist(), self._items), key=lambda pair: pair[0], reverse=True
        )
        return [item.text for score, item in ranked[:top_k] if score >= threshold]

    def __len__(self) -> int:
        return len(self._items)
