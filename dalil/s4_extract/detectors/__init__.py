"""Structural detectors that find boundaries inside a text buffer.

Each ``detect`` returns ``Span`` objects; turning spans into parents is the
caller's policy, never the detector's."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A detected structural signal over a text buffer.

    start/end are character offsets into the text the detector was run on.
    label is the detector-specific type (e.g. 'heading', 'quran_ref').
    meta carries detector-specific extras (heading text, surah/ayah nums, etc.).
    """
    start: int
    end: int
    label: str
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label,
                "text": self.text, "meta": self.meta}
