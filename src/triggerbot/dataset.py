"""Labelled recording sessions.

A session folder holds one long recording, the 1.5 s clips cut from it, and a manifest:

    data/<session>/
        recording.wav
        clips/chunk_0000.wav ...
        manifest.csv          filename,start_sec,end_sec,label

Labels are `said_trigger_word` / `did_not_say_trigger_word`. Clips overlap (hop 0.75 s), so
the same utterance usually appears in two neighbouring clips. That's why evaluation keeps
whole sessions apart instead of splitting clips at random.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import fix_length, read_wav

POSITIVE = "said_trigger_word"
NEGATIVE = "did_not_say_trigger_word"


@dataclass
class Session:
    name: str
    audio: np.ndarray  # (clips, WINDOW_SAMPLES) at 16 kHz
    y: np.ndarray  # (clips,) 1 = trigger word
    starts: np.ndarray  # clip start times in the long recording (s)
    ends: np.ndarray
    recording: Path

    @property
    def events(self) -> list[tuple[float, float]]:
        """Merge overlapping positive clips into (start, end) spans: one span per time the
        word was actually said."""
        spans: list[list[float]] = []
        for s, e, y in sorted(zip(self.starts, self.ends, self.y, strict=True)):
            if not y:
                continue
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        return [(a, b) for a, b in spans]


def load_session(folder: Path) -> Session:
    with open(folder / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unknown = {r["label"] for r in rows} - {POSITIVE, NEGATIVE}
    if unknown:
        raise ValueError(f"{folder}: unlabelled or unknown labels {unknown}")
    rows.sort(key=lambda r: float(r["start_sec"]))
    return Session(
        name=folder.name,
        audio=np.stack([fix_length(read_wav(folder / "clips" / r["filename"])) for r in rows]),
        y=np.array([r["label"] == POSITIVE for r in rows], dtype=int),
        starts=np.array([float(r["start_sec"]) for r in rows]),
        ends=np.array([float(r["end_sec"]) for r in rows]),
        recording=folder / "recording.wav",
    )
