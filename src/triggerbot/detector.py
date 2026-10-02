"""Turning clip scores into detections on continuous audio.

A 1.5 s window slides over the stream every 0.25 s. The word fires when a window's score
crosses the threshold, then a refractory period stops the same utterance, which several
overlapping windows all contain, from firing again.

The same `Trigger` logic is used by the offline evaluation (`scan`) and by the live bot, so
the numbers measured offline describe what the bot actually does.
"""

from collections.abc import Callable

import numpy as np

from .audio import SAMPLE_RATE, WINDOW_SAMPLES

HOP_SECONDS = 0.25
REFRACTORY_SECONDS = 2.0


class Trigger:
    def __init__(self, threshold: float, refractory: float = REFRACTORY_SECONDS):
        self.threshold, self.refractory = threshold, refractory
        self.last = -np.inf

    def update(self, t: float, score: float) -> bool:
        if score >= self.threshold and t - self.last >= self.refractory:
            self.last = t
            return True
        return False


def scan(audio: np.ndarray, predict_proba: Callable[[np.ndarray], np.ndarray], threshold: float,
         hop: float = HOP_SECONDS, refractory: float = REFRACTORY_SECONDS) -> list[float]:
    """Run the detector over a whole recording. Returns detection times (window centres, s)."""
    step = int(hop * SAMPLE_RATE)
    starts = np.arange(0, max(len(audio) - WINDOW_SAMPLES, 0) + 1, step)
    if len(starts) == 0:
        return []
    windows = np.stack([audio[s:s + WINDOW_SAMPLES] for s in starts])
    scores = np.concatenate([predict_proba(windows[i:i + 256])
                             for i in range(0, len(windows), 256)])
    trigger = Trigger(threshold, refractory)
    centres = (starts + WINDOW_SAMPLES / 2) / SAMPLE_RATE
    return [float(t) for t, s in zip(centres, scores, strict=True) if trigger.update(t, s)]


class StreamingDetector:
    """Incremental version for live audio: feed chunks of any size, get detection times back."""

    def __init__(self, predict_proba, threshold: float, hop: float = HOP_SECONDS,
                 refractory: float = REFRACTORY_SECONDS):
        self.predict_proba = predict_proba
        self.step = int(hop * SAMPLE_RATE)
        self.trigger = Trigger(threshold, refractory)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.consumed = 0  # samples dropped from the front of the buffer so far

    def feed(self, samples: np.ndarray) -> list[float]:
        self.buffer = np.concatenate([self.buffer, samples.astype(np.float32)])
        hits = []
        while len(self.buffer) >= WINDOW_SAMPLES:
            score = float(self.predict_proba(self.buffer[None, :WINDOW_SAMPLES])[0])
            t = (self.consumed + WINDOW_SAMPLES / 2) / SAMPLE_RATE
            if self.trigger.update(t, score):
                hits.append(t)
            self.buffer = self.buffer[self.step:]
            self.consumed += self.step
        return hits


def match_events(detections: list[float], events: list[tuple[float, float]],
                 tolerance: float = 0.5) -> dict:
    """A detection is correct if it falls inside a real utterance (+/- tolerance) that hasn't
    been matched yet; every other detection is a false trigger."""
    matched = [False] * len(events)
    false_triggers = 0
    for t in detections:
        for i, (a, b) in enumerate(events):
            if not matched[i] and a - tolerance <= t <= b + tolerance:
                matched[i] = True
                break
        else:
            false_triggers += 1
    return {"events": len(events), "detected": int(sum(matched)),
            "false_triggers": false_triggers}
