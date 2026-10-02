"""Cross-session evaluation.

Train on one recording session, test on another, for every ordering. Two things are measured
on the held-out session:

1. Clip level: precision, recall and F1 at the chosen threshold, plus average precision
   (area under the precision-recall curve, threshold-free).
2. Stream level, which is what the bot experiences: the detector slides over the *whole*
   held-out recording. How many times the word was really said were caught, and how many
   false triggers per minute were posted?

The threshold is chosen on the training session only, from out-of-fold predictions on
contiguous blocks, so it is never tuned on the test recording.
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score

from .audio import SAMPLE_RATE, read_wav
from .dataset import Session
from .detector import match_events, scan


def f1_at(y, p, thr) -> tuple[float, float, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / (y == 1).sum() if (y == 1).sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def choose_threshold(audio, y, factory: Callable, n_blocks: int = 3, seed: int = 0) -> float:
    """Out-of-fold scores on contiguous blocks of the training session, then the threshold
    with the best F1 on those scores."""
    model = factory()
    if hasattr(model, "fixed_threshold"):
        return model.fixed_threshold
    oof = np.zeros(len(y))
    for block in np.array_split(np.arange(len(y)), n_blocks):
        rest = np.setdiff1d(np.arange(len(y)), block)
        m = factory()
        m.fit(audio[rest], y[rest])
        oof[block] = m.predict_proba(audio[block])
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.unique(np.round(oof, 3)):
        f1 = f1_at(y, oof, thr)[2]
        if f1 > best_f1 or (f1 == best_f1 and thr > best_thr):  # ties: prefer fewer triggers
            best_thr, best_f1 = float(thr), f1
    return best_thr


def evaluate_pair(train: Session, test: Session, factory: Callable) -> dict:
    thr = choose_threshold(train.audio, train.y, factory)
    model = factory()
    model.fit(train.audio, train.y)
    p = model.predict_proba(test.audio)
    precision, recall, f1 = f1_at(test.y, p, thr)

    recording = read_wav(test.recording)
    detections = scan(recording, model.predict_proba, thr)
    stream = match_events(detections, test.events)
    minutes = len(recording) / SAMPLE_RATE / 60
    return {
        "model": model.name,
        "train": train.name,
        "test": test.name,
        "threshold": thr,
        "clip": {"precision": precision, "recall": recall, "f1": f1,
                 "average_precision": float(average_precision_score(test.y, p))
                 if np.ptp(p) > 0 else float(test.y.mean()),
                 "positives": int(test.y.sum()), "clips": len(test.y)},
        "stream": {**stream, "minutes": minutes,
                   "false_triggers_per_min": stream["false_triggers"] / minutes},
    }


def cross_session(sessions: list[Session], factories: list[Callable],
                  log: Callable[[str], None] = print) -> list[dict]:
    if len(sessions) < 2:
        raise ValueError("Cross-session evaluation needs at least 2 sessions")
    results = []
    for i, test in enumerate(sessions):
        others = [s for j, s in enumerate(sessions) if j != i]
        train = others[0] if len(others) == 1 else _concat(others)
        for factory in factories:
            log(f"  train {train.name} -> test {test.name}: {factory().name}")
            results.append(evaluate_pair(train, test, factory))
    return results


def _concat(sessions: list[Session]) -> Session:
    return Session(
        name="+".join(s.name for s in sessions),
        audio=np.concatenate([s.audio for s in sessions]),
        y=np.concatenate([s.y for s in sessions]),
        starts=np.concatenate([s.starts for s in sessions]),
        ends=np.concatenate([s.ends for s in sessions]),
        recording=sessions[0].recording,
    )


def markdown(results: list[dict]) -> str:
    lines = [
        ("| Train → test | Model | Threshold | Precision | Recall | F1 | Avg. precision "
         "| Words caught | False triggers / min |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c, s = r["clip"], r["stream"]
        lines.append(
            f"| {r['train']} → {r['test']} | {r['model']} | {r['threshold']:.2f} | "
            f"{c['precision']:.2f} | {c['recall']:.2f} | {c['f1']:.2f} | "
            f"{c['average_precision']:.2f} | {s['detected']}/{s['events']} | "
            f"{s['false_triggers_per_min']:.2f} |"
        )
    return "\n".join(lines)
