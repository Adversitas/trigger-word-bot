"""Offline tests: audio front end, detector logic and evaluation helpers.
No Discord connection and no recordings needed."""
import numpy as np
import pytest

from triggerbot.audio import (
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    fix_length,
    log_mel,
    pcm16_to_float,
    resample,
)
from triggerbot.dataset import Session
from triggerbot.detector import StreamingDetector, Trigger, match_events, scan
from triggerbot.evaluate import choose_threshold, f1_at
from triggerbot.models import AlwaysNo, LoudnessDemo, MelStatsLogistic, load_detector

# --------------------------------------------------------------------------- audio


def test_stereo_pcm_is_downmixed():
    stereo = np.array([1000, 3000, -2000, 0], dtype="<i2").tobytes()
    np.testing.assert_allclose(pcm16_to_float(stereo, 2), [2000 / 32768, -1000 / 32768])


def test_resampling_keeps_speech_frequencies_and_removes_aliasing():
    t = np.arange(48_000) / 48_000
    tone = np.sin(2 * np.pi * 440 * t)  # in the speech band: kept
    out = resample(tone, 48_000)
    assert len(out) == 16_000
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) - 440) <= 1
    high = resample(np.sin(2 * np.pi * 15_000 * t), 48_000)  # above 8 kHz: filtered out
    assert np.abs(high[100:-100]).max() < 0.1


def test_log_mel_shape():
    assert log_mel(np.zeros(WINDOW_SAMPLES, dtype=np.float32)).shape == (148, N_MELS)
    assert len(fix_length(np.ones(10))) == WINDOW_SAMPLES


# --------------------------------------------------------------------------- detector


def test_trigger_has_refractory_period():
    trig = Trigger(threshold=0.5, refractory=2.0)
    fired = [trig.update(t, 0.9) for t in (0.0, 0.25, 1.9, 2.0, 2.5)]
    assert fired == [True, False, False, True, False]


def loud_bursts(times, total_seconds=20):
    """Silence with 0.4 s noise bursts at the given times; a fake 'model' fires on loud windows."""
    rng = np.random.default_rng(0)
    x = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)
    for t in times:
        s = int(t * SAMPLE_RATE)
        x[s:s + int(0.4 * SAMPLE_RATE)] = rng.normal(0, 0.5, int(0.4 * SAMPLE_RATE))
    return x


def loudness_model(windows):
    return np.sqrt((windows**2).mean(axis=1)) * 5


def test_scan_fires_once_per_burst():
    detections = scan(loud_bursts([3, 10, 15]), loudness_model, threshold=0.5)
    assert len(detections) == 3
    for t, true in zip(detections, [3, 10, 15], strict=True):
        assert abs(t - true) < 1.5


def test_streaming_detector_matches_offline_scan():
    audio = loud_bursts([4, 12])
    offline = scan(audio, loudness_model, 0.5)
    det = StreamingDetector(loudness_model, 0.5)
    live = []
    for start in range(0, len(audio), 1234):  # arbitrary chunk sizes, like network packets
        live += det.feed(audio[start:start + 1234])
    np.testing.assert_allclose(live, offline)


def test_match_events():
    r = match_events([1.0, 1.2, 5.0, 9.0], [(0.5, 2.0), (8.5, 9.5)])
    assert r == {"events": 2, "detected": 2, "false_triggers": 2}


# --------------------------------------------------------------------------- evaluation


def test_events_merge_overlapping_positive_clips():
    s = Session("s", np.zeros((4, 1)), np.array([1, 1, 0, 1]), np.array([0, 0.75, 1.5, 3.0]),
                np.array([1.5, 2.25, 3.0, 4.5]), recording=None)
    assert s.events == [(0, 2.25), (3.0, 4.5)]


def test_f1_and_always_no_never_fires():
    y = np.array([1, 0, 1, 0])
    assert f1_at(y, np.array([0.9, 0.1, 0.8, 0.7]), 0.5) == pytest.approx((2 / 3, 1.0, 0.8))
    assert choose_threshold(np.zeros((4, 10)), y, AlwaysNo) > 1


def test_logistic_learns_tone_vs_noise():
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    tone = [0.3 * np.sin(2 * np.pi * 1000 * t) + rng.normal(0, 0.01, len(t)) for _ in range(20)]
    noise = [rng.normal(0, 0.1, len(t)) for _ in range(20)]
    X, y = np.array(tone + noise, dtype=np.float32), np.array([1] * 20 + [0] * 20)
    m = MelStatsLogistic()
    m.fit(X, y)
    assert (m.predict_proba(X).round() == y).mean() == 1.0


def test_bot_falls_back_to_demo_detector(tmp_path):
    det = load_detector(tmp_path / "missing.pt")
    assert isinstance(det, LoudnessDemo)
    detections = scan(loud_bursts([5]), det.predict_proba, det.threshold)
    assert len(detections) == 1
