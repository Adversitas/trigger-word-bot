"""Audio I/O and the log-mel front end, in NumPy only.

Discord delivers 48 kHz, 16-bit stereo PCM. Speech models work at 16 kHz mono, so audio is
downmixed and decimated by 3 (with a low-pass filter first, so frequencies above 8 kHz don't
fold back into the speech band as aliasing).
"""

import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

N_FFT = 400  # 25 ms
HOP = 160  # 10 ms
N_MELS = 40


def pcm16_to_float(raw: bytes, channels: int) -> np.ndarray:
    """Interleaved 16-bit PCM bytes -> mono float32 in [-1, 1]."""
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x[: len(x) - len(x) % channels].reshape(-1, channels).mean(axis=1)
    return x


@lru_cache(maxsize=4)
def _lowpass(factor: int, taps: int = 63) -> np.ndarray:
    """Windowed-sinc low-pass filter with cutoff at the new Nyquist frequency."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(n / factor) * np.hamming(taps)
    return (h / h.sum()).astype(np.float32)


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    if sr % target == 0:  # integer decimation, e.g. 48 kHz -> 16 kHz
        factor = sr // target
        return np.convolve(x, _lowpass(factor), mode="same")[::factor].astype(np.float32)
    # General case (rare here): linear interpolation
    t_new = np.arange(int(len(x) * target / sr)) * sr / target
    return np.interp(t_new, np.arange(len(x)), x).astype(np.float32)


def read_wav(path: Path) -> np.ndarray:
    """Any 16-bit PCM WAV -> 16 kHz mono float32."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        x = pcm16_to_float(w.readframes(w.getnframes()), w.getnchannels())
        return resample(x, w.getframerate())


def fix_length(x: np.ndarray, n: int = WINDOW_SAMPLES) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


@lru_cache(maxsize=1)
def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale: fine resolution at low
    frequencies, where most speech information is, like the human ear."""

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def log_mel(x: np.ndarray) -> np.ndarray:
    """(samples,) -> (frames, N_MELS) log-mel spectrogram."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    return np.log(power @ mel_filterbank().T + 1e-6).astype(np.float32)


def log_mel_batch(xs: np.ndarray) -> np.ndarray:
    return np.stack([log_mel(x) for x in xs])
