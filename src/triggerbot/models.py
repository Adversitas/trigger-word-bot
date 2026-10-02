"""Clip classifiers: P(trigger word in this 1.5 s window).

    fit(audio, y)               audio: (n, WINDOW_SAMPLES) float32 at 16 kHz
    predict_proba(audio) -> (n,)

With only a few dozen positive examples, augmentation does most of the work for the CNN:
every epoch sees each clip shifted in time, at a different volume, with background noise and
with random bands of the spectrogram hidden (SpecAugment).
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audio import N_MELS, SAMPLE_RATE, log_mel_batch


class AlwaysNo:
    """Never fires. Its accuracy is high (most clips are negative); its recall is zero."""

    name = "always 'no'"
    fixed_threshold = 1.01  # never fires, whatever the training data

    def fit(self, audio, y):
        self.p = float(y.mean())

    def predict_proba(self, audio):
        return np.full(len(audio), self.p)


class MelStatsLogistic:
    """Mean, standard deviation and maximum of each mel band over the clip -> logistic
    regression. Ignores word timing entirely, which makes it a useful baseline for the CNN."""

    name = "logistic regression (mel statistics)"

    def _features(self, audio):
        m = log_mel_batch(audio)
        return np.concatenate([m.mean(axis=1), m.std(axis=1), m.max(axis=1)], axis=1)

    def fit(self, audio, y):
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=0.1, class_weight="balanced",
                                                      max_iter=2000))
        self.model.fit(self._features(audio), y)

    def predict_proba(self, audio):
        return self.model.predict_proba(self._features(audio))[:, 1]


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(audio)
    max_shift = int(0.3 * SAMPLE_RATE)
    for i, x in enumerate(audio):
        shift = rng.integers(-max_shift, max_shift + 1)
        x = np.roll(x, shift)
        if shift > 0:
            x[:shift] = 0
        elif shift < 0:
            x[shift:] = 0
        x = x * 10 ** (rng.uniform(-6, 6) / 20)  # +/- 6 dB
        rms = np.sqrt(np.mean(x**2)) + 1e-8
        snr_db = rng.uniform(15, 40)
        out[i] = x + rng.normal(0, rms / 10 ** (snr_db / 20), len(x))
    return out


def spec_augment(m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = m.copy()
    for spec in m:
        f = rng.integers(0, 7)
        f0 = rng.integers(0, N_MELS - f + 1)
        spec[:, f0:f0 + f] = spec.mean()
        t = rng.integers(0, 21)
        t0 = rng.integers(0, spec.shape[0] - t + 1)
        spec[t0:t0 + t, :] = spec.mean()
    return m


class SmallCNN:
    """Three conv blocks on the log-mel spectrogram, then global pooling and one output."""

    name = "CNN (log-mel + augmentation)"

    def __init__(self, epochs: int = 40, batch_size: int = 32, lr: float = 1e-3, seed: int = 0):
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.threshold = 0.5

    @staticmethod
    def _build():
        from torch import nn

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                                 nn.ReLU(), nn.MaxPool2d(2))

        return nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                             nn.Linear(64, 1))

    def _to_tensor(self, mels):
        import torch

        x = (mels - self.mean) / self.std
        return torch.from_numpy(x[:, None].astype(np.float32)).to(self.device)

    def fit(self, audio, y):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clean = log_mel_batch(audio)
        self.mean, self.std = clean.mean(axis=(0, 1)), clean.std(axis=(0, 1)) + 1e-6
        self.net = self._build().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        targets = torch.from_numpy(y.astype(np.float32)).to(self.device)

        for _ in range(self.epochs):
            self.net.train()
            mels = spec_augment(log_mel_batch(augment(audio, rng)), rng)
            order = rng.permutation(len(audio))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs at least 2 samples
                    continue
                opt.zero_grad()
                loss_fn(self.net(self._to_tensor(mels[idx])).squeeze(1), targets[idx]).backward()
                opt.step()

    def predict_proba(self, audio):
        import torch

        self.net.eval()
        with torch.no_grad():
            logits = self.net(self._to_tensor(log_mel_batch(audio))).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "mean": self.mean, "std": self.std,
                    "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SmallCNN":
        import torch

        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls()
        model.device = device
        model.net = cls._build().to(device)
        model.net.load_state_dict(ckpt["state"])
        model.net.eval()
        model.mean, model.std, model.threshold = ckpt["mean"], ckpt["std"], ckpt["threshold"]
        return model


class LoudnessDemo:
    """Placeholder detector for the proof of concept: fires when speech in the window is loud
    (e.g. a shout). It does not recognise any word; it only shows the full pipeline working
    end to end before a real model exists. Any object with `predict_proba(audio)` and a
    `threshold` can replace it."""

    name = "demo: loudness"
    threshold = 0.5

    def __init__(self, loud_rms: float = 0.15):
        self.loud_rms = loud_rms

    def predict_proba(self, audio):
        rms = np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2, axis=1))
        return np.clip(rms / (2 * self.loud_rms), 0, 1)


def load_detector(model_path: Path):
    """The trained CNN if one exists, otherwise the loudness demo."""
    if model_path.exists():
        return SmallCNN.load(model_path)
    return LoudnessDemo()


def all_models():
    return [AlwaysNo(), MelStatsLogistic(), SmallCNN()]
