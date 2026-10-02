# Trigger Word Bot (proof of concept)

A Discord bot that listens in a voice channel and posts a message whenever someone says a chosen
**trigger word**:

> 🔔 **Hamza** said the word (3 today)

This repository is a **proof of concept**. The complete pipeline is built and tested: Discord
audio → preprocessing → detector → message. The trigger-word model itself is still
experimental, because it was trained on only a handful of examples (see
[Status](#status-what-works-and-what-doesnt)). Until a good model exists, the bot runs with a
demo detector that fires on loud speech, so the whole flow can be tried today.

---

## How it works

```
 Discord voice channel
   │  48 kHz stereo PCM per speaker (discord-ext-voice-recv)
   ▼
 TriggerSink ── keeps audio only for users who typed !optin; nothing written to disk
   │  every 250 ms
   ▼
 audio.py ───── downmix to mono, low-pass + resample to 16 kHz, log-mel spectrogram
   │
   ▼
 StreamingDetector ── 1.5 s window sliding every 0.25 s → detector score
   │                   fires above the threshold, then waits 2 s (refractory period)
   ▼
 "🔔 <name> said the word (n today)" in the text channel
```

**Privacy by design.** The bot only analyses users who explicitly opt in with `!optin`, and
stops with `!optout`. Audio lives in memory for about 1.5 seconds and is never saved. The bot
token is read from the environment, never from the code.

**Non-blocking.** Discord delivers audio on a separate thread. That thread only appends bytes
under a lock; an asyncio task collects them every 250 ms and runs the detector in a worker
thread, so the bot stays responsive.

**One detection logic everywhere.** The bot and the offline evaluation use the same
`StreamingDetector` and `Trigger` code, so the numbers measured offline describe what the bot
actually does.

### Detectors

| Detector | What it is | Status |
|---|---|---|
| `LoudnessDemo` | Fires when speech in the window is loud | Default: demonstrates the pipeline, doesn't recognise words |
| `SmallCNN` | 3 conv blocks on log-mel spectrograms, trained with augmentation (time shift, volume, noise, SpecAugment) | Trained with `triggerbot train`; used automatically once `models/detector.pt` exists |
| `MelStatsLogistic` | Logistic regression on per-band mel statistics | Baseline for the evaluation |

Any object with a `predict_proba(audio)` method and a `threshold` attribute can be plugged in.
A Whisper transcription-based detector would be the natural next one (see
[Next steps](#next-steps)).

---

## Status: what works and what doesn't

The training data came from two recordings of one speaker, cut into 1.5-second clips and
labelled by hand with an interactive tool: **340 clips, but only 9 separate times the word was
actually said**.

The evaluation trains on one recording, tests on the other, and also slides the detector over
the whole held-out recording, exactly as the bot would. The results
([reports/RESULTS.md](reports/RESULTS.md)):

| Train → test | Model | Recall (clips) | Avg. precision | Prevalence | Words caught | False triggers / min |
|---|---|---|---|---|---|---|
| rec. 2 → rec. 1 | logistic regression | 0.56 | 0.09 | 0.06 | 6/6 | 16.0 |
| rec. 2 → rec. 1 | CNN | 0.62 | 0.06 | 0.06 | 6/6 | 17.3 |
| rec. 1 → rec. 2 | logistic regression | 0.65 | 0.32 | 0.27 | 3/3 | 19.5 |
| rec. 1 → rec. 2 | CNN | 0.70 | 0.19 | 0.27 | 3/3 | 25.1 |

**Reading:** the models' average precision is about equal to the share of positive clips
(the prevalence), which is what a model with no skill scores. They catch every utterance only
by firing **16–25 false alarms per minute**. With 9 examples, the models can't learn what the
word sounds like. The evaluation catches this, which is its job.

---

## Next steps

1. **Zero-shot detection with Whisper:** transcribe each window with a small Whisper model and
   look for the word. It needs no training data, if the word is one Whisper can spell.
2. **More data for the CNN:** 50–100 utterances across several sessions and voices, plus plenty
   of ordinary speech as negatives. Record them with an opt-in collection command and label
   them with the existing clip tool.
3. **Pretrained audio embeddings** (e.g. wav2vec 2.0) with a small classifier on top: a
   middle ground that needs far fewer examples than training from scratch.

Each option would be measured with the same cross-session evaluation and its
false-triggers-per-minute number.

---

## Running it

### 1. Create the Discord bot

1. In the [Discord Developer Portal](https://discord.com/developers/applications), create an
   application, then a bot. Under **Bot**, enable **Message Content Intent**.
2. Invite it to your server (OAuth2 → URL Generator → scope `bot`) with the **Connect**,
   **Speak**, **Read Messages** and **Send Messages** permissions.
3. Copy the token into a `.env` file (see `.env.example`) or set `DISCORD_TOKEN`. Never put the
   token in the code.

### 2. Start the bot

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install -e ".[bot]"
set DISCORD_TOKEN=your-token    # Linux / macOS: export DISCORD_TOKEN=your-token
python -m triggerbot.bot
```

Or with Docker: `docker build -t triggerbot .` then
`docker run --rm -e DISCORD_TOKEN=your-token -v .:/app triggerbot python -m triggerbot.bot`.

### 3. Use it in Discord

| Command | Effect |
|---|---|
| `!join` | Bot joins your voice channel and announces in the current text channel |
| `!optin` / `!optout` | Allow / stop the bot analysing *your* voice |
| `!count` | How many times each user said the word today |
| `!leave` | Bot leaves the voice channel |

### 4. Train and evaluate a model (optional)

Put labelled sessions in `data/<session>/`: `recording.wav`, `clips/*.wav`, and a
`manifest.csv` with columns `filename,start_sec,end_sec,label`, where `label` is
`said_trigger_word` or `did_not_say_trigger_word`. Then:

```bash
pip install -e ".[dev]"
triggerbot evaluate               # cross-session results -> reports/RESULTS.md
triggerbot train                  # CNN on all sessions -> models/detector.pt (used by the bot)
triggerbot scan some_audio.wav    # times at which the word is detected
pytest                            # offline tests, no Discord or recordings needed
```

Recordings are personal data: `data/` and all `.wav` files are git-ignored.

---

## Project structure

```
src/triggerbot/
├── audio.py      PCM conversion, anti-aliased resampling, log-mel spectrogram (NumPy)
├── dataset.py    labelled sessions; merges overlapping positive clips into utterances
├── models.py     demo detector, baselines, CNN with augmentation, save/load
├── detector.py   sliding-window streaming detector, refractory trigger, event matching
├── evaluate.py   cross-session evaluation, out-of-fold threshold choice, stream metrics
├── bot.py        Discord bot: opt-in audio sink, background detection loop, commands
└── cli.py        triggerbot evaluate / train / scan
tests/            audio, detector, evaluation and bot-fallback tests
```
