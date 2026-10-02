import json
from pathlib import Path

import typer

app = typer.Typer(help="Train, evaluate and run the trigger-word detector.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Trigger-word spotting for Discord voice chat."""


def _sessions(data: Path):
    from .dataset import load_session

    folders = sorted(p for p in data.iterdir() if (p / "manifest.csv").exists())
    if not folders:
        raise typer.BadParameter(f"No session folders with a manifest.csv in {data}")
    return [load_session(f) for f in folders]


@app.command()
def evaluate(data: Path = Path("data"), out: Path = Path("reports")):
    """Train on each session in turn and test on the others; write reports/RESULTS.md."""
    from .evaluate import cross_session, markdown
    from .models import AlwaysNo, MelStatsLogistic, SmallCNN

    sessions = _sessions(data)
    for s in sessions:
        typer.echo(f"{s.name}: {len(s.y)} clips, {s.y.sum()} positive, {len(s.events)} utterances")
    results = cross_session(sessions, [AlwaysNo, MelStatsLogistic, SmallCNN])
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    table = markdown(results)
    (out / "RESULTS.md").write_text("# Results\n\n" + table + "\n", encoding="utf-8")
    typer.echo("\n" + table)


@app.command()
def train(data: Path = Path("data"), out: Path = Path("models/detector.pt")):
    """Train the CNN on every session and save it, with its threshold, for the bot."""
    import numpy as np

    from .evaluate import choose_threshold
    from .models import SmallCNN

    sessions = _sessions(data)
    audio = np.concatenate([s.audio for s in sessions])
    y = np.concatenate([s.y for s in sessions])
    threshold = choose_threshold(audio, y, SmallCNN)
    model = SmallCNN()
    model.fit(audio, y)
    model.threshold = threshold
    model.save(out)
    typer.echo(f"Trained on {len(y)} clips ({y.sum()} positive); threshold {threshold:.2f}; "
               f"saved {out}")


@app.command()
def scan(recording: Path = typer.Argument(..., exists=True),
         model_path: Path = typer.Option(Path("models/detector.pt"), "--model")):
    """Print the times at which the trigger word is detected in a WAV file."""
    from .audio import read_wav
    from .detector import scan as run_scan
    from .models import SmallCNN

    model = SmallCNN.load(model_path)
    for t in run_scan(read_wav(recording), model.predict_proba, model.threshold):
        typer.echo(f"{int(t // 60):02d}:{t % 60:05.2f}")


if __name__ == "__main__":
    app()
