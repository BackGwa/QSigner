from collections import defaultdict
from pathlib import Path

import click
from tqdm import tqdm

from qsinger.analyzer import Analyzer
from qsinger.data.audio import SAMPLE_RATE, load_audio, split_at_silences
from qsinger.data.dataset import Item, RawDataset, item_name, read_dataset_info
from qsinger.paths import BASE_DIR
from qsinger.training import diffsinger, vocoder

EXPERIMENTS = {"acoustic": "voice-acoustic", "variance": "voice-variance"}


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of WAV/FLAC recordings (subfolders included).",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output folder. A rerun reuses its analysis.",
)
@click.option(
    "--min-confidence",
    default=0.5,
    show_default=True,
    help="Segments the analyzer is less sure about are left out.",
)
@click.option("--acoustic-steps", default=30000, show_default=True)
@click.option("--variance-steps", default=20000, show_default=True)
@click.option("--learning-rate", default=0.0002, show_default=True)
@click.option(
    "--batch-frames",
    default=30000,
    show_default=True,
    help="Acoustic batch size in frames.",
)
@click.option(
    "--finetune-vocoder",
    is_flag=True,
    help="Also fine-tune the vocoder on these recordings.",
)
@click.option("--vocoder-steps", default=2000, show_default=True)
def train(
    data,
    out,
    min_confidence,
    acoustic_steps,
    variance_steps,
    learning_rate,
    batch_frames,
    finetune_vocoder,
    vocoder_steps,
):
    raw_dir = out / "raw"
    if not raw_dir.exists():
        _analyze(data, raw_dir, min_confidence)
    datasets = [read_dataset_info(d) for d in sorted(raw_dir.iterdir())]

    for model, steps in (
        ("acoustic", acoustic_steps),
        ("variance", variance_steps),
    ):
        config = diffsinger.finetuned(
            diffsinger.CONFIGS[model](datasets, out / "binary" / model),
            BASE_DIR / model / "model.ckpt",
            steps,
            learning_rate,
        )
        if model == "acoustic":
            config["max_batch_frames"] = batch_frames
        diffsinger.fit(config, out, EXPERIMENTS[model])
        click.echo(f"{model}: {out / 'checkpoints' / EXPERIMENTS[model]}")
    if finetune_vocoder:
        wavs = [d.root / "wavs" for d in datasets]
        checkpoint = vocoder.finetune(wavs, out / "vocoder", vocoder_steps)
        click.echo(f"vocoder: {checkpoint}")


def _analyze(data: Path, raw_dir: Path, min_confidence: float):
    analyzer = Analyzer()
    recordings = {
        p: load_audio(p)
        for p in sorted(data.rglob("*"))
        if p.suffix.lower() in (".wav", ".flac")
    }
    segments = [
        (path, s, e)
        for path, samples in recordings.items()
        for s, e in split_at_silences(samples)
    ]
    labels = [
        analyzer.analyze(recordings[p][s:e])
        for p, s, e in tqdm(segments, unit="segment")
    ]

    items = defaultdict(list)
    for index, ((path, start, end), label) in enumerate(zip(segments, labels)):
        transcription = label.transcription
        if transcription.confidence >= min_confidence:
            item = Item(
                item_name(path.stem, f"{index:04d}"),
                recordings[path][start:end],
                transcription.phonemes,
                transcription.durations,
                label.notes,
            )
            items[label.notes is not None].append(item)
    if not items[True]:
        raise click.ClickException("No singing was found.")

    for sung, group in items.items():
        kind = "singing" if sung else "speech"
        minutes = sum(len(i.audio) for i in group) / SAMPLE_RATE / 60
        click.echo(f"{kind}: {minutes:.1f} min")
        dataset = RawDataset(raw_dir / kind, "voice", with_notes=sung)
        for item in group:
            dataset.add(item)
        dataset.save()


if __name__ == "__main__":
    main()
