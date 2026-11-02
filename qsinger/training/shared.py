import csv
import shutil
import sys
from pathlib import Path

from qsinger.analyzer.breath import BreathDetector
from qsinger.analyzer.pitch import PitchExtractor
from qsinger.analyzer.recognizer import PhoneRecognizer
from qsinger.analyzer.transcriber import Transcriber
from qsinger.data.audio import load_audio
from qsinger.data.dataset import (
    DatasetInfo,
    Item,
    Note,
    RawDataset,
    read_dataset_info,
)
from qsinger.paths import BASE_DIR, WORK_DIR
from qsinger.training import diffsinger

SHARED_DIR = WORK_DIR / "shared"
EXPERIMENTS = {"acoustic": "base-acoustic", "variance": "base-variance"}


def main(data_dir: Path):
    sources = [read_dataset_info(d) for d in sorted(data_dir.iterdir())]
    datasets = sources + _relabeled(sources)
    for model, experiment in EXPERIMENTS.items():
        diffsinger.fit(
            diffsinger.CONFIGS[model](datasets, SHARED_DIR / "binary" / model),
            SHARED_DIR,
            experiment,
        )
        (BASE_DIR / model).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            diffsinger.latest_checkpoint(SHARED_DIR, experiment),
            BASE_DIR / model / "model.ckpt",
        )
        shutil.copy(
            SHARED_DIR / "checkpoints" / experiment / "config.yaml",
            BASE_DIR / model / "config.yaml",
        )


# lets shared models learn analyzer labels
def _relabeled(sources: list[DatasetInfo]) -> list[DatasetInfo]:
    transcriber = Transcriber(PhoneRecognizer(), BreathDetector())
    pitch = PitchExtractor()
    relabeled = []
    for source in sources:
        target = SHARED_DIR / "relabeled" / source.root.name
        if source.sung:
            if not target.exists():
                _relabel(source, target, transcriber, pitch)
            relabeled.append(read_dataset_info(target))
    return relabeled


def _relabel(
    source: DatasetInfo,
    target: Path,
    transcriber: Transcriber,
    pitch: PitchExtractor,
):
    with open(source.root / "transcriptions.csv", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    dataset = RawDataset(target, source.speaker, with_notes=True)
    for row in rows:
        samples = load_audio(source.root / "wavs" / f"{row['name']}.wav")
        transcription = transcriber.transcribe(samples, pitch.f0(samples))
        notes = [
            Note(n, float(d))
            for n, d in zip(row["note_seq"].split(), row["note_dur"].split())
        ]
        dataset.add(
            Item(
                row["name"],
                samples,
                transcription.phonemes,
                transcription.durations,
                notes,
            )
        )
    dataset.save()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
