import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from qsinger.data.audio import SAMPLE_RATE
from qsinger.data.phonemes import word_lengths

REST = "rest"


def item_name(*parts: str) -> str:
    return re.sub(r"[^\w\-]+", "_", "-".join(parts))


@dataclass
class Note:
    name: str
    duration: float


@dataclass
class Item:
    name: str
    audio: np.ndarray
    phonemes: list[str]
    durations: list[float]
    notes: list[Note] | None = None


class RawDataset:
    def __init__(self, root: Path, speaker: str, with_notes: bool):
        self.root = root
        self.speaker = speaker
        self.with_notes = with_notes
        self.rows = []
        (root / "wavs").mkdir(parents=True, exist_ok=True)

    def add(self, item: Item):
        sf.write(
            self.root / "wavs" / f"{item.name}.wav",
            item.audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        row = {
            "name": item.name,
            "ph_seq": " ".join(item.phonemes),
            "ph_dur": " ".join(f"{d:.6f}" for d in item.durations),
        }
        if self.with_notes:
            row["ph_num"] = " ".join(map(str, word_lengths(item.phonemes)))
            row["note_seq"] = " ".join(note.name for note in item.notes)
            row["note_dur"] = " ".join(
                f"{note.duration:.6f}" for note in item.notes
            )
        self.rows.append(row)

    def save(self):
        columns = ["name", "ph_seq", "ph_dur"] + (
            ["ph_num", "note_seq", "note_dur"] if self.with_notes else []
        )
        with open(
            self.root / "transcriptions.csv", "w", encoding="utf8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        info = {"speaker": self.speaker, "sung": self.with_notes}
        with open(self.root / "dataset.yaml", "w", encoding="utf8") as file:
            yaml.safe_dump(info, file)


@dataclass
class DatasetInfo:
    root: Path
    speaker: str
    sung: bool


def read_dataset_info(root: Path) -> DatasetInfo:
    with open(root / "dataset.yaml", encoding="utf8") as file:
        info = yaml.safe_load(file)
    return DatasetInfo(root, info["speaker"], info["sung"])
