import csv
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

from qsinger.data.audio import SAMPLE_RATE, load_audio
from qsinger.data.phonemes import BREATH
from qsinger.paths import BREATH_CHECKPOINT

STEP = 0.01
N_MELS = 80
HOP = int(SAMPLE_RATE * STEP)
CROP = 400
HELD_OUT_EVERY = 20


# level and flatness miss breaths
class BreathNet(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        layers, width = [], N_MELS
        for dilation in (1, 2, 4, 8):
            layers += [
                nn.Conv1d(
                    width, channels, 5, padding=2 * dilation, dilation=dilation
                ),
                nn.ReLU(),
            ]
            width = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(mel))[:, 0]


def log_mel(audio: np.ndarray, n_steps: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=40,
        fmax=16000,
    )
    mel = np.log(np.maximum(mel, 1e-5))[:, :n_steps]
    return np.pad(
        mel, ((0, 0), (0, n_steps - mel.shape[1])), mode="edge"
    ).astype(np.float32)


class BreathDetector:
    def __init__(self, checkpoint: Path = BREATH_CHECKPOINT):
        saved = torch.load(checkpoint)
        self.model = BreathNet().cuda().eval()
        self.model.load_state_dict(saved["model"])
        self.threshold = saved["threshold"]

    @torch.no_grad()
    def marks(self, audio: np.ndarray, n_steps: int) -> np.ndarray:
        mel = torch.from_numpy(log_mel(audio, n_steps))[None].cuda()
        return (
            torch.sigmoid(self.model(mel))[0].cpu().numpy() >= self.threshold
        )


def train(raw_dirs: list[Path], steps: int = 4000):
    items = _labeled_items(raw_dirs)
    held_out = items[HELD_OUT_EVERY - 1 :: HELD_OUT_EVERY]
    training = [
        item
        for i, item in enumerate(items)
        if i % HELD_OUT_EVERY != HELD_OUT_EVERY - 1
    ]
    print(f"| {len(training)} training items, {len(held_out)} held out")
    model = BreathNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positive = np.mean(np.concatenate([target for _, target in training]))
    # full class weight overmarks
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            np.sqrt((1 - positive) / positive), device="cuda"
        )
    )
    rng = random.Random(0)
    for step in range(1, steps + 1):
        mels, targets = zip(
            *(_crop(*rng.choice(training), rng) for _ in range(32))
        )
        logits = model(torch.from_numpy(np.stack(mels)).cuda())
        loss = loss_fn(logits, torch.from_numpy(np.stack(targets)).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"| step {step} loss {loss.item():.3f}", flush=True)
    threshold = _choose_threshold(model.eval(), held_out)
    BREATH_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "threshold": threshold},
        BREATH_CHECKPOINT,
    )


def _labeled_items(
    raw_dirs: list[Path],
) -> list[tuple[np.ndarray, np.ndarray]]:
    items = []
    for raw_dir in raw_dirs:
        with open(raw_dir / "transcriptions.csv", encoding="utf8") as file:
            rows = list(csv.DictReader(file))
        if not any(BREATH in row["ph_seq"].split() for row in rows):
            continue
        for row in rows:
            target = []
            for phoneme, duration in zip(
                row["ph_seq"].split(), row["ph_dur"].split()
            ):
                target += [phoneme == BREATH] * int(
                    round(float(duration) / STEP)
                )
            audio = load_audio(raw_dir / "wavs" / f"{row['name']}.wav")
            items.append(
                (
                    log_mel(audio, len(target)),
                    np.array(target, dtype=np.float32),
                )
            )
        print(f"| {raw_dir.name}: {len(rows)} items")
    return items


def _crop(mel: np.ndarray, target: np.ndarray, rng: random.Random):
    if len(target) < CROP:
        pad = CROP - len(target)
        return np.pad(mel, ((0, 0), (0, pad)), mode="edge"), np.pad(
            target, (0, pad)
        )
    start = rng.randrange(len(target) - CROP + 1)
    return mel[:, start : start + CROP], target[start : start + CROP]


@torch.no_grad()
def _choose_threshold(model, held_out) -> float:
    probabilities = [
        torch.sigmoid(model(torch.from_numpy(mel)[None].cuda()))[0]
        .cpu()
        .numpy()
        for mel, _ in held_out
    ]
    best = (0.0, 0.5)
    for threshold in np.arange(0.3, 0.96, 0.05):
        hits = found = labeled = 0
        for p, (_, target) in zip(probabilities, held_out):
            predicted, truth = p >= threshold, target > 0.5
            hits += int((predicted & truth).sum())
            found += int(predicted.sum())
            labeled += int(truth.sum())
        precision, recall = hits / max(found, 1), hits / max(labeled, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(
            f"| threshold {threshold:.2f} precision {precision:.1%} "
            f"recall {recall:.1%} F1 {f1:.2f}"
        )
        best = max(best, (f1, float(threshold)))
    print(f"| chosen threshold {best[1]:.2f}")
    return best[1]
