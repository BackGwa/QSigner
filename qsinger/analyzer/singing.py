import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qsinger.analyzer.breath import CROP, HOP, BreathNet, log_mel
from qsinger.data.audio import load_audio
from qsinger.data.dataset import read_dataset_info
from qsinger.paths import SINGING_CHECKPOINT

ITEMS_PER_DATASET = 300
HELD_OUT_EVERY = 10
# speech taken as singing costs more than missed singing
MAX_SPEECH_ERROR = 0.02


class SingingNet(BreathNet):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return super().forward(mel).mean(dim=1)


class SingingDetector:
    def __init__(self, checkpoint: Path = SINGING_CHECKPOINT):
        saved = torch.load(checkpoint)
        self.model = SingingNet().cuda().eval()
        self.model.load_state_dict(saved["model"])
        self.threshold = saved["threshold"]

    @torch.no_grad()
    def is_singing(self, audio: np.ndarray) -> bool:
        mel = torch.from_numpy(log_mel(audio, len(audio) // HOP))[None].cuda()
        return torch.sigmoid(self.model(mel)).item() >= self.threshold


# needs singing and speech from the same singers
def train(raw_dirs: list[Path], steps: int = 3000):
    items = _labeled_items(raw_dirs)
    held_out = items[HELD_OUT_EVERY - 1 :: HELD_OUT_EVERY]
    training = [
        item
        for i, item in enumerate(items)
        if i % HELD_OUT_EVERY != HELD_OUT_EVERY - 1
    ]
    by_label = {
        label: [mel for mel, sung in training if sung == label]
        for label in (0.0, 1.0)
    }
    model = SingingNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    targets = torch.tensor([1.0] * 16 + [0.0] * 16, device="cuda")
    for step in range(1, steps + 1):
        mels = [
            _crop(rng.choice(by_label[label]), rng)
            for label in [1.0] * 16 + [0.0] * 16
        ]
        loss = loss_fn(model(torch.from_numpy(np.stack(mels)).cuda()), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"| step {step} loss {loss.item():.3f}", flush=True)
    threshold = _choose_threshold(model.eval(), held_out)
    SINGING_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "threshold": threshold},
        SINGING_CHECKPOINT,
    )


def _labeled_items(raw_dirs: list[Path]) -> list[tuple[np.ndarray, float]]:
    rng = random.Random(0)
    items = []
    for raw_dir in raw_dirs:
        sung = float(read_dataset_info(raw_dir).sung)
        wavs = sorted((raw_dir / "wavs").glob("*.wav"))
        for wav in rng.sample(wavs, min(ITEMS_PER_DATASET, len(wavs))):
            audio = load_audio(wav)
            items.append((log_mel(audio, len(audio) // HOP), sung))
        print(f"| {raw_dir.name}: {'sung' if sung else 'spoken'}", flush=True)
    rng.shuffle(items)
    return items


def _crop(mel: np.ndarray, rng: random.Random) -> np.ndarray:
    if mel.shape[1] < CROP:
        return np.pad(mel, ((0, 0), (0, CROP - mel.shape[1])), mode="edge")
    start = rng.randrange(mel.shape[1] - CROP + 1)
    return mel[:, start : start + CROP]


@torch.no_grad()
def _choose_threshold(model, held_out) -> float:
    probabilities = np.array(
        [
            torch.sigmoid(model(torch.from_numpy(mel)[None].cuda())).item()
            for mel, _ in held_out
        ]
    )
    sung = np.array([label for _, label in held_out]) > 0.5
    chosen = 0.95
    for threshold in np.arange(0.95, 0.04, -0.05):
        speech_error = np.mean(probabilities[~sung] >= threshold)
        recall = np.mean(probabilities[sung] >= threshold)
        print(
            f"| threshold {threshold:.2f} singing found {recall:.1%} "
            f"speech taken as singing {speech_error:.1%}"
        )
        if speech_error <= MAX_SPEECH_ERROR:
            chosen = float(threshold)
    print(f"| chosen threshold {chosen:.2f}")
    return chosen
