from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 44100
HOP = SAMPLE_RATE // 100
WINDOW = HOP * 4
SILENCE_DB = -40.0  # relative to loudest frame
MIN_SILENCE = 0.3
KEEP_SILENCE = 0.1
MIN_SEGMENT = 1.0
MAX_SEGMENT = 15.0


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def split_at_silences(audio: np.ndarray) -> list[tuple[int, int]]:
    rms = librosa.feature.rms(
        y=audio, frame_length=WINDOW, hop_length=HOP, center=True
    )[0]
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    silent = db < db.max() + SILENCE_DB

    segments = []
    for start, end in _sound_regions(silent):
        segments.extend(_limit_length(start, end, db))
    min_frames = int(MIN_SEGMENT * 100)
    n_samples = len(audio)
    return [
        (start * HOP, min(end * HOP, n_samples))
        for start, end in segments
        if end - start >= min_frames
    ]


def _sound_regions(silent: np.ndarray) -> list[tuple[int, int]]:
    min_silence, keep = int(MIN_SILENCE * 100), int(KEEP_SILENCE * 100)
    regions, region_start, silence_start = [], None, None
    for i, is_silent in enumerate(np.append(silent, True)):
        if not is_silent:
            if region_start is None:
                region_start = max(i - keep, 0)
            silence_start = None
            continue
        if silence_start is None:
            silence_start = i
        if region_start is not None and (
            i - silence_start >= min_silence or i == len(silent)
        ):
            regions.append(
                (region_start, min(silence_start + keep, len(silent)))
            )
            region_start = None
    return regions


def _limit_length(
    start: int, end: int, db: np.ndarray
) -> list[tuple[int, int]]:
    max_frames = int(MAX_SEGMENT * 100)
    pieces = []
    while end - start > max_frames:
        low, high = start + max_frames // 2, start + max_frames
        cut = low + int(np.argmin(db[low:high]))
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces
