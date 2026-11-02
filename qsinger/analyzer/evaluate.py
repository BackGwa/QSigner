from pathlib import Path

import kaldialign
import numpy as np

from qsinger.analyzer.breath import BreathDetector
from qsinger.analyzer.pitch import STEP, PitchExtractor
from qsinger.analyzer.recognizer import PhoneRecognizer
from qsinger.analyzer.transcriber import Transcriber
from qsinger.data.audio import load_audio
from qsinger.data.phonemes import BREATH, GLOBAL_PHONEMES

EXAMPLES = 3
GAP = "*"


def report(held_out: list[tuple[Path, list[str], list[float]]]):
    transcriber = Transcriber(PhoneRecognizer(), BreathDetector())
    pitch = PitchExtractor()
    edits = length = breath_hits = breath_found = breath_labeled = 0
    onset_errors = []
    for index, (wav, phonemes, durations) in enumerate(held_out):
        audio = load_audio(wav)
        result = transcriber.transcribe(audio, pitch.f0(audio))
        reference = _onsets(phonemes, durations)
        hypothesis = _onsets(result.phonemes, result.durations)
        item_edits, matches = _align(
            [p for p, _ in reference], [p for p, _ in hypothesis]
        )
        edits += item_edits
        length += len(reference)
        onset_errors += [
            abs(reference[i][1] - hypothesis[j][1]) for i, j in matches
        ]
        labeled, found = _breath_steps(phonemes, durations), _breath_steps(
            result.phonemes, result.durations
        )
        n = min(len(labeled), len(found))
        breath_hits += int(np.sum(labeled[:n] & found[:n]))
        breath_found += int(found.sum())
        breath_labeled += int(labeled.sum())
        if index < EXAMPLES:
            print(wav.stem)
            print("  label    ", " ".join(phonemes))
            print("  analyzer ", " ".join(result.phonemes))
    print(f"phoneme error rate  {edits / max(length, 1):.1%}")
    print(f"mean onset error    {np.mean(onset_errors) * 1000:.0f} ms")
    print(f"breath precision    {breath_hits / max(breath_found, 1):.1%}")
    print(f"breath recall       {breath_hits / max(breath_labeled, 1):.1%}")


def _onsets(phonemes, durations) -> list[tuple[str, float]]:
    starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    return [
        (p, float(s))
        for p, s in zip(phonemes, starts)
        if p not in GLOBAL_PHONEMES
    ]


def _breath_steps(phonemes, durations) -> np.ndarray:
    steps = []
    for phoneme, duration in zip(phonemes, durations):
        steps += [phoneme == BREATH] * int(round(duration / STEP))
    return np.array(steps, dtype=bool)


def _align(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, list[tuple[int, int]]]:
    edits, matches, i, j = 0, [], 0, 0
    for ref, hyp in kaldialign.align(reference, hypothesis, GAP):
        if ref == hyp:
            matches.append((i, j))
        else:
            edits += 1
        i += ref != GAP
        j += hyp != GAP
    return edits, matches
