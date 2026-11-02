from dataclasses import dataclass

import librosa
import numpy as np

from qsinger.analyzer.breath import BreathDetector
from qsinger.analyzer.recognizer import BLANK_ID, PhoneRecognizer
from qsinger.data.audio import SAMPLE_RATE
from qsinger.data.phonemes import (
    BREATH,
    IPA_CHARACTERS,
    SILENCE,
    VOWELS,
    from_ipa,
    split_phones,
)

STEP = 0.01
SILENCE_DB = -40.0  # relative to loudest frame
MIN_SILENCE = 0.1
MIN_BREATH_IN_PAUSE = 0.1
MIN_BREATH_AFTER_VOWEL = 0.15  # keeps devoiced vowels
CONSONANT_LENGTH = 0.15  # consonant noise
_PAUSE, _BREATH = -1, -3


@dataclass
class Transcription:
    phonemes: list[str]
    durations: list[float]
    confidence: float


# CTC blanks cover held vowels
class Transcriber:
    def __init__(self, recognizer: PhoneRecognizer, breath: BreathDetector):
        self.recognizer = recognizer
        self.breath = breath
        self.allowed = np.array(
            [BLANK_ID]
            + [
                i
                for i, token in enumerate(recognizer.tokens)
                if token in IPA_CHARACTERS
            ]
        )

    def transcribe(self, audio: np.ndarray, f0: np.ndarray) -> Transcription:
        log_probs, period = self.recognizer.log_probs(audio)
        best = self.allowed[log_probs[:, self.allowed].argmax(axis=1)]
        tokens, frames = _collapse(best)
        phones = split_phones([self.recognizer.tokens[t] for t in tokens])
        starts, phonemes, dropped = from_ipa([phone for _, phone in phones])
        onsets = [frames[phones[s][0]] * period for s in starts]
        mapped_ratio = 1 - dropped / len(phones) if phones else 0.0
        confidence = (
            float(np.mean(np.exp(log_probs[frames, tokens])))
            if tokens
            else 0.0
        )
        phonemes, durations = _timeline(
            phonemes, onsets, audio, f0, self._breath_marks(audio)
        )
        return Transcription(phonemes, durations, confidence * mapped_ratio)

    def _breath_marks(self, audio: np.ndarray) -> np.ndarray:
        return self.breath.marks(
            audio, max(int(round(len(audio) / SAMPLE_RATE / STEP)), 1)
        )


def _collapse(path: np.ndarray) -> tuple[list[int], list[int]]:
    tokens, frames = [], []
    previous = BLANK_ID
    for frame, token in enumerate(path):
        if token != BLANK_ID and token != previous:
            tokens.append(int(token))
            frames.append(frame)
        previous = token
    return tokens, frames


def _timeline(
    phonemes: list[str],
    onsets: list[float],
    audio: np.ndarray,
    f0: np.ndarray,
    breath_marks: np.ndarray,
) -> tuple[list[str], list[float]]:
    n_steps = max(int(round(len(audio) / SAMPLE_RATE / STEP)), 1)
    owner = np.full(n_steps, _PAUSE)
    onset_steps = [
        min(int(round(onset / STEP)), n_steps - 1) for onset in onsets
    ]
    for i, onset_step in enumerate(onset_steps):
        owner[onset_step:] = i
    silent = _silent_steps(audio, n_steps)
    breath = _breath_steps(breath_marks, f0, owner, onset_steps, phonemes)
    for step in range(n_steps):
        resumed = (
            step > 0
            and owner[step - 1] < 0
            and owner[step] == _last_owner(owner, step)
        )
        if breath[step]:
            owner[step] = _BREATH
        elif silent[step] or resumed:
            owner[step] = _PAUSE

    labels, durations = [], []
    for step, index in enumerate(owner):
        label = {_PAUSE: SILENCE, _BREATH: BREATH}.get(index) or phonemes[
            index
        ]
        if step > 0 and index == owner[step - 1]:
            durations[-1] += STEP
        else:
            labels.append(label)
            durations.append(STEP)
    durations[-1] += len(audio) / SAMPLE_RATE - sum(durations)
    return labels, [round(d, 6) for d in durations]


def _last_owner(owner: np.ndarray, step: int) -> int:
    for previous in owner[step - 1 :: -1]:
        if previous >= 0:
            return previous
    return -2


def _breath_steps(marks, f0, owner, onset_steps, phonemes) -> np.ndarray:
    n_steps = len(owner)
    voiced = np.pad(f0[:n_steps] > 0, (0, max(n_steps - len(f0), 0)))
    candidate = ~voiced & np.pad(
        marks[:n_steps], (0, max(n_steps - len(marks), 0))
    )

    breath = np.zeros(n_steps, dtype=bool)
    start = None
    for step in range(n_steps + 1):
        same_run = (
            step < n_steps
            and candidate[step]
            and (start is None or owner[step] == owner[start])
        )
        if same_run and start is None:
            start = step
        elif not same_run and start is not None:
            index = owner[start]
            if index < 0:
                begin, minimum = start, MIN_BREATH_IN_PAUSE
            elif phonemes[index] in VOWELS:
                begin, minimum = start, MIN_BREATH_AFTER_VOWEL
            else:
                begin, minimum = (
                    max(
                        start,
                        onset_steps[index] + int(CONSONANT_LENGTH / STEP),
                    ),
                    MIN_BREATH_IN_PAUSE,
                )
            if (step - begin) * STEP >= minimum:
                breath[begin:step] = True
            start = step if step < n_steps and candidate[step] else None
    return breath


def _levels(audio: np.ndarray, n_steps: int) -> np.ndarray:
    hop = int(SAMPLE_RATE * STEP)
    rms = librosa.feature.rms(
        y=audio, frame_length=hop * 4, hop_length=hop, center=True
    )[0][:n_steps]
    rms = np.pad(rms, (0, n_steps - len(rms)), mode="edge")
    return 20 * np.log10(np.maximum(rms, 1e-10))


def _silent_steps(audio: np.ndarray, n_steps: int) -> np.ndarray:
    db = _levels(audio, n_steps)
    quiet = db < db.max() + SILENCE_DB
    silent = np.zeros(n_steps, dtype=bool)
    min_steps = int(MIN_SILENCE / STEP)
    start = None
    for step, is_quiet in enumerate(np.append(quiet, False)):
        if is_quiet and start is None:
            start = step
        elif not is_quiet and start is not None:
            if step - start >= min_steps:
                silent[start:step] = True
            start = None
    return silent
