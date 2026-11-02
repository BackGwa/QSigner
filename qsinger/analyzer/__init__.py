from dataclasses import dataclass

import numpy as np

from qsinger.analyzer.breath import BreathDetector
from qsinger.analyzer.notes import NoteEstimator
from qsinger.analyzer.pitch import PitchExtractor
from qsinger.analyzer.recognizer import PhoneRecognizer
from qsinger.analyzer.singing import SingingDetector
from qsinger.analyzer.transcriber import Transcriber, Transcription
from qsinger.data.dataset import REST, Note


@dataclass
class SegmentLabel:
    transcription: Transcription
    notes: list[Note] | None


class Analyzer:
    def __init__(self):
        self.pitch = PitchExtractor()
        self.transcriber = Transcriber(PhoneRecognizer(), BreathDetector())
        self.notes = NoteEstimator()
        self.singing = SingingDetector()

    def analyze(self, audio: np.ndarray) -> SegmentLabel:
        transcription = self.transcriber.transcribe(
            audio, self.pitch.f0(audio)
        )
        notes = None
        if self.singing.is_singing(audio):
            estimated = self.notes.estimate(audio)
            # variance training rejects all rests
            if any(note.name != REST for note in estimated):
                notes = estimated
        return SegmentLabel(transcription, notes)
