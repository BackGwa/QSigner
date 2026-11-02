import json

import librosa
import numpy as np
import onnxruntime

from qsinger.data.audio import SAMPLE_RATE
from qsinger.data.dataset import REST, Note
from qsinger.paths import release_file

# GAME defaults
BOUNDARY_THRESHOLD = 0.2
BOUNDARY_RADIUS = 0.02
D3PM_STEPS = 8
PRESENCE_THRESHOLD = 0.2


class NoteEstimator:
    def __init__(self):
        config = release_file("game", "config.json")
        self.config = json.loads(config.read_text(encoding="utf8"))
        self.encoder, self.segmenter, self.estimator, self.bd2dur = (
            onnxruntime.InferenceSession(
                config.with_name(f"{name}.onnx"),
                providers=["CUDAExecutionProvider"],
            )
            for name in ("encoder", "segmenter", "estimator", "bd2dur")
        )

    def estimate(self, audio: np.ndarray) -> list[Note]:
        rate = self.config["samplerate"]
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=rate)
        duration = np.array([len(audio) / rate], dtype=np.float32)
        x_seg, x_est, mask_t = self.encoder.run(
            None,
            {
                "waveform": audio[np.newaxis].astype(np.float32),
                "duration": duration,
            },
        )
        known = np.zeros_like(mask_t, dtype=bool)
        boundaries = self._segment(x_seg, known, mask_t)
        durations, mask_n = self.bd2dur.run(
            None, {"boundaries": boundaries, "maskT": mask_t}
        )
        presence, scores = self.estimator.run(
            None,
            {
                "x_est": x_est,
                "boundaries": boundaries,
                "maskT": mask_t,
                "maskN": mask_n,
                "threshold": np.array(PRESENCE_THRESHOLD, dtype=np.float32),
            },
        )
        names = librosa.midi_to_note(scores[0], unicode=False)
        notes = [
            Note(name if present else REST, float(length))
            for length, present, name, valid in zip(
                durations[0], presence[0], names, mask_n[0]
            )
            if valid
        ]
        notes[-1].duration += float(duration[0]) - sum(
            note.duration for note in notes
        )
        return notes

    def _segment(self, x_seg, known, mask_t) -> np.ndarray:
        inputs = {
            "x_seg": x_seg,
            "known_boundaries": known,
            "maskT": mask_t,
            "threshold": np.array(BOUNDARY_THRESHOLD, dtype=np.float32),
            "radius": np.array(
                round(BOUNDARY_RADIUS / self.config["timestep"]),
                dtype=np.int64,
            ),
        }
        if self.config.get("languages"):
            inputs["language"] = np.array(
                [self.config["languages"].get("ja", 0)], dtype=np.int64
            )
        if not self.config["loop"]:
            return self.segmenter.run(None, inputs)[0]
        boundaries = known
        for t in np.arange(D3PM_STEPS, dtype=np.float32) / D3PM_STEPS:
            inputs["prev_boundaries"] = boundaries
            inputs["t"] = np.array([t], dtype=np.float32)
            boundaries = self.segmenter.run(None, inputs)[0]
        return boundaries
