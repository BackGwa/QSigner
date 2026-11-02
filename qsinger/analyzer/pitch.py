import sys

import numpy as np

from qsinger.data.audio import SAMPLE_RATE
from qsinger.paths import DIFFSINGER_ROOT, release_file

STEP = 0.01


class PitchExtractor:
    def __init__(self):
        sys.path.insert(0, str(DIFFSINGER_ROOT))
        from modules.pe.rmvpe import RMVPE

        self.rmvpe = RMVPE(str(release_file("rmvpe", "model.pt")))

    def f0(self, audio: np.ndarray) -> np.ndarray:
        f0 = self.rmvpe.infer_from_audio(audio, sample_rate=SAMPLE_RATE)
        n_steps = max(int(round(len(audio) / SAMPLE_RATE / STEP)), 1)
        return np.pad(f0[:n_steps], (0, max(n_steps - len(f0), 0)))
