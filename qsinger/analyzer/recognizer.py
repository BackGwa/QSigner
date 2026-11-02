import librosa
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from lhotse.features.kaldi.extractors import Fbank, FbankConfig

from qsinger.data.audio import SAMPLE_RATE
from qsinger.paths import RECOGNIZER_CHECKPOINT, ZIPA_REPO

ZIPA_SAMPLE_RATE = 16000
BLANK_ID = 0


class PhoneRecognizer:
    def __init__(self):
        self.session = onnxruntime.InferenceSession(
            RECOGNIZER_CHECKPOINT, providers=["CUDAExecutionProvider"]
        )
        self.tokens = _load_tokens(hf_hub_download(ZIPA_REPO, "tokens.txt"))
        # ZIPA training features
        self.fbank = Fbank(
            FbankConfig(num_filters=80, dither=0.0, snip_edges=False)
        )

    def log_probs(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        audio = librosa.resample(
            audio, orig_sr=SAMPLE_RATE, target_sr=ZIPA_SAMPLE_RATE
        )
        features = self.fbank.extract(audio, ZIPA_SAMPLE_RATE).astype(
            np.float32
        )
        outputs = self.session.run(
            None,
            {
                "x": features[np.newaxis],
                "x_lens": np.array([len(features)], dtype=np.int64),
            },
        )
        log_probs = outputs[0][0]
        return log_probs, len(audio) / ZIPA_SAMPLE_RATE / len(log_probs)


def _load_tokens(path: str) -> list[str]:
    tokens = {}
    with open(path, encoding="utf8") as file:
        for line in file:
            token, index = line.rstrip("\n").rsplit(" ", 1)
            tokens[int(index)] = token
    return [tokens[i] for i in range(len(tokens))]
