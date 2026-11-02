import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import sentencepiece
import torch
from huggingface_hub import hf_hub_download
from lhotse import CutSet, Fbank, FbankConfig, Recording, SupervisionSegment
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
)

from qsinger.data.phonemes import GLOBAL_PHONEMES, to_ipa
from qsinger.paths import (
    ICEFALL_ROOT,
    RECOGNIZER_CHECKPOINT,
    WORK_DIR,
    ZIPA_REPO,
    ZIPA_ROOT,
)

RECIPE = ZIPA_ROOT / "zipformer_crctc"
EXP_DIR = WORK_DIR / "zipa"
HELD_OUT_EVERY = 50
EPOCH_NAME = 99  # loaded by ZIPA export
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_BATCH_SECONDS = 600.0


def finetune(raw_dirs: list[Path]) -> list[tuple]:
    sys.path[:0] = [str(ICEFALL_ROOT), str(RECIPE)]
    from train import get_model, get_params, get_parser

    params = get_params()
    params.update(vars(get_parser().parse_args([])))
    params.update(
        use_transducer=False,
        use_ctc=True,
        use_cr_ctc=False,
        use_attention_decoder=False,
    )
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(ZIPA_ROOT / "ipa_simplified" / "unigram_127.model")
    )
    params.blank_id = tokenizer.piece_to_id("<blk>")
    params.vocab_size = tokenizer.get_piece_size()
    model = get_model(params)
    state = torch.load(
        hf_hub_download(ZIPA_REPO, "zipa_small_crctc_300000_avg10.pth"),
        map_location="cpu",
    )
    model.load_state_dict(state.get("model", state), strict=False)
    model.cuda().train()

    cuts, held_out = _cuts(raw_dirs)
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        ),
    )
    sampler = DynamicBucketingSampler(
        cuts, max_duration=MAX_BATCH_SECONDS, shuffle=True, num_buckets=10
    )
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=None, num_workers=os.cpu_count()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            targets = tokenizer.encode(
                batch["supervisions"]["text"], out_type=int
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoder_out, encoder_out_lens = model.forward_encoder(
                    batch["inputs"].cuda(),
                    batch["supervisions"]["num_frames"].cuda(),
                )
            loss = model.forward_ctc(
                encoder_out.float(),
                encoder_out_lens,
                torch.tensor([t for target in targets for t in target]),
                torch.tensor([len(target) for target in targets]),
            ) / len(targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step % 50 == 0:
                print(
                    f"| epoch {epoch} step {step} ctc loss {loss.item():.3f}"
                )

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict()}, EXP_DIR / f"epoch-{EPOCH_NAME}.pt"
    )
    subprocess.run(
        [
            sys.executable,
            str(RECIPE / "export-onnx-ctc.py"),
            "--epoch",
            str(EPOCH_NAME),
            "--avg",
            "1",
            "--use-averaged-model",
            "False",
            "--exp-dir",
            str(EXP_DIR),
            "--tokens",
            hf_hub_download(ZIPA_REPO, "tokens.txt"),
            "--use-transducer",
            "False",
            "--use-ctc",
            "True",
        ],
        cwd=ZIPA_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ICEFALL_ROOT), str(RECIPE)]),
        },
        check=True,
    )
    shutil.copy(EXP_DIR / "model.onnx", RECOGNIZER_CHECKPOINT)
    return held_out


def _cuts(raw_dirs: list[Path]) -> tuple[CutSet, list[tuple]]:
    cuts, held_out = [], []
    for raw_dir in raw_dirs:
        with open(raw_dir / "transcriptions.csv", encoding="utf8") as file:
            rows = list(csv.DictReader(file))
        for index, row in enumerate(rows):
            wav = raw_dir / "wavs" / f"{row['name']}.wav"
            phonemes = [
                p for p in row["ph_seq"].split() if p not in GLOBAL_PHONEMES
            ]
            if index % HELD_OUT_EVERY == HELD_OUT_EVERY - 1:
                held_out.append(
                    (
                        wav,
                        row["ph_seq"].split(),
                        [float(d) for d in row["ph_dur"].split()],
                    )
                )
            elif phonemes:
                recording = Recording.from_file(wav, recording_id=row["name"])
                cut = recording.to_cut()
                cut.supervisions = [
                    SupervisionSegment(
                        id=row["name"],
                        recording_id=row["name"],
                        start=0.0,
                        duration=recording.duration,
                        text="".join(to_ipa(phonemes)),
                    )
                ]
                cuts.append(cut.resample(16000))
    random.Random(0).shuffle(cuts)
    return CutSet.from_cuts(cuts), held_out
