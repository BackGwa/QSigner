import re
import subprocess
import sys
from pathlib import Path

import yaml

from qsinger.paths import SINGING_VOCODERS_ROOT, release_file
from qsinger.training.diffsinger import resolved_config

EXPERIMENT = "vocoder"
# generator and discriminator steps counted separately
UPDATES_PER_STEP = 2
VALIDATION_INTERVAL = 500  # checkpoint at each validation


def finetune(wav_dirs: list[Path], work_dir: Path, steps: int) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    config = resolved_config(
        SINGING_VOCODERS_ROOT / "configs" / "ft_hifigan.yaml",
        SINGING_VOCODERS_ROOT,
    )
    config.update(
        data_input_path=[str(d) for d in wav_dirs],
        data_out_path=[
            str(work_dir / "features" / str(i)) for i in range(len(wav_dirs))
        ],
        DataIndexPath=str(work_dir / "index"),
        val_num=5,
        finetune_ckpt_path=str(release_file("vocoder-finetune", "*.ckpt")),
        freezing_enabled=True,  # per SingingVocoders guide
        frozen_params=["discriminator.mpd"],
        max_updates=steps * UPDATES_PER_STEP,
        val_check_interval=VALIDATION_INTERVAL * UPDATES_PER_STEP,
        crop_mel_frames=32,  # for 16 GB GPUs
        batch_size=10,
    )
    config_path = work_dir / "vocoder.yaml"
    with open(config_path, "w", encoding="utf8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    if not (work_dir / "index").exists():
        _run(
            "process.py",
            ["--config", str(config_path), "--num_cpu", "4", "--strx", "1"],
            work_dir,
        )
    _run(
        "train.py",
        [
            "--config",
            str(config_path),
            "--exp_name",
            EXPERIMENT,
            "--work_dir",
            str(work_dir),
        ],
        work_dir,
    )

    # skip versioned copies from resume
    checkpoints = [
        p
        for p in (work_dir / EXPERIMENT).rglob("model_ckpt_steps_*.ckpt")
        if re.fullmatch(r"model_ckpt_steps_\d+", p.stem)
    ]
    latest = max(checkpoints, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    exported = work_dir / "vocoder.ckpt"  # generator only
    _run(
        "export_ckpt.py",
        ["--ckpt_path", str(latest), "--save_path", str(exported)],
        work_dir,
    )
    return exported


# fails for checkpoints outside cwd
def _run(script: str, args: list[str], work_dir: Path):
    subprocess.run(
        [sys.executable, str(SINGING_VOCODERS_ROOT / script), *args],
        cwd=work_dir,
        check=True,
    )
