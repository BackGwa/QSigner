import os
import subprocess
import sys
from pathlib import Path

import yaml

from qsinger.data.dataset import DatasetInfo
from qsinger.data.phonemes import IPA
from qsinger.paths import DIFFSINGER_ROOT, REPO_ROOT, release_file

VALIDATION_ITEMS = 2
VOICE_BINARIZERS = {
    "preprocessing.acoustic_binarizer.AcousticBinarizer": (
        "qsinger.training.binarizers.VoiceAcousticBinarizer"
    ),
    "preprocessing.variance_binarizer.VarianceBinarizer": (
        "qsinger.training.binarizers.VoiceVarianceBinarizer"
    ),
}


def acoustic_config(datasets: list[DatasetInfo], binary_dir: Path) -> dict:
    config = _template("config_acoustic.yaml", datasets, binary_dir)
    config.update(
        vocoder="NsfHifiGAN",
        vocoder_ckpt=str(release_file("vocoder", "model.ckpt")),
        use_energy_embed=True,
        use_breathiness_embed=True,
        use_voicing_embed=True,
        use_tension_embed=False,
        use_key_shift_embed=False,
        use_speed_embed=False,
    )
    config["augmentation_args"]["random_pitch_shifting"]["enabled"] = False
    config["augmentation_args"]["random_time_stretching"]["enabled"] = False
    return config


def variance_config(datasets: list[DatasetInfo], binary_dir: Path) -> dict:
    config = _template(
        "config_variance.yaml", [d for d in datasets if d.sung], binary_dir
    )
    config.update(
        predict_dur=True,
        predict_pitch=True,
        predict_energy=True,
        predict_breathiness=True,
        predict_voicing=True,
        predict_tension=False,
    )
    return config


CONFIGS = {"acoustic": acoustic_config, "variance": variance_config}


def finetuned(
    config: dict, checkpoint: Path, max_updates: int, learning_rate: float
) -> dict:
    config.update(
        use_spk_id=False,
        num_spk=1,
        finetune_enabled=True,
        finetune_ckpt_path=str(checkpoint),
        finetune_ignored_params=["model.fs2.spk_embed", "model.spk_embed"],
        finetune_strict_shapes=False,
        max_updates=max_updates,
        optimizer_args={**config["optimizer_args"], "lr": learning_rate},
        binarizer_cls=VOICE_BINARIZERS[config["binarizer_cls"]],
    )
    return config


def _template(
    template: str, datasets: list[DatasetInfo], binary_dir: Path
) -> dict:
    config = resolved_config(
        DIFFSINGER_ROOT / "configs" / "templates" / template
    )
    speakers = sorted({d.speaker for d in datasets})
    config.update(
        dictionaries={"ja": _write_dictionary(binary_dir.parent)},
        datasets=[_dataset_entry(d, speakers) for d in datasets],
        binary_data_dir=str(binary_dir),
        pe="rmvpe",
        pe_ckpt=str(release_file("rmvpe", "model.pt")),
        hnsep="vr",
        hnsep_ckpt=str(release_file("hnsep", "model.pt")),
        use_spk_id=len(speakers) > 1,
        num_spk=len(speakers),
    )
    return config


# identity dictionary kept outside the binary folder
def _write_dictionary(folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dictionary.txt"
    path.write_text(
        "".join(f"{ph}\t{ph}\n" for ph in sorted(IPA)), encoding="utf8"
    )
    return str(path)


# scripts run outside their folder
def resolved_config(path: Path, root: Path = DIFFSINGER_ROOT) -> dict:
    with open(path, encoding="utf8") as file:
        config = yaml.safe_load(file)
    bases = config.pop("base_config", [])
    merged = {}
    for base in [bases] if isinstance(bases, str) else bases:
        _override(merged, resolved_config(root / base, root))
    _override(merged, config)
    return merged


def _override(old: dict, new: dict):
    for key, value in new.items():
        if isinstance(value, dict) and key in old:
            _override(old[key], value)
        else:
            old[key] = value


def _dataset_entry(dataset: DatasetInfo, speakers: list[str]) -> dict:
    with open(dataset.root / "transcriptions.csv", encoding="utf8") as file:
        names = [
            line.split(",", 1)[0]
            for line in file.readlines()[1 : 1 + VALIDATION_ITEMS]
        ]
    return {
        "raw_data_dir": str(dataset.root),
        "speaker": dataset.speaker,
        "spk_id": speakers.index(dataset.speaker),
        "language": "ja",
        "test_prefixes": names,
    }


# rerun resumes training
def fit(config: dict, work_dir: Path, experiment: str):
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / f"{experiment}.yaml"
    with open(config_path, "w", encoding="utf8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    if not (Path(config["binary_data_dir"]) / "train.data").exists():
        _run("binarize.py", ["--config", str(config_path)], work_dir)
    _run(
        "train.py",
        ["--config", str(config_path), "--exp_name", experiment],
        work_dir,
    )


def latest_checkpoint(work_dir: Path, experiment: str) -> Path:
    checkpoints = (work_dir / "checkpoints" / experiment).glob(
        "model_ckpt_steps_*.ckpt"
    )
    return max(checkpoints, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def _run(script: str, args: list[str], work_dir: Path):
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
    }  # for qsinger binarizers
    subprocess.run(
        [sys.executable, str(DIFFSINGER_ROOT / "scripts" / script), *args],
        cwd=work_dir,
        env=env,
        check=True,
    )
