import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFSINGER_ROOT = REPO_ROOT / "third_party" / "DiffSinger"
SINGING_VOCODERS_ROOT = REPO_ROOT / "third_party" / "SingingVocoders"
ZIPA_ROOT = REPO_ROOT / "third_party" / "zipa"
ICEFALL_ROOT = REPO_ROOT / "third_party" / "icefall"
WORK_DIR = REPO_ROOT / "work"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
BASE_DIR = CHECKPOINT_DIR / "base"
BREATH_CHECKPOINT = CHECKPOINT_DIR / "breath.pt"
SINGING_CHECKPOINT = CHECKPOINT_DIR / "singing.pt"
RECOGNIZER_CHECKPOINT = CHECKPOINT_DIR / "recognizer.onnx"
ZIPA_REPO = "anyspeech/zipa-small-crctc-300k"
RELEASES = {
    "rmvpe": "https://github.com/openvpi/DiffSinger/releases/download/"
    "v2.1.0/rmvpe.zip",
    "hnsep": "https://github.com/yxlllc/vocal-remover/releases/download/"
    "hnsep_240512/hnsep_240512.zip",
    "vocoder": "https://github.com/openvpi/vocoders/releases/download/"
    "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip",
    "vocoder-finetune": "https://github.com/openvpi/SingingVocoders/releases/"
    "download/v1.0.0/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip",
    "game": "https://github.com/openvpi/GAME/releases/download/"
    "v1.0.3/GAME-1.0.3-small-onnx.zip",
}


def release_file(name: str, pattern: str) -> Path:
    folder = CHECKPOINT_DIR / name
    if not folder.exists():
        archive, _ = urllib.request.urlretrieve(RELEASES[name])
        shutil.unpack_archive(archive, folder, "zip")
    return next(folder.rglob(pattern))
