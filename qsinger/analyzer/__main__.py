import sys
from pathlib import Path

from qsinger.analyzer import breath, evaluate, recognizer_finetune, singing

raw_dirs = sorted(Path(sys.argv[1]).iterdir())
held_out = recognizer_finetune.finetune(raw_dirs)
breath.train(raw_dirs)
singing.train(raw_dirs)
evaluate.report(held_out)
