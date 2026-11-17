"""Paths shared by the bundled benchmark protocols."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def env_path(name, default):
    return str(Path(os.environ.get(name, default)).expanduser().resolve())


DATASETS_ROOT = Path(env_path('DATASETS_ROOT', ROOT / 'data'))
CHECKPOINTS_ROOT = Path(env_path('CHECKPOINTS_ROOT', ROOT / 'checkpoints'))
CDFSOD_PATH = env_path('CDFSOD_PATH', DATASETS_ROOT / 'cdfsod')
ODINW_PATH = env_path('ODINW_PATH', DATASETS_ROOT / 'odinw_13')
RF100_VL_FSOD_PATH = env_path('RF100_VL_FSOD_PATH', DATASETS_ROOT / 'rf100-vl-fsod')
MMGDINOB_PATH = env_path(
    'MMGDINOB_PATH', CHECKPOINTS_ROOT / 'grounding_dino_swin-b_pretrain_all-f9818a7c.pth')
MMGDINOL_PATH = env_path(
    'MMGDINOL_PATH', CHECKPOINTS_ROOT / 'grounding_dino_swin-l_pretrain_all-56d69e78.pth')
