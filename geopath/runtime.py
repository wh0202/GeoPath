"""Local runner support for the benchmark deterministic warning policy."""

import os

import torch
from mmengine.runner import Runner


class GeoPathRunner(Runner):
    def set_randomness(self, seed=None, diff_rank_seed=False, deterministic=False,
                       warn_only=True):
        if deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not deterministic or not warn_only:
            return super().set_randomness(
                seed=seed, diff_rank_seed=diff_rank_seed, deterministic=deterministic)
        super().set_randomness(
            seed=seed, diff_rank_seed=diff_rank_seed, deterministic=False)
        self._deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
