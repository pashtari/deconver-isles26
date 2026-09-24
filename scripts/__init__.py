"""Python package used by the MONAI bundle configs (referenced there as ``scripts.<name>``)."""

import os

# Let the CUDA caching allocator grow segments instead of fragmenting: sliding-window
# inference with 128^3 windows otherwise fails on 16 GB GPUs. Must be set before the
# first CUDA allocation; both names cover old and new torch versions.
for _var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    os.environ.setdefault(_var, "expandable_segments:True")

from .metrics import MeanDice, MeanHausdorffDistance
from .utils import (
    CheckpointLoader,
    LogModelInfoHandler,
    build_network,
    find_checkpoint,
    load_weights,
    reset_cpu_affinity,
)

reset_cpu_affinity()
