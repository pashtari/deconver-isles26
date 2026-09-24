"""Helpers referenced from the bundle configs."""

import glob
import logging
import os
import re

import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint
from monai.bundle import ConfigParser


def build_network(models_file, model, overrides=None):
    """Instantiate one entry of ``configs/models.yaml``.

    Only the requested definition is parsed, so the other networks are never built.
    ``overrides`` maps constructor arguments to replacement values, e.g.
    ``{"res_block": True}``.
    """
    parser = ConfigParser(ConfigParser.load_config_file(models_file))
    if model not in parser["models"]:
        raise KeyError(f"unknown model {model!r}; available: {sorted(parser['models'])}")
    for key, value in (overrides or {}).items():
        parser[f"models#{model}#{key}"] = value
    return parser.get_parsed_content(f"models#{model}")


def find_checkpoint(run_dir):
    """Return the ``checkpoint_epoch=*.pt`` file with the highest epoch in ``run_dir``."""
    paths = glob.glob(os.path.join(run_dir, "checkpoint_epoch=*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint_epoch=*.pt in {run_dir}")
    return max(paths, key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)))


def load_weights(network, path):
    """Load the model weights of a training checkpoint (or a bare state dict) into ``network``."""
    # Checkpoints written by this repository hold trainer/optimizer state as well,
    # so they are loaded with ``weights_only=False``.
    state = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(state.get("model", state))
    return network


class CheckpointLoader:
    """Restore the objects in ``load_dict`` from a checkpoint when the engine starts.

    Behaves like ``monai.handlers.CheckpointLoader`` but reads the file with
    ``weights_only=False``, which the checkpoints written by this repository require.
    """

    def __init__(self, load_path, load_dict, strict=True):
        self.load_path = load_path
        self.load_dict = load_dict
        self.strict = strict
        self.logger = logging.getLogger(__name__)

    def attach(self, engine):
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine):
        checkpoint = torch.load(self.load_path, map_location="cpu", weights_only=False)
        if len(self.load_dict) == 1 and next(iter(self.load_dict)) not in checkpoint:
            checkpoint = {next(iter(self.load_dict)): checkpoint}  # bare state dict
        prior_max_epochs = engine.state.max_epochs
        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint, strict=self.strict)
        if prior_max_epochs is not None and engine.state.epoch > prior_max_epochs:
            raise ValueError(
                f"checkpoint epoch {engine.state.epoch} exceeds max_epochs {prior_max_epochs}"
            )
        engine.state.max_epochs = prior_max_epochs
        self.logger.info(f"Restored {list(self.load_dict)} from {self.load_path}")


def reset_cpu_affinity():
    """Undo CPU pinning done by some HPC launchers, which would starve DataLoader workers."""
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass


class LogModelInfoHandler:
    """Log the parameter count of a model when the engine starts."""

    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger(__name__)

    def attach(self, engine):
        engine.add_event_handler(Events.STARTED, self.log_model_info)

    def log_model_info(self, engine):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.model._get_name()}")
        self.logger.info(f"Total params: {total:,d} ({total / 1e6:.2f}M)")
        self.logger.info(f"Trainable params: {trainable:,d}")
