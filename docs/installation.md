# Installation

```bash
conda create -n isles26 python=3.12 -y
conda activate isles26
pip install -r requirements.txt
```

`requirements.txt` installs MONAI 1.5.2, PyTorch Ignite, the
[`deconver`](https://github.com/pashtari/deconver) package (pinned to the commit used for
the paper) and the remaining runtime dependencies. PyTorch comes from PyPI; if that wheel
does not match your CUDA driver, install PyTorch first from its own index, for example

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Check the installation from the repository root:

```bash
python -c "import monai, deconver, scripts; print(monai.__version__)"
```

## Tested versions

| | Python | torch | MONAI | pytorch-ignite |
| --- | --- | --- | --- | --- |
| This release (verified) | 3.12 | 2.10.0 (CUDA 12.6) | 1.5.2 | 0.5.3 |
| Paper's training runs | 3.10 / 3.11 | 2.4 | 1.4.0 | 0.5.1 |

The released checkpoints load unchanged in both stacks.

## Hardware

Every run trained on one GPU with 128^3 patches and a batch of two (nnU-Net about 2.5
min, SegResNet and Deconver about 6.5 min, SwinUNETR-V2 about 13 min per epoch on a
V100-class GPU; 300 epochs). `CacheDataset` keeps the preprocessed training set in RAM,
which needs roughly 50 GB at `--cache_rate 1.0`; lower the rate on smaller machines.

Evaluation and prediction use sliding-window inference with 128^3 windows, four windows
per forward pass. One window of the headline Deconver takes about 8 GB of GPU memory, so
on GPUs with 16 GB or less add `--val_inferer#sw_batch_size 1` (results do not change).
