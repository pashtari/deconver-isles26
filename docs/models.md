# Models and configurations

[`configs/models.yaml`](../configs/models.yaml) defines one network per row of the
paper's tables. All networks follow the encoder-decoder template of Fig. 1: `[M_l]`
encoder blocks per level, one decoder block per level, initial width `C0`, resolution
halved and channels doubled at each level (Deconver caps the width at 512).

| name | family | [M_l] | C0 | k | Params (M) | Dice (%) |
| --- | --- | --- | --- | --- | --- | --- |
| `nnunet` | nnU-Net (residual conv, instance norm) ** | [1,1,1,1,1] | 32 | - | 22.75 | 62.54 * |
| `segresnet` | SegResNet (group norm, 8 groups) | [1,2,2,4] | 64 | - | 75.17 | 63.00 * |
| `swinunetr` | SwinUNETR-V2 | [2,2,2,2] | 48 | - | 72.76 | 62.41 |
| `deconver_m1224_c32_k3` | Deconver | [1,2,2,4] | 32 | 3 | 6.46 | 62.76 |
| `deconver_m11111_c32_k3` | Deconver | [1,1,1,1,1] | 32 | 3 | 10.55 | 62.99 |
| `deconver_m11111_c32_k5` | Deconver | [1,1,1,1,1] | 32 | 5 | 11.13 | 62.85 |
| `deconver_m11111_c64_k3` | Deconver | [1,1,1,1,1] | 64 | 3 | 24.23 | 63.40 |
| `deconver_m111111_c32_k3` | Deconver | [1,1,1,1,1,1] | 32 | 3 | 24.30 | 62.91 |
| `deconver_m1224_c64_k3` | **Deconver (headline)** | [1,2,2,4] | 64 | 3 | 25.46 | **63.59** |
| `deconver_m12244_c64_k3` | Deconver | [1,2,2,4,4] | 64 | 3 | 52.77 | 63.56 |

\* the paper averages a second training run (62.33 and 62.79) that is not distributed;
the distributed runs alone give 62.75 and 63.20, see [reproduce.md](reproduce.md).

\*\* the config matches the distributed checkpoints, which use residual blocks; the paper
describes the plain variant (22.57M parameters), see [reproduce.md](reproduce.md).

Deconver names encode the encoder blocks per level (`m`), the initial width (`c`) and the
NDC kernel size (`k`). Every Deconver uses instance normalization, depthwise NDC groups,
a source expansion ratio of 4 and one NDC iteration; the SegResNet + Deconver ensemble
of Table 1 combines `segresnet` and `deconver_m1224_c64_k3`.

## Selecting and modifying a model

- `--model <name>` picks an entry for training, evaluation and prediction.
- `--network_overrides "{'kernel_size': [5, 5, 5]}"` replaces constructor arguments of
  the selected entry without editing the file. Command-line values are Python literals:
  write `True`/`False`, not `true`/`false` (MONAI's CLI keeps the latter as strings).
- To add a configuration, add an entry to `configs/models.yaml`; its name becomes the
  folder under `logs/`.

## Training runs

Each trained model lives in `logs/<name>/fold<k>/`:

- `log.txt` - training log: loss, learning rate and validation metrics every 30 epochs
- `tb/` - TensorBoard events (`train_loss`, `val_mean_dice`, `val_mean_hd`, example slices)
- `checkpoint_epoch=300.pt` - final checkpoint holding `model`, `optimizer`, `lr_scheduler`,
  `trainer` and metric-logger state; `scripts.load_weights` extracts the weights. Three
  reconstructed checkpoints (SegResNet folds 2 and 4, SwinUNETR fold 0) carry no optimizer
  state, see [reproduce.md](reproduce.md)
- `evaluation/` - written by `configs/evaluate.yaml`

The Dice of a run is its `val_mean_dice` at epoch 300, the mean over the validation
cases of the fold; the paper averages the five folds.
