# Reproducing the paper

All commands run from the repository root with the environment of
[installation.md](installation.md) and the data layout of [data.md](data.md).
Command-line values are Python literals (`--amp True`, `--roi_size "[96,96,96]"`).

## 1. Train

One model on one fold (300 epochs):

```bash
python -m monai.bundle run --config_file configs/train.yaml \
    --model deconver_m1224_c64_k3 --fold 0 --data_dir /path/to/ISLES26
```

Outputs go to `logs/deconver_m1224_c64_k3/fold0/`. Resume an interrupted run with
`--ckpt_path logs/<model>/fold<k>/checkpoint_epoch=<n>.pt`.

The full grid, 10 models x 5 folds, on a PBS cluster:

```bash
DATA_DIR=/path/to/ISLES26 VENV=/path/to/venv bash hpc/submit_grid.sh          # every model
DATA_DIR=/path/to/ISLES26 VENV=/path/to/venv bash hpc/submit_grid.sh nnunet   # one model
```

[`hpc/train.pbs`](../hpc/train.pbs) holds the resource request (1 GPU, 4 cores, 96 GB
RAM, 72 h); adapt it to your scheduler. Each job runs the command above with
`--num_workers 4`, and PBS output goes to `logs/pbs/`.

## 2. Collect the cross-validation results

```bash
python scripts/collect_results.py
```

reads the last validation metrics of every run from `logs/*/fold*/tb`, averages them
over the folds and writes [`results/results_average.csv`](../results/results_average.csv)
(one row per model: number of folds, mean Dice (%) and mean HD95 (mm), sorted by Dice)
and [`results/results_folds.csv`](../results/results_folds.csv) (one row per model
and fold). Both also carry the parameter count and the GPU inference throughput of each
model, written by

```bash
python scripts/benchmark.py
```

which measures 128^3 input patches segmented per second (forward pass only, batch size 1,
fp32) on the first visible GPU; `results_average.csv` also records the input size, batch,
precision, GPU and PyTorch version of the measurement.

## 3. Evaluate a trained model

```bash
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']" \
    --model deconver_m1224_c64_k3 --fold 0 --data_dir /path/to/ISLES26
```

loads `logs/<model>/fold<k>/checkpoint_epoch=*.pt`, runs sliding-window inference
(128^3 windows, 50% overlap, Gaussian blending) on the fold's validation cases and writes
`logs/<model>/fold<k>/evaluation/`: `metrics.csv` (summary statistics of Dice and HD95)
and `val_mean_dice_raw.csv`, `val_mean_hd_raw.csv` (per case). Metrics are computed in
the preprocessed 1 mm space, exactly as during training. On GPUs with 16 GB or less add
`--val_inferer#sw_batch_size 1` (also for the ensemble and prediction commands below).

## 4. Ensemble (Table 1, last row)

```bash
for fold in 0 1 2 3 4; do
    python -m monai.bundle run --config_file "['configs/train.yaml','configs/ensemble.yaml']" \
        --fold $fold --data_dir /path/to/ISLES26
done
```

averages the sigmoid maps of `segresnet` and `deconver_m1224_c64_k3` (fold-matched
checkpoints) before thresholding at 0.5 and writes
`logs/ensemble/segresnet+deconver_m1224_c64_k3/fold<k>/evaluation/`. The paper's 64.44
is the mean over the validation cases of all folds. Other members:
`--members "['nnunet','deconver_m1224_c64_k3']"`.

## 5. Segment new images

```bash
python -m monai.bundle run --config_file "['configs/train.yaml','configs/predict.yaml']" \
    --input_dir /path/to/images --pred_dir predictions
```

segments every `*_T1w.nii.gz` under `--input_dir` (`--input_pattern` changes the glob)
with the five Deconver folds and writes `<name>_lesion.nii.gz` masks in the original
image space. The challenge submission used the ten-network ensemble:
`--models "['segresnet','deconver_m1224_c64_k3']" --folds "[0,1,2,3,4]"`.

## Mapping runs to the paper

| Paper row | `logs/` folder | Dice (%) in `results/results_average.csv` |
| --- | --- | --- |
| Table 1, nnU-Net | `nnunet` | 62.75; the paper averages it with a second run: (62.75 + 62.33) / 2 = 62.54 |
| Table 1, SegResNet | `segresnet` | 63.20; the paper averages it with a second run: (63.20 + 62.79) / 2 = 63.00 |
| Table 1, SwinUNETR-V2 | `swinunetr` | 62.41 over 5 folds (62.26 over the 4 folds with logs) |
| Table 1 and 2, Deconver 25.46M | `deconver_m1224_c64_k3` | 63.59 |
| Table 2, other rows | `deconver_m1224_c32_k3` ... `deconver_m12244_c64_k3` | 62.77 *, 62.99, 62.85, 63.40, 62.91, 63.56 |

\* printed as 62.76 in the paper (the exact mean is 62.766).

## Known gaps in the released runs

- **SwinUNETR fold 4**: log, TensorBoard events and checkpoint were lost in transfer.
  The paper's 62.41 includes this fold; the four remaining folds average 62.26.
- **Reconstructed checkpoints**: the files of SegResNet folds 2 and 4 and SwinUNETR
  fold 0 arrived corrupt (their tail overwritten by foreign data). The model weights lie
  in the intact part and were recovered, so these three checkpoints hold the `model`,
  `trainer` and `lr_scheduler` state but no optimizer state: they evaluate and predict
  normally but cannot resume training. SwinUNETR fold 2 lost 33 of its model tensors and
  could not be recovered; its log and TensorBoard curves are intact.
- **nnU-Net checkpoints use residual blocks.** The grid script that trained
  `logs/nnunet` passed `res_block false` on the command line, which MONAI's CLI keeps as
  the string `'false'` (truthy), so these five checkpoints are residual DynUNets with
  22.75M parameters. `configs/models.yaml` defines this residual variant so that the
  checkpoints load as they are. The paper describes the plain variant (22.57M), which
  only the undistributed second run (Dice 62.33) trained; to train it, pass
  `--network_overrides "{'res_block': False}"`.
- **Second runs of nnU-Net and SegResNet.** Table 1 of the paper averages a second
  training run of each, trained on another machine (five-fold mean Dice 62.33 and 62.79).
  For SegResNet it repeated the same configuration; for nnU-Net it is the only run with
  plain blocks (see above). Neither run is distributed, so `results/results_average.csv`
  reports the distributed runs alone.
- The five `deconver_m11111_c32_k5` runs were restarted from checkpoints of an
  interrupted first attempt with the same configuration (epochs 167, 167, 116, 87 and
  59) and trained on to epoch 300 like every other run.
