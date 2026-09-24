# Deconver for Stroke Lesion Segmentation

Code, configurations and cross-validation results for the paper *Deconver for Stroke
Lesion Segmentation* (SWITCH+ workshop, MICCAI 2026): a controlled comparison of
[Deconver](https://github.com/pashtari/deconver) with nnU-Net, SegResNet and
SwinUNETR-V2 on the ISLES'26 training cohort.

## Key idea

Deconver keeps the block layout of a vision transformer but replaces self-attention with
a *Deconv Mixer* built around a nonnegative deconvolution (NDC) layer. All four
architectures are trained under one fixed pipeline (same preprocessing, augmentation,
optimizer, folds and inference), so differences in Dice reflect the network block rather
than pipeline engineering.

## Results

Five-fold cross-validation on the ISLES'26 training cohort (1451 native-space T1-weighted
scans; mean Dice over the validation cases of all folds, Table 1 of the paper):

| Model | [M_l] | C0 | Params (M) | Dice (%) |
| --- | --- | --- | --- | --- |
| nnU-Net | [1,1,1,1,1] | 32 | 22.57 | 62.54 |
| SegResNet | [1,2,2,4] | 64 | 75.17 | 63.00 |
| SwinUNETR-V2 | [2,2,2,2] | 48 | 72.76 | 62.41 |
| **Deconver** | [1,2,2,4] | 64 | 25.46 | **63.59** |
| Ensemble (SegResNet + Deconver) | | | 100.63 | **64.44** |

The paper's nnU-Net and SegResNet rows average a second training run (62.33 and 62.79)
that is not distributed; [results/results_average.csv](results/results_average.csv) gives
the distributed runs alone (62.75 and 63.20). The distributed nnU-Net uses residual blocks
(22.75M parameters), see [docs/reproduce.md](docs/reproduce.md).

Deconver reaches the best single-model Dice with about a third of the parameters of the
two large baselines, and a 10.55M-parameter variant already matches SegResNet. The
Deconver ablation over depth, width and NDC kernel size is in
[docs/models.md](docs/models.md); the five-fold averages of every configuration are in
[results/results_average.csv](results/results_average.csv) and the per-fold values in
[results/results_folds.csv](results/results_folds.csv).

## Installation

```bash
conda create -n isles26 python=3.12 -y && conda activate isles26
pip install -r requirements.txt
```

See [docs/installation.md](docs/installation.md) for CUDA notes and tested versions.

## Quick start

Run every command from the repository root; `--data_dir` points to the ISLES'26 training
data (layout in [docs/data.md](docs/data.md)).

```bash
# train the headline Deconver on fold 0 -> logs/deconver_m1224_c64_k3/fold0/
python -m monai.bundle run --config_file configs/train.yaml \
    --model deconver_m1224_c64_k3 --fold 0 --data_dir /path/to/ISLES26

# evaluate it on its validation fold (per-case Dice and HD95)
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']" \
    --model deconver_m1224_c64_k3 --fold 0 --data_dir /path/to/ISLES26

# segment new T1-weighted images with the five Deconver folds
python -m monai.bundle run --config_file "['configs/train.yaml','configs/predict.yaml']" \
    --input_dir /path/to/images --pred_dir predictions
```

## Reproducing the paper

Every row of Tables 1 and 2 is one entry of [configs/models.yaml](configs/models.yaml),
trained on all five folds:

```bash
DATA_DIR=/path/to/ISLES26 bash hpc/submit_grid.sh    # 10 models x 5 folds on a PBS cluster
python scripts/benchmark.py                           # parameters and GPU throughput -> results/benchmark.csv
python scripts/collect_results.py                     # -> results/*.csv
python -m monai.bundle run --config_file "['configs/train.yaml','configs/ensemble.yaml']" \
    --fold 0 --data_dir /path/to/ISLES26              # SegResNet + Deconver ensemble, one fold
```

[docs/reproduce.md](docs/reproduce.md) walks through the procedure, maps every run to
the paper's tables and lists the known gaps in the released checkpoints.

## Repository structure

```
configs/      MONAI bundle configs: models.yaml (every network), train / evaluate /
              ensemble / predict pipelines, datalist.json (cases and 5-fold split)
scripts/      Python package used by the configs, plus make_datalist.py and collect_results.py
hpc/          PBS job scripts for the full training grid
logs/         one folder per model and fold: log.txt, tb/, checkpoint_epoch=300.pt (not tracked by git)
results/      five-fold averages (sorted by Dice) and per-fold Dice and HD95 of every configuration
docs/         installation, data, models, reproduction
manuscript/   LaTeX source and PDF of the paper
poster/       A0 poster for SWITCH+ 2026 (PDF, HTML source and build scripts)
```

## Citation

```bibtex
@inproceedings{vyncke2026deconver,
  title     = {Deconver for Stroke Lesion Segmentation},
  author    = {Vyncke, Niels and Noei, Shahryar and Jurman, Giuseppe and Pi{\v{z}}urica, Aleksandra and Ashtari, Pooya},
  booktitle = {SWITCH+ Workshop, MICCAI 2026},
  year      = {2026}
}

@article{ashtari2025deconver,
  title   = {Deconver: A Deconvolutional Network for Medical Image Segmentation},
  author  = {Ashtari, Pooya and Noei, Shahryar and Haredasht, Fateme Nateghi and Chen, Jonathan H and Jurman, Giuseppe and Pi{\v{z}}urica, Aleksandra and Van Huffel, Sabine},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025}
}
```

## License and acknowledgments

Released under the Apache 2.0 license ([LICENSE](LICENSE)). Built on
[MONAI](https://monai.io) and the [deconver](https://github.com/pashtari/deconver)
package. Data: ISLES'26 training cohort (de la Rosa et al., 2026,
[doi:10.5281/zenodo.19856506](https://doi.org/10.5281/zenodo.19856506)).
