# Data

The experiments use the ISLES'26 training cohort (de la Rosa et al., 2026,
[doi:10.5281/zenodo.19856506](https://doi.org/10.5281/zenodo.19856506)): native-space
T1-weighted MRI with binary expert lesion masks, acquired at more than 60 centers and
released without resampling or spatial normalization. Download it from the challenge and
keep the released layout:

```
<data_dir>/ATLAS3_Training_Raw/<site>/<subject>/ses-1/anat/
    <subject>_ses-1_space-orig_desc-brain_T1w.nii.gz
    <subject>_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
```

Pass `<data_dir>` to every command with `--data_dir` (default: `~/ISLES26`).

## Datalist and folds

[`configs/datalist.json`](../configs/datalist.json) lists the 1451 training cases (55
site folders) with their fold assignment: five folds stratified by lesion volume at the
subject level, shared by every model and configuration. Paths are relative to
`<data_dir>`. The file was produced by

```bash
python scripts/make_datalist.py --data_dir /path/to/ISLES26
```

which pairs each T1w image with its mask, computes the lesion volume, bins the volumes
into five quantile strata and draws a stratified 5-fold split (seed 42). Re-running the
script overwrites `configs/datalist.json`; use `--save_path` to write elsewhere.

## Preprocessing

Preprocessing is defined in [`configs/train.yaml`](../configs/train.yaml) and runs on
the fly (Sect. 2.4 of the paper): crop to the foreground bounding box with a 10-voxel
margin, reorient to RAS, resample to 1 mm isotropic (linear for images, nearest for
masks), z-score the non-zero voxels and zero-pad to at least 128^3. Nothing is written
to disk.
