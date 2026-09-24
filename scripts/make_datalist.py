"""Build ``configs/datalist.json``: every ISLES'26 training case with a stratified 5-fold split.

Folds are stratified by lesion volume (quantile bins) at the subject level.

    python scripts/make_datalist.py --data_dir /path/to/ISLES26
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from monai import transforms
from monai.data import Dataset
from sklearn.model_selection import StratifiedKFold


DATASET_DIRNAME = "ATLAS3_Training_Raw"
IMAGE_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
LABEL_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"


def find_dataset_root(data_dir):
    """Return the directory containing the site folders."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    nested_root = os.path.join(data_dir, DATASET_DIRNAME)

    if os.path.isdir(nested_root):
        return nested_root
    if os.path.basename(data_dir) == DATASET_DIRNAME and os.path.isdir(data_dir):
        return data_dir
    raise FileNotFoundError(
        f"Could not find {DATASET_DIRNAME!r} in or at {data_dir!r}."
    )


def make_datalist(data_dir):
    """Collect all paired T1-weighted images and binary lesion masks."""
    dataset_root = find_dataset_root(data_dir)
    image_pattern = os.path.join(
        dataset_root, "*", "sub-*", "ses-1", "anat", f"*{IMAGE_SUFFIX}"
    )
    images = sorted(glob.glob(image_pattern))
    if not images:
        raise FileNotFoundError(f"No T1-weighted images matched {image_pattern!r}.")

    datalist = []
    for image_path in images:
        anat_dir = os.path.dirname(image_path)
        subject_dir = os.path.dirname(os.path.dirname(anat_dir))
        subject_id = os.path.basename(subject_dir)
        site = os.path.basename(os.path.dirname(subject_dir))
        label_path = image_path.removesuffix(IMAGE_SUFFIX) + LABEL_SUFFIX

        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f"Missing lesion mask for {subject_id}: {label_path}"
            )

        datalist.append(
            {
                "id": subject_id,
                "site": site,
                "image": os.path.relpath(image_path, data_dir),
                "label": os.path.relpath(label_path, data_dir),
            }
        )

    return datalist


def lesion_volume(mask):
    """Compute binary lesion volume in cubic millimetres."""
    lesion_num_voxels = torch.count_nonzero(mask > 0)
    voxel_size = np.prod(np.abs(np.asarray(mask.meta["pixdim"][1:4], dtype=float)))
    return float(lesion_num_voxels.item() * voxel_size)


def add_volume(datalist, data_dir):
    """Add physical lesion volume to each datalist entry."""
    transform = transforms.Compose(
        [
            transforms.CopyItemsd(keys="label", names="volume"),
            transforms.LoadImaged(keys="volume"),
            transforms.Lambdad(keys="volume", func=lesion_volume),
        ]
    )
    absolute_datalist = [
        {**sample, "label": os.path.join(data_dir, sample["label"])}
        for sample in datalist
    ]
    dataset = Dataset(absolute_datalist, transform=transform)

    volumes = []
    for sample in dataset:
        volumes.append(sample["volume"])

    return [
        {**sample, "volume": volume}
        for sample, volume in zip(datalist, volumes, strict=True)
    ]


def volume_strata(volumes, num_bins):
    """Quantize lesion volumes into approximately equally populated bins."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(volumes, quantiles))
    if len(edges) < 3:
        raise ValueError("Lesion volumes do not contain enough variation to stratify.")
    return np.digitize(volumes, edges[1:-1], right=True)


def stratified_kfold(datalist, num_bins=5, num_folds=5):
    """Assign stratified folds based on binary lesion volume."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    volumes = np.asarray([sample["volume"] for sample in datalist], dtype=float)
    strata = volume_strata(volumes, num_bins)
    stratum_counts = np.bincount(strata)
    if np.min(stratum_counts[stratum_counts > 0]) < num_folds:
        raise ValueError(
            "Each non-empty volume bin must contain at least num_folds samples."
        )

    splitter = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=42
    )
    folds = np.empty(len(datalist), dtype=int)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(datalist)), strata)
    ):
        folds[validation_indices] = fold

    result = []
    for sample, fold in zip(datalist, folds, strict=True):
        result.append(
            {
                key: value
                for key, value in {**sample, "fold": int(fold)}.items()
                if key != "volume"
            }
        )
    return sorted(result, key=lambda sample: sample["id"])


def export_datalist(training, file_path):
    """Write the dataset and fold assignments to JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as output_file:
        json.dump({"training": training, "test": []}, output_file, indent=2)
        output_file.write("\n")
    print(f"Saved {len(training)} training cases to {file_path}.")


def main(args):
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    datalist = make_datalist(data_dir)
    datalist = add_volume(datalist, data_dir)
    datalist = stratified_kfold(
        datalist, num_bins=args.num_bins, num_folds=args.num_folds
    )

    if args.save_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "configs",
            "datalist.json",
        )
    else:
        file_path = args.save_path
    export_datalist(datalist, os.path.abspath(os.path.expanduser(file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the ISLES'26 datalist with stratified cross-validation folds."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Path to the ISLES26 directory (or directly to its "
            f"{DATASET_DIRNAME} sub-directory)."
        ),
    )
    parser.add_argument(
        "--save_path", default=None, help="Output JSON path (default: configs/datalist.json)."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Number of lesion-volume strata.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    main(parser.parse_args())
