"""Collect the cross-validation results of every model under ``logs/``.

The last validation metrics of each fold are read from the TensorBoard logs and written
to ``results/results_folds.csv``; their averages over the folds, sorted by Dice, go to
``results/results_average.csv``.

    python scripts/collect_results.py
"""

import argparse
import csv
import glob
import os
from statistics import mean

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

TAGS = {"val_mean_dice": "dice", "val_mean_hd": "hd95"}


def last_scalars(tb_dir):
    """Return the last logged value of each tag in ``TAGS``."""
    values = {}
    for event_file in sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))):
        with open(event_file, "rb") as f:
            if not any(tag.encode() in f.read() for tag in TAGS):
                continue
        for event in EventFileLoader(event_file).Load():
            for value in event.summary.value:
                if value.tag in TAGS:
                    values[TAGS[value.tag]] = value.tensor.float_val[0] if value.tensor.float_val else value.simple_value
    return values


def collect(logs_dir):
    """Per-fold metrics of every model found under ``logs_dir``: {model: {fold: values}}."""
    results = {}
    for fold_dir in sorted(glob.glob(os.path.join(logs_dir, "*", "fold[0-9]"))):
        values = last_scalars(os.path.join(fold_dir, "tb"))
        if "dice" in values:
            results.setdefault(os.path.basename(os.path.dirname(fold_dir)), {})[int(fold_dir[-1])] = values
    return results


def average(results):
    """One row per model, averaged over its folds and sorted by Dice."""
    rows = []
    for model, folds in results.items():
        hd95 = [v["hd95"] for v in folds.values() if "hd95" in v]
        rows.append(
            {
                "model": model,
                "folds": len(folds),
                "dice": 100 * mean(v["dice"] for v in folds.values()),
                "hd95": mean(hd95) if hd95 else None,
            }
        )
    return sorted(rows, key=lambda r: r["dice"], reverse=True)


def per_fold(results, order):
    """One row per model and fold, models in the given order."""
    return [
        {"model": model, "fold": fold, "dice": 100 * v["dice"], "hd95": v.get("hd95")}
        for model in order
        for fold, v in sorted(results[model].items())
    ]


def benchmark(results_dir):
    """{model: {params_M, throughput_img_s}} from results/benchmark.csv, or {}."""
    path = os.path.join(results_dir, "benchmark.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["model"]: {"params_M": r["params_M"], "throughput_img_s": r["throughput_img_s"]} for r in csv.DictReader(f)}


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else f"{row[k]:.2f}" if isinstance(row[k], float) else row[k] for k in fields})


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs_dir", default=os.path.join(root, "logs"))
    parser.add_argument("--results_dir", default=os.path.join(root, "results"))
    args = parser.parse_args()

    results = collect(args.logs_dir)
    averages = average(results)
    bench = benchmark(args.results_dir)
    extra = ["params_M", "throughput_img_s"] if bench else []
    for row in averages:
        row.update(bench.get(row["model"], {k: "" for k in extra}))
    folds = per_fold(results, [r["model"] for r in averages])
    for row in folds:
        row.update(bench.get(row["model"], {k: "" for k in extra}))
    os.makedirs(args.results_dir, exist_ok=True)
    write_csv(os.path.join(args.results_dir, "results_average.csv"), averages, ["model", "folds"] + extra + ["dice", "hd95"])
    write_csv(os.path.join(args.results_dir, "results_folds.csv"), folds, ["model", "fold"] + extra + ["dice", "hd95"])

    print("| model | folds |" + "".join(f" {k} |" for k in extra) + " dice | hd95 |")
    print("| --- | --- |" + " --- |" * len(extra) + " --- | --- |")
    for r in averages:
        print(f"| {r['model']} | {r['folds']} |" + "".join(f" {r[k]} |" for k in extra) + f" {r['dice']:.2f} | {'' if r['hd95'] is None else f'{r['hd95']:.2f}'} |")
    print(f"\nWrote results_average.csv and results_folds.csv to {args.results_dir}")


if __name__ == "__main__":
    main()
