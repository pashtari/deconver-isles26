"""Measure the size and inference throughput of every model in ``configs/models.yaml``.

Throughput is the number of 128^3 input patches segmented per second (forward pass only,
batch size 1, fp32, no gradient) after a warm-up, on the first visible GPU. The parameter
count, throughput and measurement conditions are written into ``results/results_average.csv``
(and parameters and throughput into ``results/results_folds.csv``), matched by model name.

    python scripts/benchmark.py [--iters 20] [--roi 128]
"""

import argparse
import csv
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts  # noqa: E402


BENCH_COLUMNS = ["params_M", "throughput_img_s", "input", "batch", "precision", "gpu", "torch"]


def merge_into(path, rows, columns):
    """Add or refresh ``columns`` of every model row of the CSV at ``path`` (created if missing)."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f); fields, table = list(reader.fieldnames), list(reader)
    else:
        fields, table = ["model"], [{"model": m} for m in rows]
    fields += [c for c in columns if c not in fields]
    for row in table:
        row.update({c: rows[row["model"]][c] for c in columns} if row["model"] in rows else {})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    print(f"Updated {path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20, help="timed forward passes per model")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--roi", type=int, default=128, help="edge length of the cubic input patch")
    parser.add_argument("--results_dir", default=os.path.join(root, "results"))
    args = parser.parse_args()

    models_file = os.path.join(root, "configs", "models.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = True
    x = torch.randn(1, 1, args.roi, args.roi, args.roi, device=device)

    rows = {}
    for model in yaml.safe_load(open(models_file))["models"]:
        net = scripts.build_network(models_file, model).to(device).eval()
        params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            for _ in range(args.warmup):
                net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        throughput = args.iters / (time.perf_counter() - start)
        rows[model] = {"params_M": f"{params / 1e6:.2f}", "throughput_img_s": f"{throughput:.2f}", "input": f"{args.roi}^3",
                       "batch": 1, "precision": "fp32", "gpu": gpu, "torch": torch.__version__}
        print(f"{model:26s} {params / 1e6:7.2f}M  {throughput:7.2f} img/s", flush=True)
        del net
        torch.cuda.empty_cache()

    os.makedirs(args.results_dir, exist_ok=True)
    merge_into(os.path.join(args.results_dir, "results_average.csv"), rows, BENCH_COLUMNS)
    merge_into(os.path.join(args.results_dir, "results_folds.csv"), rows, BENCH_COLUMNS[:2])


if __name__ == "__main__":
    main()
