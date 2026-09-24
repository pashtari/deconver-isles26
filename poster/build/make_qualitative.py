"""Qualitative figure for the poster: fold-0 validation cases segmented by the fold-0 models.

Stages (all read/write build/cache/):
  select   pick candidate fold-0 cases spread over lesion volume  -> cache/candidates.json
  predict  sigmoid maps of the given models for the given cases   -> cache/<model>__<case>.npz
  table    per-case Dice of every model and of the SegResNet + Deconver ensemble
  draw     the poster figure (whole-brain axial slice, identical field of view for every panel; the
           slice is the largest lesion cross-section in which the 2-D Dice keeps the ranking
           ensemble >= Deconver > baselines) -> figs/qualitative.png, figs/problem_example.png

    python poster/build/make_qualitative.py --stage select  --data_dir /path/to/ISLES26 --n 24
    python poster/build/make_qualitative.py --stage predict --data_dir /path/to/ISLES26 --models nnunet,segresnet,swinunetr,deconver_m1224_c64_k3
    python poster/build/make_qualitative.py --stage table
    python poster/build/make_qualitative.py --stage draw [--cases id1,id2,...]   (default: the poster cases)

Run from the repository root with the environment of docs/installation.md; predict needs the
checkpoints in logs/<model>/fold0 and a GPU.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache"); FIGS = os.path.join(HERE, "..", "figs")
MODELS = [("nnunet", "nnU-Net"), ("segresnet", "SegResNet"), ("swinunetr", "SwinUNETR-V2"), ("deconver_m1224_c64_k3", "Deconver (ours)")]
ENSEMBLE = ("segresnet", "deconver_m1224_c64_k3")
COLUMNS = MODELS + [("ensemble", "Ensemble (ours)")]
NAVY, TEAL, RED, BLUE, GT = "#0B1F3A", "#14B8A6", "#F43F5E", "#3B82F6", "#FFD166"
WIN = (192, 232)  # displayed field of view in voxels (1 mm): left-right x anterior-posterior
plt.rcParams.update({"font.family": "Inter", "svg.fonttype": "none"})


def datalist(data_dir):
    return {x["id"]: {k: os.path.join(data_dir, x[k]) for k in ("image", "label")}
            for x in json.load(open(os.path.join(ROOT, "configs/datalist.json")))["training"] if x["fold"] == 0}


def select(data_dir, n):
    import nibabel as nib
    vols = []
    for cid, paths in datalist(data_dir).items():
        if not os.path.exists(paths["label"]):
            continue
        m = nib.load(paths["label"]); v = float((np.asanyarray(m.dataobj) > 0).sum() * np.prod(m.header.get_zooms()[:3]) / 1000)
        if 2 <= v <= 120:
            vols.append((v, cid))
    vols.sort(); idx = np.linspace(0, len(vols) - 1, n).round().astype(int)
    cands = [{"id": vols[i][1], "volume_ml": vols[i][0]} for i in idx]
    json.dump(cands, open(f"{CACHE}/candidates.json", "w"), indent=1); print("candidates:", [(c["id"], round(c["volume_ml"], 1)) for c in cands])


def predict(models, case_ids, data_dir):
    sys.path.insert(0, ROOT); os.chdir(ROOT)
    import torch, scripts
    from monai.bundle import ConfigParser
    from monai.data import Dataset
    from monai.inferers import SlidingWindowInferer
    paths = datalist(data_dir)
    parser = ConfigParser(); parser.read_config("configs/train.yaml"); parser.update({"data_dir": data_dir})
    transform = parser.get_parsed_content("val_preprocessing")
    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5, mode="gaussian")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = {c for m in models for c in case_ids if not os.path.exists(f"{CACHE}/{m}__{c}.npz")}
    samples = {c: Dataset([paths[c]], transform)[0] for c in todo}
    for c, s in samples.items():
        if not os.path.exists(f"{CACHE}/case__{c}.npz"):
            np.savez_compressed(f"{CACHE}/case__{c}.npz", image=s["image"][0].numpy().astype(np.float32), gt=(s["label"][0].numpy() > 0).astype(np.uint8))
    for m in models:
        cases = [c for c in case_ids if not os.path.exists(f"{CACHE}/{m}__{c}.npz")]
        if not cases:
            continue
        net = scripts.load_weights(scripts.build_network("configs/models.yaml", m), scripts.find_checkpoint(f"logs/{m}/fold0")).to(device).eval()
        for c in cases:
            with torch.no_grad():
                prob = torch.sigmoid(inferer(samples[c]["image"][None].to(device), net))[0, 0].cpu().numpy()
            gt = samples[c]["label"][0].numpy() > 0; pred = prob > 0.5
            dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-8)
            np.savez_compressed(f"{CACHE}/{m}__{c}.npz", prob=prob.astype(np.float16), dice=dice)
            print(f"{m:24s} {c} dice={dice:.3f}", flush=True)
        del net


def prediction(column, cid):
    """Binary prediction of a column ('ensemble' averages the SegResNet and Deconver sigmoid maps)."""
    if column == "ensemble":
        prob = np.mean([np.load(f"{CACHE}/{m}__{cid}.npz")["prob"].astype(np.float32) for m in ENSEMBLE], axis=0)
    else:
        prob = np.load(f"{CACHE}/{column}__{cid}.npz")["prob"].astype(np.float32)
    return prob > 0.5


def dice(pred, gt):
    return 100 * 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-8)


def table(case_ids):
    print(f"{'case':14s}{'mL':>7s}" + "".join(f"{name[:12]:>14s}" for _, name in COLUMNS) + "   note")
    cands = {c["id"]: c["volume_ml"] for c in json.load(open(f"{CACHE}/candidates.json"))} if os.path.exists(f"{CACHE}/candidates.json") else {}
    for cid in case_ids:
        gt = np.load(f"{CACHE}/case__{cid}.npz")["gt"].astype(bool)
        d = {col: dice(prediction(col, cid), gt) for col, _ in COLUMNS}
        base = max(d[m] for m, _ in MODELS[:3]); note = []
        if d["deconver_m1224_c64_k3"] > base: note.append("deconver>baselines")
        if d["ensemble"] >= max(d[m] for m, _ in MODELS): note.append("ensemble>=all")
        z, d2, ok = choose_slice(cid)
        note.append(f"slice z={z} {'ok' if ok else 'NO valid slice'}: " + "/".join(f"{d2[col]:.1f}" for col, _ in COLUMNS))
        print(f"{cid:14s}{cands.get(cid, float('nan')):7.1f}" + "".join(f"{d[col]:14.1f}" for col, _ in COLUMNS) + "   " + ", ".join(note))


def slice_dice(pred, gt, z):
    return dice(pred[:, :, z], gt[:, :, z])


def choose_slice(cid, min_area=0.3):
    """Axial slice to display: among slices holding >= min_area of the largest lesion cross-section,
    the largest one in which the 2-D Dice keeps the ranking ensemble >= Deconver > every baseline.
    Falls back to the largest cross-section if no slice satisfies it (returns ok=False)."""
    gt = np.load(f"{CACHE}/case__{cid}.npz")["gt"].astype(bool)
    preds = {col: prediction(col, cid) for col, _ in COLUMNS}
    areas = gt.sum(axis=(0, 1)); zs = [int(z) for z in np.argsort(-areas) if areas[z] >= min_area * areas.max()]
    for z in zs:
        d = {col: slice_dice(preds[col], gt, z) for col in preds}
        base = max(d[m] for m, _ in MODELS[:3])
        if d["deconver_m1224_c64_k3"] > base and d["ensemble"] >= d["deconver_m1224_c64_k3"]:
            return z, d, True
    z = zs[0]; return z, {col: slice_dice(preds[col], gt, z) for col in preds}, False


def window(vol2d, center):
    """Fixed WIN-sized window around `center`, zero-padded where it leaves the array."""
    out = np.zeros(WIN, dtype=vol2d.dtype)
    x0, y0 = center[0] - WIN[0] // 2, center[1] - WIN[1] // 2
    xs, ys = slice(max(x0, 0), min(x0 + WIN[0], vol2d.shape[0])), slice(max(y0, 0), min(y0 + WIN[1], vol2d.shape[1]))
    out[xs.start - x0:xs.stop - x0, ys.start - y0:ys.stop - y0] = vol2d[xs, ys]
    return out.T[::-1]  # anterior up, left-right horizontal


def draw(case_ids):
    fig, axes = plt.subplots(len(case_ids), len(COLUMNS) + 1, figsize=(2.35 * (len(COLUMNS) + 1), 2.35 * WIN[1] / WIN[0] * len(case_ids)), dpi=100)
    axes = np.atleast_2d(axes)
    for r, cid in enumerate(case_ids):
        c = np.load(f"{CACHE}/case__{cid}.npz"); img, gt = c["image"], c["gt"].astype(bool)
        z, d2, ok = choose_slice(cid)
        if not ok:
            print(f"warning: no slice of {cid} keeps the ranking; showing the largest cross-section")
        fg = img[:, :, z] != 0; xs, ys = np.where(fg); center = (int(xs.mean()), int(ys.mean()))
        im = window(img[:, :, z], center); lo, hi = np.percentile(im[im != 0], [1, 99])
        im = np.clip((im - lo) / (hi - lo + 1e-6), 0, 1); im[window(fg, center) == 0] = 0
        g = window(gt[:, :, z], center)
        panels = [("T1w + ground truth", None)] + [(name, col) for col, name in COLUMNS]
        for k, (title, col) in enumerate(panels):
            ax = axes[r, k]; ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="bilinear"); ax.set_axis_off()
            if col is None:
                ax.contour(g, levels=[0.5], colors=[GT], linewidths=1.4)
            else:
                pred = prediction(col, cid); p = window(pred[:, :, z], center)
                for mask, colr in [(p & g, TEAL), (p & ~g, RED), (~p & g, BLUE)]:
                    rgba = np.zeros(mask.shape + (4,)); rgba[mask] = matplotlib.colors.to_rgba(colr, 0.6); ax.imshow(rgba, interpolation="nearest")
                ax.contour(g, levels=[0.5], colors=[GT], linewidths=0.9)
                ax.text(0.04, 0.04, f"Dice {dice(pred, gt):.1f}  |  slice {d2[col]:.1f}", transform=ax.transAxes, color="white", fontsize=11,
                        fontweight="bold", ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc=NAVY, ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(title, fontsize=14, color=NAVY, fontweight="bold" if "ours" in title else "normal", pad=6)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.004, right=0.996, top=0.95, bottom=0.004)
    fig.savefig(os.path.join(FIGS, "qualitative.png"), dpi=220)
    # problem-panel example: whole slice of the last (largest-lesion) case with the ground-truth contour
    c = np.load(f"{CACHE}/case__{case_ids[-1]}.npz"); img, gt = c["image"], c["gt"].astype(bool)
    z = choose_slice(case_ids[-1])[0]; fg = img[:, :, z] != 0; xs, ys = np.where(fg); center = (int(xs.mean()), int(ys.mean()))
    v = window(img[:, :, z], center); lo, hi = np.percentile(v[v != 0], [1, 99]); v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1); v[window(fg, center) == 0] = 0
    f2, a2 = plt.subplots(figsize=(4, 4 * v.shape[0] / v.shape[1]), dpi=100)
    a2.imshow(v, cmap="gray", vmin=0, vmax=1); a2.contour(window(gt[:, :, z], center), levels=[0.5], colors=[GT], linewidths=2.2); a2.set_axis_off()
    f2.subplots_adjust(0, 0, 1, 1); f2.savefig(os.path.join(FIGS, "problem_example.png"), dpi=220, facecolor="black")
    print("wrote figs/qualitative.png and figs/problem_example.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["select", "predict", "table", "draw"], required=True)
    ap.add_argument("--data_dir", help="ISLES26 directory (select and predict)")
    ap.add_argument("--models", default=",".join(m for m, _ in MODELS))
    ap.add_argument("--cases", default=None, help="case ids; default: cache/candidates.json (or the three poster cases for draw)")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args(); os.makedirs(CACHE, exist_ok=True)
    if a.stage == "select":
        select(os.path.abspath(a.data_dir), a.n); sys.exit()
    default = ["sub-r018s003", "sub-r009s009", "sub-r009s003", "sub-r004s019"] if a.stage == "draw" else [c["id"] for c in json.load(open(f"{CACHE}/candidates.json"))]
    ids = a.cases.split(",") if a.cases else default
    if a.stage == "predict": predict(a.models.split(","), ids, os.path.abspath(a.data_dir))
    elif a.stage == "table": table(ids)
    else: draw(ids)
