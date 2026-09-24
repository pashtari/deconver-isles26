"""Dice vs parameters chart (Tables 1 and 2 of the SWITCH+ paper) as SVG for the poster."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "Inter", "svg.fonttype": "none", "font.size": 11})
NAVY, GRAY, TEAL, BLUE, GOLD = "#0B1F3A", "#8A94A6", "#0E9F8E", "#2F6DF6", "#E0A100"
base = {"nnU-Net": (22.57, 62.54, "s"), "SegResNet": (75.17, 63.00, "D"), "SwinUNETR-V2": (72.76, 62.41, "^")}
dec = [(6.46, 62.76, "[1,2,2,4], C₀=32"), (10.55, 62.99, "[1,1,1,1,1], C₀=32"), (11.13, 62.85, "k=5"), (24.23, 63.40, "[1,1,1,1,1], C₀=64"),
       (24.30, 62.91, "6 levels, C₀=32"), (25.46, 63.59, "Deconver [1,2,2,4], C₀=64"), (52.77, 63.56, "[1,2,2,4,4], C₀=64")]
ens = (100.63, 64.44)
fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=100)
for name, (p, d, m) in base.items():
    ax.scatter(p, d, s=170, marker=m, color=GRAY, edgecolor="white", linewidth=1.2, zorder=3)
for p, d, lab in dec:
    head = "Deconver [" in lab
    ax.scatter(p, d, s=300 if head else 120, marker="o", color=TEAL, edgecolor="white", linewidth=1.5, zorder=4)
ax.scatter(*ens, s=380, marker="*", color=GOLD, edgecolor="white", linewidth=1.2, zorder=5)
# labels
ax.annotate("nnU-Net\n22.6M", (22.57, 62.54), (20.5, 62.13), ha="center", fontsize=10.5, color=NAVY)
ax.annotate("SegResNet\n75.2M", (75.17, 63.00), (95, 63.00), ha="left", va="center", fontsize=10.5, color=NAVY)
ax.annotate("SwinUNETR-V2\n72.8M", (72.76, 62.41), (95, 62.41), ha="left", va="center", fontsize=10.5, color=NAVY)
ax.annotate("Deconver (ours)\n25.5M", (25.46, 63.59), (26.5, 63.86), ha="left", va="center", fontsize=12, fontweight="bold", color=TEAL)
ax.annotate("Ensemble: SegResNet + Deconver\n100.6M", (100.63, 64.44), (88, 64.42), ha="right", va="center", fontsize=10.5, color="#8A6B00")
ax.annotate("6.5M", (6.46, 62.76), (6.46, 62.50), ha="center", fontsize=9.5, color=TEAL)
ax.annotate("10.6M", (10.55, 62.99), (10.55, 63.20), ha="center", fontsize=9.5, color=TEAL)
ax.annotate("52.8M", (52.77, 63.56), (52.77, 63.78), ha="center", fontsize=9.5, color=TEAL)
ax.set_xscale("log"); ax.set_xlim(4.5, 230); ax.set_ylim(62.0, 64.8)
ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"])
ax.set_xlabel("Parameters (M, log scale)", color=NAVY, fontsize=11.5); ax.set_ylabel("Mean Dice (%)", color=NAVY, fontsize=11.5)
ax.grid(True, which="major", color="#E6E9EF", linewidth=1); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#C9CFD9")
ax.tick_params(colors=NAVY, labelsize=10.5)
handles = [Line2D([], [], marker="o", color=TEAL, linestyle="", markersize=9, label="Deconver variants (Table 2)"),
           Line2D([], [], marker="s", color=GRAY, linestyle="", markersize=8, label="Baselines"),
           Line2D([], [], marker="*", color=GOLD, linestyle="", markersize=13, label="Ensemble (challenge submission)")]

fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
fig.savefig(os.path.join(OUT, "dice_vs_params.svg"))
print("chart written")
