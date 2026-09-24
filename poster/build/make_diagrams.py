"""SVG diagrams for the poster: the Deconv Mixer and the four compared blocks (Fig. 1b)."""
import os, re
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
NAVY, SLATE, BORDER = "#0B1F3A", "#3D4A5C", "#0B1F3A"
FILL = {"norm": "#FFE9A8", "conv": "#FBE5D6", "act": "#F1F3F6", "mix": "#BFE8DE", "mlp": "#CFE4FF", "attn": "#DCD3F2", "down": "#FFD7D6", "ndc": "#B9A7E0"}
FONT = "Inter"

def stack(layers, width=170, box_h=30, gap=16, font=13, residuals=(), title=None, title_font=15, pad_top=10):
    """Vertical stack drawn bottom-up. layers: list of (label, kind). residuals: list of (from_idx, to_idx) meaning
    a skip from the input of layer from_idx to a sum node placed after layer to_idx."""
    n = len(layers)
    sum_after = {t for _, t in residuals}
    gaps = [gap + (18 if i in sum_after else 0) for i in range(n)]  # gap above layer i
    extra_top = 26 if (n - 1) in sum_after else 0
    H = pad_top + extra_top + n * box_h + sum(gaps[:-1]) + 24 + (title_font + 12 if title else 0)
    W = width + 52
    x0 = 12; ys = {}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">']
    top = pad_top + extra_top
    if title:
        svg.append(f'<text x="{W/2}" y="{H - 6}" text-anchor="middle" font-size="{title_font}" font-weight="700" fill="{NAVY}">{title}</text>')
    # positions: layer i (0 = bottom) at y
    y = top
    for i in range(n - 1, -1, -1):
        ys[i] = y; y += box_h + (gaps[i - 1] if i > 0 else 0)
    for i, (label, kind) in enumerate(layers):
        y = ys[i]
        dashed = ' stroke-dasharray="6,4"' if kind.endswith("?") else ""
        kind = kind.rstrip("?")
        svg.append(f'<rect x="{x0}" y="{y}" width="{width}" height="{box_h}" rx="4" fill="{FILL[kind]}" stroke="{BORDER}" stroke-width="2"{dashed}/>')
        svg.append(f'<text x="{x0 + width/2}" y="{y + box_h/2 + font*0.36}" text-anchor="middle" font-size="{font}" font-weight="600" fill="{NAVY}">{label}</text>')
    # arrows between boxes (upwards) and sum nodes
    sums = {t: f for f, t in residuals}
    ah = f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker></defs>'
    svg.insert(1, ah)
    cx = x0 + width / 2
    for i in range(n):
        y_top = ys[i]; y_next = ys[i + 1] + box_h if i + 1 < n else top - 20
        if i in sums:  # sum node between layer i and i+1
            yc = (y_top + y_next) / 2 if i + 1 < n else y_top - 16
            svg.append(f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{yc + 8}" stroke="{NAVY}" stroke-width="2" marker-end="url(#ah)"/>')
            svg.append(f'<circle cx="{cx}" cy="{yc}" r="7" fill="white" stroke="{NAVY}" stroke-width="2"/>')
            svg.append(f'<text x="{cx}" y="{yc + 4.5}" text-anchor="middle" font-size="13" font-weight="700" fill="{NAVY}">+</text>')
            svg.append(f'<line x1="{cx}" y1="{yc - 7}" x2="{cx}" y2="{y_next + 1 if i + 1 < n else yc - 24}" stroke="{NAVY}" stroke-width="2" marker-end="url(#ah)"/>')
            f = sums[i]; y_from = ys[f] + box_h + gap / 2  # tap below layer f
            xr = x0 + width + 14
            svg.append(f'<path d="M{cx},{y_from} H{xr} V{yc} H{cx + 8}" fill="none" stroke="{NAVY}" stroke-width="2" marker-end="url(#ah)"/>')
        else:
            svg.append(f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{y_next + 1}" stroke="{NAVY}" stroke-width="2" marker-end="url(#ah)"/>')
    # input arrow
    y_in = ys[0] + box_h
    svg.append(f'<line x1="{cx}" y1="{y_in + 20}" x2="{cx}" y2="{y_in + 1}" stroke="{NAVY}" stroke-width="2" marker-end="url(#ah)"/>')
    svg.append("</svg>")
    return "\n".join(svg)

# Deconv Mixer (Eq. 2)
mixer = stack([("Pointwise Conv", "conv"), ("ReLU", "act"), ("NDC", "ndc"), ("Pointwise Conv", "conv")], width=180, box_h=34, gap=22, font=15)
open(os.path.join(FIGS, "mixer.svg"), "w").write(mixer)
# Four compared blocks (Fig. 1b)
four = [
    ("nnU-Net", [("3³ Conv, stride S", "conv"), ("Instance Norm", "norm"), ("LeakyReLU", "act"), ("3³ Conv", "conv"), ("Instance Norm", "norm"), ("LeakyReLU", "act")], []),
    ("SegResNet", [("3³ Conv, stride S", "conv"), ("Group Norm", "norm"), ("ReLU", "act"), ("3³ Conv", "conv"), ("Group Norm", "norm"), ("ReLU", "act"), ("3³ Conv", "conv")], [(1, 6)]),
    ("SwinUNETR-V2", [("Patch merging, ↓2", "down"), ("Layer Norm", "norm"), ("W-MSA / SW-MSA", "attn"), ("Layer Norm", "norm"), ("MLP", "mlp")], [(1, 2), (3, 4)]),
    ("Deconver", [("S³ Conv, stride S", "conv"), ("Instance Norm", "norm"), ("Deconv Mixer", "mix"), ("Instance Norm", "norm"), ("MLP", "mlp")], [(1, 2), (3, 4)]),
]
parts = []; X = 0; maxH = 0
for title, layers, res in four:
    s = stack(layers, residuals=res, width=168, box_h=30, gap=18, font=13, title=title, title_font=16)
    inner = s.split(">", 1)[1].rsplit("</svg>", 1)[0]  # strip outer svg
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s); w, h = float(vb.group(1)), float(vb.group(2)); maxH = max(maxH, h)
    parts.append((X, inner, h)); X += w + 6
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {X - 6} {maxH}" font-family="{FONT}">']
for x, inner, h in parts:
    svg.append(f'<g transform="translate({x},{maxH - h})">{inner}</g>')
svg.append("</svg>")
open(os.path.join(FIGS, "blocks4.svg"), "w").write("\n".join(svg))
print("diagrams written")
