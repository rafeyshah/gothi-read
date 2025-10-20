#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a line image + text + per-char font colors from the manifest.

Examples:
  python scripts/visualize_line.py --manifest manifests/train.csv --num 12
  python scripts/visualize_line.py --manifest manifests/val.csv --id 00012345
"""

import argparse
import csv
import math
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import regex as re

GRAPHEME_RE = re.compile(r"\X", re.U)

# A fixed palette for font groups; extend or map dynamically as needed.
# If your .font labels are strings like 'a','b','f','G','i','r','s','t',
# we map each unique label encountered to a palette index.
DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]


def split_graphemes(text: str):
    return [g for g in GRAPHEME_RE.findall(text)]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").rstrip("\n")


def read_font_labels(p: Path):
    import regex as re
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return re.sub(r"\s+", " ", raw).strip().split(" ")


def load_manifest(manifest_csv: Path):
    rows = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def draw_sample(img_path: Path, text: str, fonts: list, out_path: Path):
    # Load image
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # handle orientation

    # Build color map for labels
    uniq = []
    for lab in fonts:
        if lab not in uniq:
            uniq.append(lab)
    label_to_color = {lab: DEFAULT_PALETTE[i % len(
        DEFAULT_PALETTE)] for i, lab in enumerate(uniq)}

    # Plot: top = image; bottom = colored text blocks
    fig_h = 6
    fig_w = max(8, img.width / 100 * 1.0)
    plt.figure(figsize=(fig_w, fig_h))

    # 1) show image
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name)

    # 2) colored text row
    plt.subplot(2, 1, 2)
    plt.axis("off")

    # We render each grapheme as a small colored rectangle with the glyph on top
    # Compute a simple grid width based on text length
    n = max(1, len(text))
    cols = min(80, n)
    rows = math.ceil(n / cols)

    x0, y0 = 0.02, 0.1
    xw, yh = 0.96 / cols, 0.8 / max(1, rows)

    gi = 0
    for r in range(rows):
        for c in range(cols):
            if gi >= len(text):
                break
            ch = text[gi]  # already split as graphemes when calling
            font_lab = fonts[gi] if gi < len(fonts) else "?"
            color = label_to_color.get(font_lab, "#000000")

            # rectangle
            rect_x = x0 + c * xw
            rect_y = y0 + r * yh
            plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), xw*0.98,
                                              yh*0.9, facecolor=color, alpha=0.25, edgecolor="none"))
            # text (centered)
            plt.text(rect_x + xw/2, rect_y + yh/2, ch,
                     ha="center", va="center", fontsize=10)

            gi += 1

    # Legend
    legend_items = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                               markerfacecolor=col) for col in label_to_color.values()]
    plt.legend(legend_items, list(label_to_color.keys()), loc="upper left",
               ncol=4, fontsize=8, frameon=False, title="Font label")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("exp/viz"))
    ap.add_argument("--num", type=int, default=8,
                    help="Number of random samples")
    ap.add_argument("--id", type=str, default=None,
                    help="Visualize a specific id (stem) instead of random pick")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if not rows:
        print("[ERROR] Empty manifest.")
        return

    chosen = []
    if args.id is not None:
        chosen = [r for r in rows if r["id"] == args.id]
        if not chosen:
            print(f"[ERROR] id {args.id} not found in {args.manifest}")
            return
    else:
        import random
        chosen = random.sample(rows, k=min(args.num, len(rows)))

    for r in chosen:
        if r["ok"] not in ("TRUE", "LEN_MISMATCH"):
            # skip items with missing files
            continue

        img_p = Path(r["image_path"])
        txt_p = Path(r["txt_path"])
        font_p = Path(r["font_path"])

        text = read_text(txt_p)
        graphemes = split_graphemes(text)
        fonts = read_font_labels(font_p)

        # If lengths mismatch, trim to min so we can still visualize
        L = min(len(graphemes), len(fonts))
        if L == 0:
            continue
        graphemes = graphemes[:L]
        fonts = fonts[:L]

        out_png = args.out_dir / f"{r['split']}_{r['id']}.png"
        draw_sample(img_p, graphemes, fonts, out_png)
        print(f"[OK] Wrote {out_png}")


if __name__ == "__main__":
    main()
