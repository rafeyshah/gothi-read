#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a CSV manifest of (image, txt, font) triples with Unicode-safe char counts.

Example:
  python scripts/build_manifest.py --data-root /path/to/data --splits train val --out-dir manifests
"""

import argparse
import csv
import sys
from pathlib import Path
import regex as re
from pathlib import Path


GRAPHEME_RE = re.compile(r"\X", re.U)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def split_graphemes(text: str):
    # Keep text as-is; normalization can be added later if needed.
    return [g for g in GRAPHEME_RE.findall(text)]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def read_font_labels(p: Path):
    """
    Return a list of per-char font labels, matching icdar24.py semantics:
    - .font file is a concatenated sequence like 'aafGs...'
    - Any whitespace is removed before counting
    """
    raw = p.read_text(encoding="utf-8")
    raw = re.sub(r"\s+", "", raw)  # remove all whitespace, be tolerant
    return list(raw)  # one label per char


def id_key(split_root: Path, p: Path) -> str:
    # relative path without the extension, using '/' as separator
    rel = p.relative_to(split_root)
    return str(rel.with_suffix("")).replace("\\", "/")


def discover_triples(split_root: Path):
    images = {}
    texts = {}
    fonts = {}

    for p in split_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        key = id_key(split_root, p)
        if ext in IMG_EXTS:
            images[key] = p
        elif ext == ".txt":
            texts[key] = p
        elif ext == ".font":
            fonts[key] = p

    triples = []
    all_keys = set(images) | set(texts) | set(fonts)
    for k in sorted(all_keys):
        triples.append((k, images.get(k), texts.get(k), fonts.get(k)))
    return triples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root data dir containing split folders (e.g., train/, val/).")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val"], help="Split folder names under data-root.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("manifests"), help="Where to write CSVs.")
    ap.add_argument("--fail-if-missing", action="store_true",
                    help="Exit non-zero if any item is missing .txt/.font.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_missing = 0
    for split in args.splits:
        split_root = args.data_root / split
        if not split_root.exists():
            print(
                f"[WARN] Split folder not found: {split_root}", file=sys.stderr)
            continue

        triples = discover_triples(split_root)
        out_csv = args.out_dir / f"{split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "id", "image_path", "txt_path", "font_path",
                    "num_graphemes", "num_fonts", "ok"
                ]
            )
            writer.writeheader()

            for stem, img_p, txt_p, font_p in triples:
                rec = {
                    "split": split,
                    "id": stem,
                    "image_path": str(img_p) if img_p else "",
                    "txt_path": str(txt_p) if txt_p else "",
                    "font_path": str(font_p) if font_p else "",
                    "num_graphemes": "",
                    "num_fonts": "",
                    "ok": ""
                }

                missing = (img_p is None) or (
                    txt_p is None) or (font_p is None)
                if missing:
                    overall_missing += 1
                    rec["num_graphemes"] = ""
                    rec["num_fonts"] = ""
                    rec["ok"] = "MISSING"
                    writer.writerow(rec)
                    continue

                # Load and compare lengths
                text = read_text(txt_p).rstrip("\n")
                graphemes = split_graphemes(text)
                fonts = read_font_labels(font_p)

                rec["num_graphemes"] = len(graphemes)
                rec["num_fonts"] = len(fonts)
                rec["ok"] = "TRUE" if len(graphemes) == len(
                    fonts) else "LEN_MISMATCH"
                writer.writerow(rec)

        print(f"[OK] Wrote {out_csv} ({len(triples)} rows)")

    if args.fail_if_missing and overall_missing > 0:
        print(
            f"[ERROR] Missing files found across splits: {overall_missing}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
