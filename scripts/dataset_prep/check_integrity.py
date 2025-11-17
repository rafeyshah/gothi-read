#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check integrity summaries using the generated manifest CSVs.

Example:
  python scripts/check_integrity.py --manifests manifests/train.csv manifests/val.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


def summarize(csv_path: Path):
    c = Counter()
    total = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            img = row["image_path"]
            txt = row["txt_path"]
            font = row["font_path"]
            ok = row["ok"]

            if not img:
                c["missing_image"] += 1
            if not txt:
                c["missing_txt"] += 1
            if not font:
                c["missing_font"] += 1
            if ok == "LEN_MISMATCH":
                c["len_mismatch"] += 1
            if ok == "TRUE":
                c["ok_true"] += 1
            if ok == "MISSING":
                c["missing_any"] += 1

    clean = c["ok_true"]
    issues = total - clean
    pct_clean = (clean / total * 100.0) if total else 0.0

    print(f"\n== {csv_path.name} ==")
    print(f"Total lines         : {total}")
    print(f"Clean (ok=True)     : {clean} ({pct_clean:.2f}%)")
    print(f"Missing image       : {c['missing_image']}")
    print(f"Missing txt         : {c['missing_txt']}")
    print(f"Missing font        : {c['missing_font']}")
    print(f"Any missing (.img/.txt/.font): {c['missing_any']}")
    print(f"Length mismatches   : {c['len_mismatch']}")
    print(f"Issues total        : {issues}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=Path, nargs="+",
                    required=True, help="CSV manifest paths.")
    args = ap.parse_args()
    for p in args.manifests:
        if not p.exists():
            print(f"[WARN] Manifest not found: {p}")
            continue
        summarize(p)


if __name__ == "__main__":
    main()
