#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter manifest CSVs to keep only clean rows (ok == "TRUE").

Usage (example):
  python scripts/filter_clean.py \
    --manifests manifests/train.csv manifests/valid.csv manifests/test.csv \
    --out-dir manifests \
    --suffix _clean

This will write:
  manifests/train_clean.csv
  manifests/valid_clean.csv
  manifests/test_clean.csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path


def filter_manifest(in_csv: Path, out_csv: Path) -> tuple[int, int]:
    total = 0
    kept = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", encoding="utf-8", newline="") as fin, \
            out_csv.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            if row.get("ok", "") == "TRUE":
                writer.writerow(row)
                kept += 1

    return kept, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True,
                    help="Input manifest CSV paths (e.g., manifests/train.csv ...).")
    ap.add_argument("--out-dir", type=Path, default=Path("manifests"),
                    help="Directory to write filtered CSVs into.")
    ap.add_argument("--suffix", type=str, default="_clean",
                    help="Suffix to append to filename stem (e.g., _clean).")
    args = ap.parse_args()

    for m in args.manifests:
        in_csv = Path(m)
        if not in_csv.exists():
            print(f"[WARN] Skipping (not found): {in_csv}")
            continue

        out_csv = args.out_dir / f"{in_csv.stem}{args.suffix}{in_csv.suffix}"
        kept, total = filter_manifest(in_csv, out_csv)
        pct = (kept / total * 100.0) if total else 0.0
        print(
            f"[OK] {in_csv.name}: kept {kept}/{total} rows ({pct:.2f}%) → {out_csv}")


if __name__ == "__main__":
    main()
