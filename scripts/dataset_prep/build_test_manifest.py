#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build an image-only manifest for the public ICDAR 2024 test set."
    )
    ap.add_argument(
        "--images_root",
        type=Path,
        required=True,
        help="Folder containing public-test images.",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for images under --images_root.",
    )
    return ap.parse_args()


def iter_images(root: Path, recursive: bool):
    walker = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(walker):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def main():
    args = parse_args()
    root = args.images_root.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"images root not found: {root}")

    rows = []
    for path in iter_images(root, recursive=args.recursive):
        sample_id = path.stem
        rows.append(
            {
                "id": sample_id,
                "image_path": str(path),
                "image_name": path.name,
                "stem": sample_id,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_path", "image_name", "stem"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
