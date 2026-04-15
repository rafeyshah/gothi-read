#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter Gothi-Read manifests by layout/font family for specialist experiments."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--families",
        default=None,
        help="Comma-separated family/layout keys to keep, e.g. antiqua,fraktur or multiple.",
    )
    ap.add_argument(
        "--exclude_families",
        default=None,
        help="Comma-separated family/layout keys to drop.",
    )
    ap.add_argument("--require_ok", action="store_true")
    return ap.parse_args()


def parse_csv_list(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    vals = {x.strip() for x in raw.split(",") if x.strip()}
    return vals or None


def infer_family(row: Dict[str, str]) -> str:
    sample_id = str(row.get("id", "")).replace("\\", "/").strip("/")
    image_path = str(row.get("image_path", "")).replace("\\", "/")
    for text in (sample_id, image_path):
        parts = [p for p in text.split("/") if p]
        for i, part in enumerate(parts[:-1]):
            if part == "single" and i + 1 < len(parts):
                return parts[i + 1]
            if part == "multiple":
                return "multiple"
    parts = [p for p in sample_id.split("/") if p]
    if parts:
        return parts[0]
    return "unknown"


def main():
    args = parse_args()
    include = parse_csv_list(args.families)
    exclude = parse_csv_list(args.exclude_families)

    rows: List[Dict[str, str]] = []
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            family = infer_family(row)
            row["family"] = family
            if args.require_ok:
                ok = str(row.get("ok", "")).strip().upper()
                if ok not in {"TRUE", "1", "YES", "Y"}:
                    continue
            if include is not None and family not in include:
                continue
            if exclude is not None and family in exclude:
                continue
            rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list((fieldnames or []))
    if "family" not in fieldnames:
        fieldnames.append("family")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
