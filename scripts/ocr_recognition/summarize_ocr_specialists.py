#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize omni vs OCR specialist metrics from PaddleOCR runs."
    )
    ap.add_argument("--omni_metrics", required=True)
    ap.add_argument(
        "--specialist_metrics",
        required=True,
        help="Comma-separated list in family=path/to/metrics.json form.",
    )
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_specialist_arg(raw: str) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        family, path = item.split("=", 1)
        pairs.append((family.strip(), path.strip()))
    return pairs


def main():
    args = parse_args()
    omni = read_json(args.omni_metrics)
    rows = [{
        "model_scope": "omni",
        "family": "all",
        "CER": omni.get("CER"),
        "WER": omni.get("WER"),
        "num_lines": omni.get("num_lines", ""),
        "metrics_path": str(Path(args.omni_metrics).resolve()),
    }]
    for family, path in parse_specialist_arg(args.specialist_metrics):
        m = read_json(path)
        rows.append(
            {
                "model_scope": "specialist",
                "family": family,
                "CER": m.get("CER"),
                "WER": m.get("WER"),
                "num_lines": m.get("num_lines", ""),
                "metrics_path": str(Path(path).resolve()),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
