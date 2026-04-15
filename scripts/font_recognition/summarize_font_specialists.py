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
        description="Summarize omni vs font-specialist metrics."
    )
    ap.add_argument("--omni_metrics", required=True)
    ap.add_argument("--specialist_metrics", required=True, help="Comma-separated font=metrics.json pairs.")
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_pairs(raw: str) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, path = item.split("=", 1)
        out.append((key.strip(), path.strip()))
    return out


def main():
    args = parse_args()
    omni = read_json(args.omni_metrics)
    rows = [
        {
            "model_scope": "omni",
            "font_group": "all",
            "font_cer": omni.get("font_cer"),
            "num_lines": omni.get("num_lines", ""),
            "metrics_path": str(Path(args.omni_metrics).resolve()),
        }
    ]
    for font_group, path in parse_pairs(args.specialist_metrics):
        m = read_json(path)
        rows.append(
            {
                "model_scope": "specialist",
                "font_group": font_group,
                "font_cer": m.get("font_cer"),
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
