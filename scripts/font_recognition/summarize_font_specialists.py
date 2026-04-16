#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

FONT_GROUPS = ["a", "G", "f", "b", "t", "s", "r", "i"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize semi-vs-omni font-recognition metrics."
    )
    ap.add_argument("--omni_metrics", required=True)
    ap.add_argument(
        "--specialist_metrics",
        default=None,
        help="Comma-separated font=metrics.json pairs.",
    )
    ap.add_argument(
        "--specialist_root",
        default=None,
        help="Root directory containing specialist runs such as runs/font-specialists-crnn.",
    )
    ap.add_argument(
        "--arch",
        default="crnn",
        choices=["crnn", "transformer"],
        help="Specialist architecture prefix used under --specialist_root.",
    )
    ap.add_argument(
        "--eval_subdir",
        default="eval_matched_val",
        help="Evaluation subdirectory under each specialist run.",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
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


def discover_pairs(root: str, arch: str, eval_subdir: str) -> List[tuple[str, str]]:
    base = Path(root)
    out: List[tuple[str, str]] = []
    for font_group in FONT_GROUPS:
        metrics_path = base / f"{arch}_{font_group}" / eval_subdir / "metrics.json"
        if metrics_path.exists():
            out.append((font_group, str(metrics_path)))
    return out


def pick_winner(delta_font_cer: Optional[float], eps: float = 1e-12) -> str:
    if delta_font_cer is None:
        return "unknown"
    if delta_font_cer < -eps:
        return "semi"
    if delta_font_cer > eps:
        return "omni"
    return "tie"


def build_markdown(rows: List[Dict], omni_scope: str) -> str:
    lines = [
        "# Semi vs Omni Font Recognition",
        "",
        f"- Omni reference scope: `{omni_scope}`",
        f"- Compared font groups: `{len(rows)}`",
        "",
        "| Font Group | Omni Font CER | Semi Font CER | Delta CER | Omni Lines | Semi Lines | Winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {font_group} | {omni_font_cer:.6f} | {semi_font_cer:.6f} | {delta_font_cer:.6f} | {omni_num_lines} | {semi_num_lines} | {winner} |".format(
                font_group=row["font_group"],
                omni_font_cer=row["omni_font_cer"],
                semi_font_cer=row["semi_font_cer"],
                delta_font_cer=row["delta_font_cer"],
                omni_num_lines=row["omni_num_lines"],
                semi_num_lines=row["semi_num_lines"],
                winner=row["winner"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    omni = read_json(args.omni_metrics)
    omni_per_font = omni.get("per_dominant_font") or {}
    omni_scope = "per_dominant_font" if omni_per_font else "global"
    pairs = parse_pairs(args.specialist_metrics) if args.specialist_metrics else []
    if args.specialist_root:
        pairs.extend(discover_pairs(args.specialist_root, args.arch, args.eval_subdir))
    deduped: Dict[str, str] = {}
    for font_group, path in pairs:
        deduped[font_group] = path
    pairs = sorted(deduped.items(), key=lambda kv: kv[0])
    if not pairs:
        raise SystemExit("No specialist metrics found. Use --specialist_metrics or --specialist_root.")

    rows = []
    for font_group, path in pairs:
        m = read_json(path)
        omni_slice = omni_per_font.get(font_group, {}) if omni_per_font else {}
        omni_font_cer = omni_slice.get("font_cer", omni.get("font_cer"))
        omni_num_lines = omni_slice.get("lines", omni.get("num_lines", ""))
        semi_font_cer = m.get("font_cer")
        semi_num_lines = m.get("num_lines", "")
        delta_font_cer = None
        if omni_font_cer is not None and semi_font_cer is not None:
            delta_font_cer = semi_font_cer - omni_font_cer
        rows.append(
            {
                "font_group": font_group,
                "omni_font_cer": omni_font_cer,
                "semi_font_cer": semi_font_cer,
                "delta_font_cer": delta_font_cer,
                "omni_num_lines": omni_num_lines,
                "semi_num_lines": semi_num_lines,
                "winner": pick_winner(delta_font_cer),
                "omni_metrics_path": str(Path(args.omni_metrics).resolve()),
                "semi_metrics_path": str(Path(path).resolve()),
            }
        )

    rows.sort(key=lambda row: row["font_group"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "omni_metrics_path": str(Path(args.omni_metrics).resolve()),
        "omni_scope": omni_scope,
        "rows": rows,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(rows, omni_scope), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
