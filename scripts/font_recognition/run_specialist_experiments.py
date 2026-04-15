#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


FONT_GROUPS = ["a", "b", "f", "G", "i", "r", "s", "t"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train/evaluate one dominant-font specialist model per font group."
    )
    ap.add_argument("--train_script", default="scripts/font_recognition/train_font_ctc.py")
    ap.add_argument("--eval_script", default="scripts/font_recognition/eval_font_cer.py")
    ap.add_argument("--train_csv", default="configs/train_clean.csv")
    ap.add_argument("--val_csv", default="configs/valid_clean.csv")
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="valid")
    ap.add_argument("--out_root", default="runs/font-specialists")
    ap.add_argument("--arch", default="crnn", choices=["crnn", "transformer"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--image_height", type=int, default=48)
    ap.add_argument("--max_width", type=int, default=1536)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smooth_window", type=int, default=1)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--eval_on_full_val", action="store_true")
    ap.add_argument("--no_weighted_sampler", action="store_true")
    return ap.parse_args()


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for font_group in FONT_GROUPS:
        model_dir = out_root / f"{args.arch}_{font_group}"
        train_cmd = [
            sys.executable,
            args.train_script,
            "--train_csv",
            args.train_csv,
            "--val_csv",
            args.val_csv,
            "--font_vocab",
            args.font_vocab,
            "--data_root",
            args.data_root,
            "--train_split",
            args.train_split,
            "--val_split",
            args.val_split,
            "--out_dir",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--image_height",
            str(args.image_height),
            "--max_width",
            str(args.max_width),
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--arch",
            args.arch,
            "--dominant_font_filter",
            font_group,
        ]
        if args.train_limit is not None:
            train_cmd.extend(["--train_limit", str(args.train_limit)])
        if args.val_limit is not None:
            train_cmd.extend(["--val_limit", str(args.val_limit)])
        if args.no_weighted_sampler:
            train_cmd.append("--no_weighted_sampler")
        run_cmd(train_cmd)

        eval_manifest = args.val_csv
        eval_filter = None if args.eval_on_full_val else font_group
        eval_dir = model_dir / ("eval_full_val" if args.eval_on_full_val else "eval_matched_val")
        eval_cmd = [
            sys.executable,
            args.eval_script,
            "--checkpoint",
            str(model_dir / "best.pt"),
            "--manifest_csv",
            eval_manifest,
            "--font_vocab",
            args.font_vocab,
            "--data_root",
            args.data_root,
            "--split",
            args.val_split,
            "--out_dir",
            str(eval_dir),
            "--arch",
            args.arch,
            "--smooth_window",
            str(args.smooth_window),
        ]
        if eval_filter is not None:
            eval_cmd.extend(["--dominant_font_filter", eval_filter])
        if args.val_limit is not None:
            eval_cmd.extend(["--limit", str(args.val_limit)])
        run_cmd(eval_cmd)

        train_summary = read_json(model_dir / "summary.json")
        eval_metrics = read_json(eval_dir / "metrics.json")
        rows.append(
            {
                "font_group": font_group,
                "arch": args.arch,
                "num_train_samples": train_summary.get("num_train_samples"),
                "num_val_samples": train_summary.get("num_val_samples"),
                "best_val_font_cer": train_summary.get("best_val_font_cer"),
                "eval_font_cer": eval_metrics.get("font_cer"),
                "eval_num_lines": eval_metrics.get("num_lines"),
                "eval_scope": "full_val" if args.eval_on_full_val else "matched_val",
                "model_dir": str(model_dir),
                "eval_dir": str(eval_dir),
            }
        )

    out_csv = out_root / "leaderboard.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_root / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
