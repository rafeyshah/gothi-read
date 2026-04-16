#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run the full semi-vs-omni font-recognition experiment."
    )
    ap.add_argument("--train_script", default="scripts/font_recognition/train_font_ctc.py")
    ap.add_argument("--eval_script", default="scripts/font_recognition/eval_font_cer.py")
    ap.add_argument("--specialist_script", default="scripts/font_recognition/run_specialist_experiments.py")
    ap.add_argument("--summary_script", default="scripts/font_recognition/summarize_font_specialists.py")
    ap.add_argument("--train_csv", default="configs/train_clean.csv")
    ap.add_argument("--val_csv", default="configs/valid_clean.csv")
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="valid")
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
    ap.add_argument("--no_weighted_sampler", action="store_true")
    ap.add_argument("--omni_out_dir", default=None)
    ap.add_argument("--omni_eval_dir", default=None)
    ap.add_argument("--specialist_out_root", default=None)
    ap.add_argument("--summary_out_csv", default=None)
    ap.add_argument("--summary_out_json", default=None)
    ap.add_argument("--summary_out_md", default=None)
    ap.add_argument("--skip_omni_train", action="store_true")
    ap.add_argument("--skip_omni_eval", action="store_true")
    ap.add_argument("--skip_specialists", action="store_true")
    ap.add_argument("--skip_summary", action="store_true")
    return ap.parse_args()


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_extend(cmd: List[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def main():
    args = parse_args()

    omni_out_dir = Path(args.omni_out_dir or f"runs/font-ctc-omni-{args.arch}")
    omni_eval_dir = Path(args.omni_eval_dir or (omni_out_dir / "eval_valid"))
    specialist_out_root = Path(args.specialist_out_root or f"runs/font-specialists-{args.arch}")
    summary_out_csv = Path(args.summary_out_csv or (specialist_out_root / "semi_vs_omni.csv"))
    summary_out_json = Path(args.summary_out_json or (specialist_out_root / "semi_vs_omni.json"))
    summary_out_md = Path(args.summary_out_md or (specialist_out_root / "semi_vs_omni.md"))

    if not args.skip_omni_train:
        omni_out_dir.mkdir(parents=True, exist_ok=True)
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
            str(omni_out_dir),
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
        ]
        maybe_extend(train_cmd, "--train_limit", args.train_limit)
        maybe_extend(train_cmd, "--val_limit", args.val_limit)
        if args.no_weighted_sampler:
            train_cmd.append("--no_weighted_sampler")
        run_cmd(train_cmd)

    if not args.skip_omni_eval:
        omni_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            sys.executable,
            args.eval_script,
            "--checkpoint",
            str(omni_out_dir / "best.pt"),
            "--manifest_csv",
            args.val_csv,
            "--font_vocab",
            args.font_vocab,
            "--data_root",
            args.data_root,
            "--split",
            args.val_split,
            "--out_dir",
            str(omni_eval_dir),
            "--arch",
            args.arch,
            "--smooth_window",
            str(args.smooth_window),
        ]
        maybe_extend(eval_cmd, "--limit", args.val_limit)
        run_cmd(eval_cmd)

    if not args.skip_specialists:
        specialist_out_root.mkdir(parents=True, exist_ok=True)
        specialist_cmd = [
            sys.executable,
            args.specialist_script,
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
            "--out_root",
            str(specialist_out_root),
            "--arch",
            args.arch,
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
            "--smooth_window",
            str(args.smooth_window),
        ]
        maybe_extend(specialist_cmd, "--train_limit", args.train_limit)
        maybe_extend(specialist_cmd, "--val_limit", args.val_limit)
        if args.no_weighted_sampler:
            specialist_cmd.append("--no_weighted_sampler")
        run_cmd(specialist_cmd)

    if not args.skip_summary:
        summary_out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_cmd = [
            sys.executable,
            args.summary_script,
            "--omni_metrics",
            str(omni_eval_dir / "metrics.json"),
            "--specialist_root",
            str(specialist_out_root),
            "--arch",
            args.arch,
            "--out_csv",
            str(summary_out_csv),
            "--out_json",
            str(summary_out_json),
            "--out_md",
            str(summary_out_md),
        ]
        run_cmd(summary_cmd)


if __name__ == "__main__":
    main()
