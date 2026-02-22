#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def parse_int_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def parse_bool_list(s: str) -> List[bool]:
    out: List[bool] = []
    for x in s.split(","):
        t = x.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool value: {x}")
    return out


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_metrics(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: Iterable[Dict], out_csv: Path):
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_a", required=True, help="CRNN or first model checkpoint")
    ap.add_argument("--checkpoint_b", required=True, help="Transformer or second model checkpoint")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--arch_a", default=None, choices=["crnn", "transformer"])
    ap.add_argument("--arch_b", default=None, choices=["crnn", "transformer"])
    ap.add_argument("--manifest_csv", default="configs/valid_clean.csv")
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_height", type=int, default=48)
    ap.add_argument("--max_width", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smooth_windows", default="1,3,5")
    ap.add_argument("--prior_flags", default="true,false", help="Comma list, e.g. true,false")
    ap.add_argument("--expected_len_flags", default="true,false", help="Comma list, e.g. true,false")
    ap.add_argument("--prior_csv", default="configs/train_clean.csv")
    ap.add_argument("--expected_len_from", default="configs/valid_clean.csv")
    ap.add_argument("--ref_csv", default="configs/valid_clean.csv")
    ap.add_argument("--out_dir", default="runs/font-grid-search")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    eval_root = out_dir / "evals"
    ens_root = out_dir / "ensembles"
    eval_root.mkdir(parents=True, exist_ok=True)
    ens_root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "eval_font_cer.py"
    ens_script = script_dir / "ensemble_preds.py"

    windows = parse_int_list(args.smooth_windows)
    prior_flags = parse_bool_list(args.prior_flags)
    expected_flags = parse_bool_list(args.expected_len_flags)

    eval_variants: Dict[str, List[Dict]] = {args.name_a: [], args.name_b: []}
    ckpt_info = [
        (args.name_a, args.checkpoint_a, args.arch_a),
        (args.name_b, args.checkpoint_b, args.arch_b),
    ]

    # 1) Evaluate each checkpoint over smooth windows.
    for name, ckpt, arch in ckpt_info:
        for sw in windows:
            tag = f"{name}_sw{sw}"
            out = eval_root / tag
            cmd = [
                sys.executable,
                str(eval_script),
                "--checkpoint",
                ckpt,
                "--manifest_csv",
                args.manifest_csv,
                "--font_vocab",
                args.font_vocab,
                "--data_root",
                args.data_root,
                "--split",
                args.split,
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--image_height",
                str(args.image_height),
                "--max_width",
                str(args.max_width),
                "--smooth_window",
                str(sw),
                "--out_dir",
                str(out),
            ]
            if arch:
                cmd.extend(["--arch", arch])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            run_cmd(cmd)
            m = read_metrics(out / "metrics.json")
            eval_variants[name].append(
                {
                    "name": name,
                    "smooth_window": sw,
                    "font_cer": m["font_cer"],
                    "preds_path": str(out / "preds.txt"),
                    "out_dir": str(out),
                }
            )

    # 2) Grid-search ensemble over all evaluated variants + flags.
    results: List[Dict] = []
    for va, vb in itertools.product(eval_variants[args.name_a], eval_variants[args.name_b]):
        for use_prior in prior_flags:
            for use_len in expected_flags:
                ens_tag = (
                    f"{args.name_a}_sw{va['smooth_window']}__"
                    f"{args.name_b}_sw{vb['smooth_window']}__"
                    f"prior{int(use_prior)}_len{int(use_len)}"
                )
                ens_out = ens_root / ens_tag
                cmd = [
                    sys.executable,
                    str(ens_script),
                    "--pred_a",
                    va["preds_path"],
                    "--pred_b",
                    vb["preds_path"],
                    "--name_a",
                    args.name_a,
                    "--name_b",
                    args.name_b,
                    "--out_dir",
                    str(ens_out),
                ]
                if use_prior and args.prior_csv:
                    cmd.extend(["--prior_csv", args.prior_csv])
                if use_len and args.expected_len_from:
                    cmd.extend(["--expected_len_from", args.expected_len_from])
                if args.ref_csv:
                    cmd.extend(["--ref_csv", args.ref_csv])
                if args.data_root:
                    cmd.extend(["--data_root", args.data_root])
                run_cmd(cmd)

                m = read_metrics(ens_out / "metrics.json")
                row = {
                    "ensemble_tag": ens_tag,
                    "font_cer": m.get("font_cer"),
                    "num_eval_lines": m.get("num_eval_lines"),
                    "a_smooth_window": va["smooth_window"],
                    "b_smooth_window": vb["smooth_window"],
                    "use_prior": use_prior,
                    "use_expected_len": use_len,
                    "pred_a": va["preds_path"],
                    "pred_b": vb["preds_path"],
                    "ensemble_out_dir": str(ens_out),
                }
                results.append(row)

    # 3) Rank and save summary.
    results = [r for r in results if r.get("font_cer") is not None]
    results.sort(key=lambda x: x["font_cer"])
    if not results:
        raise RuntimeError("No scored runs found. Ensure --ref_csv is provided and valid.")

    leaderboard = out_dir / "leaderboard.csv"
    write_csv(results, leaderboard)
    (out_dir / "leaderboard.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    best = results[0]
    summary = {
        "best": best,
        "num_trials": len(results),
        "leaderboard_csv": str(leaderboard),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
