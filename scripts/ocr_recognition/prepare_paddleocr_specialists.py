#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


SINGLE_FAMILIES = [
    "antiqua",
    "bastarda",
    "fraktur",
    "gotico-antiqua",
    "italic",
    "rotunda",
    "schwabacher",
    "textura",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare PaddleOCR specialist subsets/configs for Gothi-Read OCR."
    )
    ap.add_argument("--train_manifest", default="configs/train_clean.csv")
    ap.add_argument("--val_manifest", default="configs/valid_clean.csv")
    ap.add_argument("--template_config", default="configs/PP-OCRv5_gothi_rec.yml")
    ap.add_argument("--out_root", default="runs/paddleocr-specialists")
    ap.add_argument(
        "--families",
        default="all_single",
        help="Comma list of families, or all_single, or all_single_plus_multiple.",
    )
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--paddleocr_root", default=None, help="Optional /content/PaddleOCR root.")
    ap.add_argument("--train_tool", default="tools/train.py")
    ap.add_argument("--eval_tool", default="tools/eval.py")
    ap.add_argument("--export_tool", default="tools/export_model.py")
    ap.add_argument("--run_train", action="store_true")
    ap.add_argument("--run_eval", action="store_true")
    ap.add_argument("--run_export", action="store_true")
    return ap.parse_args()


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


def parse_family_arg(raw: str) -> List[str]:
    key = raw.strip().lower()
    if key == "all_single":
        return list(SINGLE_FAMILIES)
    if key == "all_single_plus_multiple":
        return list(SINGLE_FAMILIES) + ["multiple"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_dataset_path(path_str: str, sample_id: str, suffix: str, data_root: Optional[str]) -> Path:
    raw = Path(path_str)
    if raw.exists():
        return raw
    if not data_root:
        return raw
    root = Path(data_root)
    norm = path_str.replace("\\", "/")
    marker = "/dataset/"
    cands: List[Path] = []
    if marker in norm:
        cands.append(root / norm.split(marker, 1)[1])
    sid = sample_id.replace("\\", "/").strip("/")
    if sid:
        cands.append(root / sid).with_suffix(suffix)
    for cand in cands:
        if cand.exists():
            return cand
    return cands[0] if cands else raw


def read_gt_text(row: Dict[str, str], data_root: Optional[str]) -> str:
    gt_text = row.get("gt_text")
    if isinstance(gt_text, str) and gt_text != "":
        return gt_text.strip()
    sample_id = str(row.get("id", "")).strip()
    txt_path = str(row.get("txt_path", "")).strip()
    if not txt_path:
        return ""
    resolved = resolve_dataset_path(txt_path, sample_id, ".txt", data_root)
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8").strip()


def subset_manifest_rows(manifest_path: Path, family: str) -> tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            ok = str(row.get("ok", "")).strip().upper()
            if ok not in {"TRUE", "1", "YES", "Y"}:
                continue
            if infer_family(row) != family:
                continue
            row["family"] = family
            rows.append(row)
    if "family" not in fieldnames:
        fieldnames.append("family")
    return rows, fieldnames


def write_subset_manifest(out_csv: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rec_labels(rows: Iterable[Dict[str, str]], out_txt: Path, data_root: Optional[str]) -> int:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_txt.open("w", encoding="utf-8") as f:
        for row in rows:
            sample_id = str(row.get("id", "")).strip()
            image_path = str(row.get("image_path", "")).strip()
            text = read_gt_text(row, data_root)
            if not image_path or text == "":
                continue
            resolved_img = resolve_dataset_path(image_path, sample_id, Path(image_path).suffix or ".jpg", data_root)
            f.write(f"{resolved_img}\t{text}\n")
            count += 1
    return count


def update_ppocr_config(template_path: Path, train_labels: Path, val_labels: Path, out_dir: Path, family: str) -> dict:
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    cfg["Global"]["save_model_dir"] = str((out_dir / "output").resolve())
    cfg["Global"]["save_inference_dir"] = str((out_dir / "inference").resolve())
    cfg["Global"]["save_res_path"] = str((out_dir / f"predicts_{family}.txt").resolve())
    cfg["Train"]["dataset"]["label_file_list"] = [str(train_labels.resolve())]
    cfg["Eval"]["dataset"]["label_file_list"] = [str(val_labels.resolve())]
    return cfg


def paddle_cmd(paddleocr_root: Path, tool_rel: str, config_path: Path, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [sys.executable, str((paddleocr_root / tool_rel).resolve()), "-c", str(config_path.resolve())]
    if extra:
        cmd.extend(extra)
    return cmd


def run_cmd(cmd: List[str], cwd: Optional[Path] = None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main():
    args = parse_args()
    families = parse_family_arg(args.families)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.template_config)
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    paddleocr_root = Path(args.paddleocr_root) if args.paddleocr_root else None

    rows_out = []
    for family in families:
        spec_dir = out_root / family
        train_rows, fieldnames = subset_manifest_rows(train_manifest, family)
        val_rows, _ = subset_manifest_rows(val_manifest, family)

        train_subset_csv = spec_dir / "train_subset.csv"
        val_subset_csv = spec_dir / "val_subset.csv"
        write_subset_manifest(train_subset_csv, train_rows, fieldnames)
        write_subset_manifest(val_subset_csv, val_rows, fieldnames)

        train_labels = spec_dir / "rec_gt_train.txt"
        val_labels = spec_dir / "rec_gt_val.txt"
        num_train = write_rec_labels(train_rows, train_labels, args.data_root)
        num_val = write_rec_labels(val_rows, val_labels, args.data_root)

        cfg = update_ppocr_config(template_path, train_labels, val_labels, spec_dir, family)
        config_path = spec_dir / f"PP-OCRv5_gothi_{family}.yml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        train_cmd = eval_cmd = export_cmd = None
        if paddleocr_root is not None:
            train_cmd = paddle_cmd(paddleocr_root, args.train_tool, config_path)
            eval_cmd = paddle_cmd(
                paddleocr_root,
                args.eval_tool,
                config_path,
                extra=["-o", f"Global.checkpoints={str((spec_dir / 'output' / 'best_accuracy').resolve())}"],
            )
            export_cmd = paddle_cmd(
                paddleocr_root,
                args.export_tool,
                config_path,
                extra=[
                    "-o",
                    f"Global.checkpoints={str((spec_dir / 'output' / 'best_accuracy').resolve())}",
                    f"Global.save_inference_dir={str((spec_dir / 'inference').resolve())}",
                ],
            )
            if args.run_train:
                run_cmd(train_cmd, cwd=paddleocr_root)
            if args.run_eval:
                run_cmd(eval_cmd, cwd=paddleocr_root)
            if args.run_export:
                run_cmd(export_cmd, cwd=paddleocr_root)

        row = {
            "family": family,
            "num_train_rows": len(train_rows),
            "num_val_rows": len(val_rows),
            "num_train_labels": num_train,
            "num_val_labels": num_val,
            "subset_dir": str(spec_dir.resolve()),
            "train_subset_csv": str(train_subset_csv.resolve()),
            "val_subset_csv": str(val_subset_csv.resolve()),
            "train_labels": str(train_labels.resolve()),
            "val_labels": str(val_labels.resolve()),
            "config_path": str(config_path.resolve()),
            "train_cmd": " ".join(train_cmd) if train_cmd else "",
            "eval_cmd": " ".join(eval_cmd) if eval_cmd else "",
            "export_cmd": " ".join(export_cmd) if export_cmd else "",
        }
        rows_out.append(row)

    leaderboard_csv = out_root / "specialists.csv"
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    (out_root / "specialists.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
