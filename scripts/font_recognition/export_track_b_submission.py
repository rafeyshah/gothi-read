#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import regex as re
except ImportError:
    import re


COMPETITION_FONTS = {"a", "b", "f", "G", "i", "r", "s", "t"}
COMP_PATTERN = re.compile(
    r"(.[\u02F3\u1D53\u0300\u2013\u032E\u208D\u203F\u0311\u0323\u035E\u031C\u02FC\u030C\u02F9\u0328\u032D\u02F4\u032F\u0330\u035C\u0302\u0327\u0357\u0308\u0351\u0304\u02F2\u0352\u0355\u032C\u030B\u0339\u0301\u02F1\u0303\u0306\u030A\u0325\u0307\u0354\u02F0\u0060\u030d\u0364\u0303]*)",
    re.UNICODE | re.IGNORECASE,
)


def split_competition_text(text: str) -> List[str]:
    return list(COMP_PATTERN.findall(text or ""))


def parse_pred_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = str(row.get("id") or row.get("img_id") or "").strip()
                pred = str(
                    row.get("pred_text")
                    or row.get("pred_fonts")
                    or row.get("pred")
                    or row.get("pred_ensemble")
                    or ""
                )
                if sample_id:
                    out[sample_id] = pred
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    sample_id, pred = line.split("\t", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    sample_id, pred = parts
                out[sample_id.strip()] = pred
    return out


def load_manifest_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("id") or row.get("stem") or "").strip()
            image_path = str(row.get("image_path") or "").strip()
            if not sample_id and image_path:
                sample_id = Path(image_path).stem
            if sample_id:
                ids.append(sample_id)
    return ids


def align_font_sequence(font_seq: str, target_len: int, mode: str) -> str:
    seq = list((font_seq or "").strip())
    if target_len <= 0:
        return ""
    if not seq:
        return "a" * target_len
    if mode == "strict":
        if len(seq) != target_len:
            raise ValueError(f"font length {len(seq)} != target length {target_len}")
        return "".join(seq)
    if len(seq) == target_len:
        return "".join(seq)
    if len(seq) == 1:
        return seq[0] * target_len
    aligned = []
    src_last = len(seq) - 1
    for i in range(target_len):
        src_idx = round(i * src_last / max(1, target_len - 1))
        aligned.append(seq[src_idx])
    return "".join(aligned)


def validate_font_labels(font_seq: str) -> List[str]:
    return sorted({ch for ch in font_seq if ch not in COMPETITION_FONTS})


def write_preview(rows: Sequence[dict], out_csv: Path):
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "ocr_text",
                "font_out",
                "ocr_len",
                "font_pred_len",
                "font_out_len",
                "font_align_changed",
                "invalid_labels",
            ],
        )
        writer.writeheader()
        for row in rows[:200]:
            writer.writerow(row)


def iter_submission_rows(sample_ids: Iterable[str], ocr_map: Dict[str, str], font_map: Dict[str, str], align_mode: str):
    rows = []
    missing_ocr = 0
    missing_font = 0
    changed = 0
    invalid_labels = set()
    for sample_id in sample_ids:
        text = ocr_map.get(sample_id, "")
        font_pred = font_map.get(sample_id, "")
        if sample_id not in ocr_map:
            missing_ocr += 1
        if sample_id not in font_map:
            missing_font += 1
        target_len = len(split_competition_text(text))
        font_out = align_font_sequence(font_pred, target_len=target_len, mode=align_mode)
        if font_out != font_pred:
            changed += 1
        invalid = validate_font_labels(font_out)
        invalid_labels.update(invalid)
        rows.append(
            {
                "id": sample_id,
                "ocr_text": text,
                "font_out": font_out,
                "ocr_len": target_len,
                "font_pred_len": len(font_pred),
                "font_out_len": len(font_out),
                "font_align_changed": int(font_out != font_pred),
                "invalid_labels": "".join(invalid),
            }
        )
    return rows, missing_ocr, missing_font, changed, sorted(invalid_labels)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export ICDAR 2024 Track B submission files (.txt/.font + zip)."
    )
    ap.add_argument("--ocr_preds", required=True, help="preds.txt/csv with id->OCR text")
    ap.add_argument("--font_preds", required=True, help="preds.txt/csv with id->font sequence")
    ap.add_argument(
        "--manifest_csv",
        required=True,
        help="Image-only public-test manifest; defines which ids to export.",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--align_mode",
        default="resample",
        choices=["resample", "strict"],
        help="How to force font length to match OCR character count.",
    )
    ap.add_argument(
        "--zip_name",
        default="trackb_submission.zip",
        help="Name of the generated zip file.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    export_dir = out_dir / "submission_files"
    export_dir.mkdir(parents=True, exist_ok=True)

    ocr_map = parse_pred_file(Path(args.ocr_preds))
    font_map = parse_pred_file(Path(args.font_preds))
    sample_ids = load_manifest_ids(Path(args.manifest_csv))
    if not sample_ids:
        raise RuntimeError("No ids found in manifest_csv.")

    rows, missing_ocr, missing_font, changed, invalid_labels = iter_submission_rows(
        sample_ids, ocr_map, font_map, args.align_mode
    )

    for row in rows:
        sample_id = row["id"]
        (export_dir / f"{sample_id}.txt").write_text(row["ocr_text"], encoding="utf-8")
        (export_dir / f"{sample_id}.font").write_text(row["font_out"], encoding="utf-8")

    zip_path = out_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(export_dir.glob("*")):
            zf.write(path, arcname=path.name)

    stats = {
        "num_lines": len(rows),
        "missing_ocr_ids": missing_ocr,
        "missing_font_ids": missing_font,
        "aligned_length_mismatches": changed,
        "rows_with_invalid_labels": sum(1 for row in rows if row["invalid_labels"]),
        "invalid_labels": invalid_labels,
        "output_dir": str(export_dir),
        "zip_file": str(zip_path),
        "align_mode": args.align_mode,
        "splitter": "competition_published_regex",
    }
    (out_dir / "submission_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_preview(rows, out_dir / "submission_preview.csv")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
