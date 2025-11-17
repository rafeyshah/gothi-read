#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TroCR evaluation for Gothi-Read (ICDAR 2024 Track B)

- Uses GPU if available (Colab T4 etc.).
- Reads lines from manifest CSV (image_path, txt_path, ok).
- No manual resizing: TrOCRProcessor handles preprocessing.
- Batched inference for speed.
- Outputs:
    - preds.txt      (img_id \t prediction)
    - perline.csv    (img_id, gt, pred, CER)
    - metrics.json   (CER, WER, per_book, per_fontmix)

Example usage (on Colab):

    # Handwritten base (faster)
    python trocr_eval.py \
        --manifest /content/manifests/valid_clean.csv \
        --model microsoft/trocr-base-handwritten \
        --num_beams 1 \
        --batch_size 8 \
        --max_length 128 \
        --limit 8000

    # Large printed (slower, stronger)
    python trocr_eval.py \
        --manifest /content/manifests/valid_clean.csv \
        --model microsoft/trocr-large-printed \
        --num_beams 1 \
        --batch_size 4 \
        --max_length 128 \
        --limit 8000
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ---------------------------------------------------------------------
# Edit distance / CER / WER
# ---------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Make sure a is the shorter string
    if n > m:
        a, b = b, a
        n, m = m, n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for j in range(1, m + 1):
        curr[0] = j
        cb = b[j - 1]
        for i in range(1, n + 1):
            ca = a[i - 1]
            cost = 0 if ca == cb else 1
            curr[i] = min(
                prev[i] + 1,        # deletion
                curr[i - 1] + 1,    # insertion
                prev[i - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def cer_per_line(gt: str, pred: str) -> float:
    denom = max(1, len(gt))
    return _edit_distance(gt, pred) / denom


def compute_overall_metrics(
    ids: List[str],
    gts: List[str],
    preds: List[str],
) -> Dict[str, Any]:
    """Compute overall CER, WER and group-wise CER."""
    # Overall CER
    total_edits = 0
    total_chars = 0
    for gt, pred in zip(gts, preds):
        total_edits += _edit_distance(gt, pred)
        total_chars += max(1, len(gt))
    overall_cer = total_edits / max(1, total_chars)

    # Overall WER
    total_word_edits = 0
    total_words = 0
    for gt, pred in zip(gts, preds):
        gt_words = gt.split()
        pred_words = pred.split()
        total_word_edits += _edit_distance(" ".join(gt_words),
                                           " ".join(pred_words))
        total_words += max(1, len(gt_words))
    overall_wer = total_word_edits / max(1, total_words)

    # Per-line (for later CSV)
    per_line = [
        {"img_id": i, "gt": g, "pred": p, "CER": cer_per_line(g, p)}
        for i, g, p in zip(ids, gts, preds)
    ]

    # Rough per-book / per-fontmix stats (based on img_id path)
    def infer_fontmix(img_id: str) -> str:
        if "single" in img_id:
            return "single"
        if "multiple" in img_id:
            return "multiple"
        return "unknown"

    def infer_book(img_id: str) -> str:
        parts = img_id.split("/")
        return parts[-2] if len(parts) >= 2 else "unknown"

    per_book: Dict[str, Dict[str, Any]] = {}
    per_fontmix: Dict[str, Dict[str, Any]] = {}

    for row in per_line:
        img_id = row["img_id"]
        gt = row["gt"]
        pred = row["pred"]

        book = infer_book(img_id)
        fm = infer_fontmix(img_id)

        for key, store in ((book, per_book), (fm, per_fontmix)):
            d = store.setdefault(key, {"edits": 0, "chars": 0, "lines": 0})
            d["edits"] += _edit_distance(gt, pred)
            d["chars"] += max(1, len(gt))
            d["lines"] += 1

    for store in (per_book, per_fontmix):
        for k, v in store.items():
            v["CER"] = v["edits"] / max(1, v["chars"])

    return {
        "CER": overall_cer,
        "WER": overall_wer,
        "per_book": per_book,
        "per_fontmix": per_fontmix,
    }


# ---------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------

def load_manifest(
    csv_path: str,
    limit: int = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load id, image_path, txt_path from manifest CSV.
    Only uses rows with ok == TRUE and existing txt files.
    """
    ids: List[str] = []
    imgs: List[str] = []
    gts: List[str] = []

    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ok") != "TRUE":
                continue

            img = row.get("image_path")
            txt = row.get("txt_path")
            if not img or not txt:
                continue

            txt_path = Path(txt)
            if not txt_path.is_file():
                continue

            try:
                gt = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue

            img_id = row.get("id") or Path(img).stem

            ids.append(img_id)
            imgs.append(img)
            gts.append(gt)

            if limit is not None and len(ids) >= limit:
                break

    return ids, imgs, gts


# ---------------------------------------------------------------------
# TroCR prediction (GPU, batched, no manual resize)
# ---------------------------------------------------------------------

def predict_with_trocr(
    model_name: str,
    image_paths: List[str],
    max_length: int = 128,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
    batch_size: int = 4,
) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    preds: List[str] = []

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": early_stopping,
    }

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, **gen_kwargs)
            texts = processor.batch_decode(out_ids, skip_special_tokens=True)

            preds.extend([t.strip() for t in texts])

            if (i // batch_size) % 50 == 0:
                print(
                    f"Processed {i + len(batch_paths)}/{len(image_paths)} lines")

    return preds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Path to CSV manifest (image_path, txt_path, ok).")
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten",
                    help="HF model, e.g. microsoft/trocr-large-printed or microsoft/trocr-base-handwritten")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit for quick runs.")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir (default: <manifest_dir>/runs/<model_short>/)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ids, img_paths, gts = load_manifest(str(manifest_path), args.limit)

    if not ids:
        raise SystemExit(
            "No valid lines found in manifest (check 'ok' column and paths).")

    model_short = args.model.split("/")[-1]
    out_dir = Path(
        args.out_dir) if args.out_dir else manifest_path.parent / "runs" / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(ids)} lines from {manifest_path}")
    print(f"Model: {args.model}")
    print(f"Output dir: {out_dir}")

    preds = predict_with_trocr(
        model_name=args.model,
        image_paths=img_paths,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        early_stopping=bool(args.early_stopping),
        batch_size=args.batch_size,
    )

    # Metrics
    metrics_struct = compute_overall_metrics(ids, gts, preds)

    # preds.txt
    preds_txt = out_dir / "preds.txt"
    preds_txt.write_text(
        "\n".join(f"{img_id}\t{pred}" for img_id, pred in zip(ids, preds)),
        encoding="utf-8",
    )

    # perline.csv
    perline_csv = out_dir / "perline.csv"
    with perline_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_id", "gt", "pred", "CER"])
        writer.writeheader()
        for img_id, gt, pred in zip(ids, gts, preds):
            writer.writerow(
                {
                    "img_id": img_id,
                    "gt": gt,
                    "pred": pred,
                    "CER": cer_per_line(gt, pred),
                }
            )

    # metrics.json
    metrics_json = out_dir / "metrics.json"
    meta = {
        "model": args.model,
        "decode": {
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": bool(args.early_stopping),
        },
        "CER": metrics_struct["CER"],
        "WER": metrics_struct["WER"],
        "per_book": metrics_struct["per_book"],
        "per_fontmix": metrics_struct["per_fontmix"],
    }
    metrics_json.write_text(json.dumps(
        meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Done. CER={metrics_struct['CER']:.6f}, WER={metrics_struct['WER']:.6f}")
    print("Files written:")
    print("  ", preds_txt)
    print("  ", perline_csv)
    print("  ", metrics_json)


if __name__ == "__main__":
    main()
