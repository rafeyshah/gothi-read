#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Day 5 — Zero-shot PaddleOCR recognizer for Gothi-Read (ICDAR 2024 Track B)

- Uses PaddleOCR English model (PP-OCRv4) for printed text.
- DISABLES detection; assumes input images are already line crops.
- Runs zero-shot recognition over a manifest CSV (same format as harness.py).
- Logs CER/WER + runtime, and writes:
    - preds.txt
    - per_line.csv
    - metrics.json

Tested against API described in PaddleOCR v2.9 whl docs
(ocr(..., det=False, cls=False) for recognition-only).
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path

from paddleocr import PaddleOCR  # requires paddleocr==2.x whl
from harness import load_val_split, evaluate, cer_per_line  # reuse your day3 harness


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to val/test manifest CSV (id,image_path,txt_path,ok)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for preds/metrics. "
             "Default: runs/paddleocr_en_ppocrv4/<date>.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of lines to evaluate (for quick tests)",
    )
    ap.add_argument(
        "--lang",
        type=str,
        default="en",
        help="PaddleOCR language code. Use 'en' for Latin-like printed text.",
    )
    ap.add_argument(
        "--ocr_version",
        type=str,
        default="PP-OCRv4",
        help="OCR version for PaddleOCR (e.g., 'PP-OCRv4', 'PP-OCRv3').",
    )
    ap.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available (requires paddlepaddle-gpu install).",
    )
    return ap.parse_args()


def init_paddle_ocr(lang: str, ocr_version: str, use_gpu: bool) -> PaddleOCR:
    """
    Initialize PaddleOCR in a way that is compatible with the classic whl API.

    IMPORTANT: This expects paddleocr==2.x (e.g., 2.9.0) – not the 3.x pipeline.
    """
    ocr = PaddleOCR(
        lang=lang,
        use_angle_cls=False,      # we don't expect 180° rotations for line crops
        use_gpu=use_gpu,
        ocr_version=ocr_version,  # e.g., 'PP-OCRv4'
        # do NOT pass 'rec', 'det', 'show_log' etc. here – those caused your old errors
    )
    return ocr


def recognize_line(ocr: PaddleOCR, img_path: str) -> str:
    """
    Run recognition ONLY on a single line-crop image.

    Uses:
        result = ocr.ocr(img_path, det=False, cls=False)

    For whl 2.x, "only recognition" returns a nested list whose items look like:
        [['TEXT', 0.99]]
    so we extract the first text token robustly.
    """
    result = ocr.ocr(img_path, det=False, cls=False)

    if not result:
        return ""

    # result is usually [ [ ['TEXT', score], ['OTHER', score2], ... ] ]
    batch = result[0] if isinstance(result, (list, tuple)) else []
    if not batch:
        return ""

    first = batch[0]

    # Shape 1: ['TEXT', 0.99]
    if isinstance(first, (list, tuple)) and len(first) >= 1 and isinstance(first[0], str):
        return first[0]

    # Shape 2: [[x1,y1...], ('TEXT', 0.99)] (detection+rec style, just in case)
    if (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and isinstance(first[1], (list, tuple))
        and len(first[1]) >= 1
        and isinstance(first[1][0], str)
    ):
        return first[1][0]

    # Fallback: best-effort string
    return str(first)


def main():
    args = parse_args()

    # 1) Load manifest (same helper as TrOCR harness)
    ids, img_paths, gts = load_val_split(args.manifest, limit=args.limit)

    if len(ids) == 0:
        raise RuntimeError(f"No valid lines found in manifest: {args.manifest}")

    # 2) Initialize PaddleOCR
    ocr = init_paddle_ocr(args.lang, args.ocr_version, args.use_gpu)

    # 3) Run recognition loop and measure runtime
    preds = []
    t0 = time.perf_counter()
    for img_path in img_paths:
        pred = recognize_line(ocr, img_path)
        preds.append(pred)
    t1 = time.perf_counter()

    runtime_s = t1 - t0
    avg_ms = runtime_s * 1000.0 / max(1, len(ids))

    # 4) Compute metrics using your existing harness.evaluate
    metrics = evaluate(preds, gts, ids)  # CER + WER (if jiwer installed)
    metrics["model"] = {
        "engine": "PaddleOCR",
        "lang": args.lang,
        "ocr_version": args.ocr_version,
        "use_gpu": bool(args.use_gpu),
        "det": False,
        "cls": False,
    }
    metrics["val_manifest"] = str(Path(args.manifest))
    metrics["runtime_seconds"] = runtime_s
    metrics["avg_latency_ms_per_line"] = avg_ms
    metrics["num_lines"] = len(ids)

    # 5) Decide output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        from datetime import datetime

        tag = f"paddleocr_{args.lang}_{args.ocr_version.replace('-', '').replace('_', '').lower()}"
        out_dir = Path("runs") / tag / datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Save preds.txt (same style as TrOCR harness)
    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for i, p in zip(ids, preds):
            f.write(f"{i}\t{p}\n")

    # 7) Save per_line.csv
    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "gt", "pred", "CER"])
        for i, g, p in zip(ids, gts, preds):
            w.writerow([i, g, p, cer_per_line(g, p)])

    # 8) Save metrics.json
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Saved results to", out_dir)
    print(f"Lines: {len(ids)}  |  total {runtime_s:.2f}s  |  avg {avg_ms:.2f} ms/line")


if __name__ == "__main__":
    main()
