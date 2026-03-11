#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

from paddleocr import PaddleOCR


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run PaddleOCR recognition on unlabeled images and save predictions."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with at least id,image_path columns.")
    src.add_argument("--images_root", help="Folder containing public-test images.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--recursive", action="store_true", help="Recursively scan --images_root.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--rec_model_dir", required=True)
    ap.add_argument("--rec_char_dict_path", required=True)
    ap.add_argument("--max_text_length", type=int, default=80)
    return ap.parse_args()


def init_ocr(args) -> PaddleOCR:
    return PaddleOCR(
        use_gpu=bool(args.use_gpu),
        use_angle_cls=False,
        det=False,
        rec=True,
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        max_text_length=args.max_text_length,
        show_log=False,
    )


def load_manifest_rows(path: Path, limit: int | None) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("id") or row.get("stem") or "").strip()
            image_path = str(row.get("image_path") or "").strip()
            if not sample_id and image_path:
                sample_id = Path(image_path).stem
            if not sample_id or not image_path:
                continue
            rows.append((sample_id, image_path))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_image_rows(root: Path, recursive: bool, limit: int | None) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    walker = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(walker):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        rows.append((path.stem, str(path)))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def recognize_all(ocr: PaddleOCR, image_rows: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    for sample_id, image_path in image_rows:
        res = ocr.ocr(image_path, det=False, cls=False)
        if not res or not res[0]:
            out.append((sample_id, "", 0.0))
            continue
        first = res[0][0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[0], str):
            text = first[0]
            score = float(first[1]) if isinstance(first[1], (int, float)) else 0.0
        else:
            text = str(first)
            score = 0.0
        out.append((sample_id, text, score))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        image_rows = load_manifest_rows(Path(args.manifest), args.limit)
        source = str(Path(args.manifest).expanduser().resolve())
    else:
        root = Path(args.images_root).expanduser().resolve()
        image_rows = load_image_rows(root, recursive=args.recursive, limit=args.limit)
        source = str(root)

    if not image_rows:
        raise RuntimeError("No images found for OCR inference.")

    ocr = init_ocr(args)
    t0 = time.perf_counter()
    preds = recognize_all(ocr, image_rows)
    elapsed = time.perf_counter() - t0

    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for sample_id, text, _ in preds:
            f.write(f"{sample_id}\t{text}\n")

    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pred_text", "score"])
        writer.writeheader()
        for sample_id, text, score in preds:
            writer.writerow({"id": sample_id, "pred_text": text, "score": score})

    metrics = {
        "num_lines": len(preds),
        "runtime_seconds": elapsed,
        "avg_latency_ms_per_line": elapsed * 1000.0 / max(1, len(preds)),
        "source": source,
        "rec_model_dir": args.rec_model_dir,
        "rec_char_dict_path": args.rec_char_dict_path,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
