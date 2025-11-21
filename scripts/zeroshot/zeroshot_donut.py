#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donut zero-shot evaluation for Gothi-Read (ICDAR 2024 Track B)

Very conservative on memory:
- batch_size default = 1
- smaller image size
- fp16 on GPU
- optional CPU fallback

Usage example (GPU):

    !python /content/drive/MyDrive/GothiRead/scripts/zeroshot/zeroshot_donut.py \
        --manifest /content/manifests/valid_clean.csv \
        --model sbhavy/donut-base-ocr \
        --batch_size 1 \
        --max_length 128

For quick test:

    --limit 200
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


# -----------------------
# Manifest + metrics utils
# -----------------------

def load_manifest(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    img_paths: List[str] = []
    gts: List[str] = []

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and len(ids) >= limit:
                break

            ok_val = str(row.get("ok", "")).strip().upper()
            if ok_val not in {"TRUE", "1", "YES", "Y"}:
                continue

            img = row.get("image_path") or row.get("img_path")
            txt = row.get("txt_path")
            if not img or not txt:
                continue

            img_path = Path(img)
            if not img_path.is_file():
                img_path = (csv_path.parent / img_path).resolve()
                if not img_path.is_file():
                    continue

            txt_path = Path(txt)
            if not txt_path.is_file():
                txt_path = (csv_path.parent / txt_path).resolve()
                if not txt_path.is_file():
                    continue

            gt = txt_path.read_text(encoding="utf-8").strip()
            img_id = img_path.stem

            ids.append(img_id)
            img_paths.append(str(img_path))
            gts.append(gt)

    if not ids:
        raise RuntimeError(f"No valid rows found in manifest: {csv_path}")

    return ids, img_paths, gts


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ca = a[i - 1]
        for j in range(1, n + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def compute_cer(gts: List[str], preds: List[str]) -> float:
    total_edits = 0
    total_chars = 0
    for gt, pred in zip(gts, preds):
        total_edits += _edit_distance(gt, pred)
        total_chars += max(1, len(gt))
    return total_edits / total_chars if total_chars > 0 else 0.0


def compute_wer(gts: List[str], preds: List[str]) -> float:
    def words(s: str) -> List[str]:
        return s.strip().split()

    total_edits = 0
    total_words = 0
    for gt, pred in zip(gts, preds):
        gt_words = words(gt)
        pred_words = words(pred)
        total_words += max(1, len(gt_words))

        m, n = len(gt_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt_words[i - 1] == pred_words[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        total_edits += dp[m][n]

    return total_edits / total_words if total_words > 0 else 0.0


# -----------------------
# Donut inference
# -----------------------

def run_donut(
    model_name: str,
    ids: List[str],
    img_paths: List[str],
    gts: List[str],
    batch_size: int = 1,
    max_length: int = 128,
    force_cpu: bool = False,
    image_size: int = 768,
) -> Dict[str, Any]:
    """
    Very memory-friendly Donut inference.
    - batch_size default 1
    - downscales image to image_size x image_size
    - fp16 on GPU
    """
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading processor and model: {model_name}")
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Reduce image size (default is quite big -> heavy on memory)
    # We set both sides to image_size, keeping aspect ratio via processor
    try:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            # size can be dict or int; we force dict with height/width
            processor.image_processor.size = {
                "height": image_size,
                "width": image_size,
            }
            print(f"Set Donut image size to {image_size}x{image_size}")
    except Exception as e:
        print("Warning: could not adjust image size:", e)

    model.to(device)

    if device.type == "cuda":
        # fp16 to save memory
        model.half()
        print("Using fp16 on GPU")

    model.eval()

    preds: List[str] = []

    print(f"Using device: {device}")
    print(
        f"Total lines: {len(ids)}, batch_size={batch_size}, max_length={max_length}")

    start_time = time.time()

    for start in range(0, len(ids), batch_size):
        end = min(len(ids), start + batch_size)
        batch_paths = img_paths[start:end]

        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        encodings = processor(images=images, return_tensors="pt")
        pixel_values = encodings.pixel_values.to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=max_length,
                        use_cache=False,  # slightly reduce memory
                    )
            else:
                generated_ids = model.generate(
                    pixel_values,
                    max_length=max_length,
                    use_cache=False,
                )

        batch_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        batch_texts = [t.strip() for t in batch_texts]

        preds.extend(batch_texts)

        print(f"Processed {end}/{len(ids)} lines", end="\r", flush=True)

        # Free intermediate stuff just in case
        del images, encodings, pixel_values, generated_ids, batch_texts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    avg_sec_per_line = total_time / len(ids)
    lines_per_sec = len(ids) / total_time if total_time > 0 else 0.0

    cer = compute_cer(gts, preds)
    wer = compute_wer(gts, preds)

    print()
    print(f"Done inference. CER={cer:.6f}, WER={wer:.6f}")
    print(f"Total time: {total_time:.2f}s  "
          f"({avg_sec_per_line*1000:.2f} ms/line, {lines_per_sec:.2f} lines/s)")

    return {
        "preds": preds,
        "CER": cer,
        "WER": wer,
        "total_seconds": total_time,
        "avg_seconds_per_line": avg_sec_per_line,
        "lines_per_second": lines_per_sec,
    }


def save_outputs(
    out_dir: Path,
    ids: List[str],
    gts: List[str],
    preds: List[str],
    metrics: Dict[str, Any],
    model_name: str,
    manifest_path: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_txt = out_dir / "preds.txt"
    perline_csv = out_dir / "perline.csv"
    metrics_json = out_dir / "metrics.json"

    with preds_txt.open("w", encoding="utf-8", newline="") as f:
        for img_id, pred in zip(ids, preds):
            f.write(f"{img_id}\t{pred}\n")

    with perline_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "gt", "pred", "CER"])
        for img_id, gt, pred in zip(ids, gts, preds):
            cer = _edit_distance(gt, pred) / max(1, len(gt))
            writer.writerow([img_id, gt, pred, f"{cer:.6f}"])

    meta: Dict[str, Any] = {
        "model": model_name,
        "manifest": str(manifest_path),
        "num_lines": len(ids),
        "CER": metrics["CER"],
        "WER": metrics["WER"],
        "total_seconds": metrics["total_seconds"],
        "avg_seconds_per_line": metrics["avg_seconds_per_line"],
        "lines_per_second": metrics["lines_per_second"],
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Outputs written to:")
    print("  ", preds_txt)
    print("  ", perline_csv)
    print("  ", metrics_json)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV")
    ap.add_argument(
        "--model",
        default="sbhavy/donut-base-ocr",
        help="Hugging Face model id for Donut",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (1 recommended for T4)",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max decoder length (tokens)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of lines (for quick tests)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory; default: next to manifest, donut_<model_short>",
    )
    ap.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU even if GPU is available (no CUDA OOM, but slower)",
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=768,
        help="Image size (height/width) fed into Donut (smaller => less memory)",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    ids, img_paths, gts = load_manifest(str(manifest_path), limit=args.limit)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        short_name = args.model.replace("/", "_")
        out_dir = manifest_path.parent / f"donut_{short_name}"

    metrics = run_donut(
        model_name=args.model,
        ids=ids,
        img_paths=img_paths,
        gts=gts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_cpu=args.force_cpu,
        image_size=args.image_size,
    )

    save_outputs(
        out_dir=out_dir,
        ids=ids,
        gts=gts,
        preds=metrics["preds"],
        metrics=metrics,
        model_name=args.model,
        manifest_path=str(manifest_path),
    )


if __name__ == "__main__":
    main()
