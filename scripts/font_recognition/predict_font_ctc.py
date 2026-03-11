#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from model_ctc import build_font_ctc_model, ctc_greedy_decode, smooth_token_sequence
from rec_loader import load_font_vocab


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PredictItem:
    sample_id: str
    image_path: Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run trained font-CTC models on unlabeled images."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest_csv", help="CSV with at least id,image_path columns.")
    src.add_argument("--images_root", help="Folder containing images to score.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", type=str, default=None, choices=["crnn", "transformer"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--image_height", type=int, default=48)
    ap.add_argument("--max_width", type=int, default=1536)
    ap.add_argument("--smooth_window", type=int, default=1)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def _resize_to_height(im: Image.Image, target_h: int, max_w: int) -> Image.Image:
    w, h = im.size
    if h <= 0:
        h = 1
    new_w = max(1, int(round(w * (target_h / float(h)))))
    new_w = min(new_w, max_w)
    return im.resize((new_w, target_h), Image.BICUBIC)


class PredictDataset(Dataset):
    def __init__(self, items: Sequence[PredictItem], image_height: int, max_width: int):
        self.items = list(items)
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        im = Image.open(item.image_path).convert("L")
        im = _resize_to_height(im, self.image_height, self.max_width)
        tensor = pil_to_tensor(im).float() / 255.0
        return {
            "image": tensor,
            "width": tensor.shape[-1],
            "sample_id": item.sample_id,
            "image_path": str(item.image_path),
        }


def collate_predict_batch(batch: Sequence[dict]) -> dict:
    max_w = max(x["image"].shape[-1] for x in batch)
    bsz = len(batch)
    h = batch[0]["image"].shape[-2]
    images = torch.ones((bsz, 1, h, max_w), dtype=torch.float32)
    widths = torch.zeros((bsz,), dtype=torch.long)
    ids: List[str] = []
    image_paths: List[str] = []
    for i, row in enumerate(batch):
        width = row["image"].shape[-1]
        images[i, :, :, :width] = row["image"]
        widths[i] = width
        ids.append(row["sample_id"])
        image_paths.append(row["image_path"])
    return {
        "images": images,
        "input_widths": widths,
        "sample_ids": ids,
        "image_paths": image_paths,
    }


def load_manifest_items(path: Path, limit: int | None) -> List[PredictItem]:
    items: List[PredictItem] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = str(row.get("image_path") or "").strip()
            sample_id = str(row.get("id") or row.get("stem") or "").strip()
            if not sample_id and image_path:
                sample_id = Path(image_path).stem
            if not image_path or not sample_id:
                continue
            items.append(PredictItem(sample_id=sample_id, image_path=Path(image_path)))
            if limit is not None and len(items) >= limit:
                break
    return items


def load_folder_items(root: Path, recursive: bool, limit: int | None) -> List[PredictItem]:
    items: List[PredictItem] = []
    walker = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(walker):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        items.append(PredictItem(sample_id=path.stem, image_path=path))
        if limit is not None and len(items) >= limit:
            break
    return items


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_csv:
        items = load_manifest_items(Path(args.manifest_csv), args.limit)
        source = str(Path(args.manifest_csv).expanduser().resolve())
    else:
        root = Path(args.images_root).expanduser().resolve()
        items = load_folder_items(root, recursive=args.recursive, limit=args.limit)
        source = str(root)

    if not items:
        raise RuntimeError("No images found for font inference.")

    labels, _ = load_font_vocab(args.font_vocab)
    id_to_label = {i + 1: lab for i, lab in enumerate(labels)}
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = args.arch or ckpt.get("arch", "crnn")
    model = build_font_ctc_model(arch, num_labels=len(labels))
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = PredictDataset(items, image_height=args.image_height, max_width=args.max_width)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_predict_batch,
    )

    rows = []
    for batch in dl:
        images = batch["images"].to(device)
        widths = batch["input_widths"].to(device)
        sample_ids = batch["sample_ids"]
        image_paths = batch["image_paths"]

        log_probs = model(images)
        input_lengths = model.output_lengths(widths)
        preds = ctc_greedy_decode(log_probs, input_lengths, id_to_label, blank_id=0)
        if args.smooth_window and args.smooth_window > 1:
            preds = [smooth_token_sequence(p, window=args.smooth_window) for p in preds]

        for sample_id, image_path, pred in zip(sample_ids, image_paths, preds):
            rows.append(
                {
                    "id": sample_id,
                    "image_path": image_path,
                    "pred_fonts": "".join(pred),
                    "pred_len": len(pred),
                }
            )

    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']}\t{row['pred_fonts']}\n")

    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_path", "pred_fonts", "pred_len"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "num_lines": len(rows),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "arch": arch,
        "smooth_window": int(args.smooth_window),
        "source": source,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
