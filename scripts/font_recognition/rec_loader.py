#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


DEFAULT_FONT_LABELS = ["a", "G", "f", "b", "t", "s", "r", "i"]


def load_font_vocab(vocab_path: Optional[str]) -> Tuple[List[str], Dict[str, int]]:
    if vocab_path is None:
        labels = list(DEFAULT_FONT_LABELS)
    else:
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        labels = data.get("labels", DEFAULT_FONT_LABELS)
    stoi = {lab: i for i, lab in enumerate(labels)}
    return labels, stoi


def parse_font_tokens(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if " " in s:
            return [tok for tok in s.split() if tok]
        return list(s)
    return []


def _candidate_from_content_prefix(path_str: str, data_root: Optional[Path]) -> Optional[Path]:
    if data_root is None:
        return None
    norm = path_str.replace("\\", "/")
    marker = "/dataset/"
    if marker not in norm:
        return None
    rel = norm.split(marker, 1)[1]
    return data_root / Path(rel)


def resolve_dataset_path(
    given_path: str,
    sample_id: str,
    suffix: str,
    data_root: Optional[str] = None,
    split_hint: Optional[str] = None,
) -> Path:
    raw = Path(given_path)
    if raw.exists():
        return raw

    root = Path(data_root) if data_root else None

    cands: List[Path] = []
    from_content = _candidate_from_content_prefix(given_path, root)
    if from_content is not None:
        cands.append(from_content)
    if root is not None and sample_id:
        split = split_hint or ("valid" if "/valid/" in given_path.replace("\\", "/") else None)
        if split:
            cands.append(root / split / f"{sample_id}{suffix}")
        cands.append(root / f"{sample_id}{suffix}")

    for c in cands:
        if c.exists():
            return c

    return cands[0] if cands else raw


@dataclass
class AlignItem:
    sample_id: str
    image_path: Path
    font_path: Path
    gt_fonts: List[str]


def load_align_items_from_csv(
    csv_path: str,
    font_stoi: Dict[str, int],
    data_root: Optional[str] = None,
    split_hint: Optional[str] = None,
    only_ok: bool = True,
    limit: Optional[int] = None,
    dominant_font_filter: Optional[Sequence[str]] = None,
) -> List[AlignItem]:
    items: List[AlignItem] = []
    allowed_dominant = set(dominant_font_filter) if dominant_font_filter else None
    p = Path(csv_path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok_val = str(row.get("ok", "")).strip().upper()
            if only_ok and ok_val not in {"TRUE", "1", "YES", "Y"}:
                continue

            sample_id = str(row.get("id", ""))
            image_path = str(row.get("image_path", "")).strip()
            font_path = str(row.get("font_path", "")).strip()
            if not image_path or not sample_id or not font_path:
                continue

            resolved_img = resolve_dataset_path(image_path, sample_id, ".jpg", data_root, split_hint)
            resolved_font = resolve_dataset_path(font_path, sample_id, ".font", data_root, split_hint)
            if (not resolved_img.exists()) or (not resolved_font.exists()):
                continue

            font_raw = resolved_font.read_text(encoding="utf-8")
            gt_fonts = parse_font_tokens(font_raw)
            if not gt_fonts:
                continue
            if any(tok not in font_stoi for tok in gt_fonts):
                continue
            if allowed_dominant is not None and dominant_font(gt_fonts) not in allowed_dominant:
                continue

            items.append(
                AlignItem(
                    sample_id=sample_id,
                    image_path=resolved_img,
                    font_path=resolved_font,
                    gt_fonts=gt_fonts,
                )
            )
            if limit is not None and len(items) >= limit:
                break
    return items


def _resize_to_height(im: Image.Image, target_h: int, max_w: int) -> Image.Image:
    w, h = im.size
    if h <= 0:
        h = 1
    new_w = max(1, int(round(w * (target_h / float(h)))))
    new_w = min(new_w, max_w)
    return im.resize((new_w, target_h), Image.BICUBIC)


class FontAlignDataset(Dataset):
    def __init__(
        self,
        items: Sequence[AlignItem],
        font_stoi: Dict[str, int],
        image_height: int = 48,
        max_width: int = 1536,
    ):
        self.items = list(items)
        self.font_stoi = font_stoi
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        im = Image.open(it.image_path).convert("L")
        im = _resize_to_height(im, self.image_height, self.max_width)
        t = pil_to_tensor(im).float() / 255.0  # [1, H, W]
        target = torch.tensor([self.font_stoi[x] + 1 for x in it.gt_fonts], dtype=torch.long)
        return {
            "image": t,
            "target_ids": target,
            "target_tokens": it.gt_fonts,
            "sample_id": it.sample_id,
            "image_path": str(it.image_path),
            "width": t.shape[-1],
        }


def collate_font_batch(batch: Sequence[dict]) -> dict:
    max_w = max(x["image"].shape[-1] for x in batch)
    bsz = len(batch)
    h = batch[0]["image"].shape[-2]
    images = torch.ones((bsz, 1, h, max_w), dtype=torch.float32)
    widths = torch.zeros((bsz,), dtype=torch.long)
    targets: List[torch.Tensor] = []
    target_lengths = torch.zeros((bsz,), dtype=torch.long)
    ids: List[str] = []
    img_paths: List[str] = []
    token_refs: List[List[str]] = []

    for i, row in enumerate(batch):
        w = row["image"].shape[-1]
        images[i, :, :, :w] = row["image"]
        widths[i] = w
        t = row["target_ids"]
        targets.append(t)
        target_lengths[i] = t.numel()
        ids.append(row["sample_id"])
        img_paths.append(row["image_path"])
        token_refs.append(row["target_tokens"])

    return {
        "images": images,
        "input_widths": widths,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": target_lengths,
        "sample_ids": ids,
        "image_paths": img_paths,
        "target_tokens": token_refs,
    }


def dominant_font(tokens: Iterable[str]) -> str:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=counts.get)
