#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from model_ctc import (
    build_font_ctc_model,
    ctc_greedy_decode,
    edit_distance,
    font_cer,
    smooth_token_sequence,
)
from rec_loader import FontAlignDataset, collate_font_batch, dominant_font, load_align_items_from_csv, load_font_vocab


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt produced by train_font_ctc.py")
    ap.add_argument("--manifest_csv", default="configs/valid_clean.csv")
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_height", type=int, default=48)
    ap.add_argument("--max_width", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="runs/font-ctc-eval")
    ap.add_argument("--arch", type=str, default=None, choices=["crnn", "transformer"])
    ap.add_argument("--smooth_window", type=int, default=1, help="Odd number; 1 disables smoothing.")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, stoi = load_font_vocab(args.font_vocab)
    id_to_label = {i + 1: lab for i, lab in enumerate(labels)}
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    arch = args.arch or ckpt.get("arch", "crnn")
    model = build_font_ctc_model(arch, num_labels=len(labels))
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    items = load_align_items_from_csv(
        args.manifest_csv,
        stoi,
        data_root=args.data_root,
        split_hint=args.split,
        only_ok=True,
        limit=args.limit,
    )
    ds = FontAlignDataset(items, stoi, image_height=args.image_height, max_width=args.max_width)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_font_batch)

    rows = []
    preds_all: List[List[str]] = []
    refs_all: List[List[str]] = []
    per_dom = defaultdict(lambda: {"edits": 0, "chars": 0, "lines": 0})

    for batch in dl:
        images = batch["images"].to(device)
        widths = batch["input_widths"].to(device)
        refs = batch["target_tokens"]
        sample_ids = batch["sample_ids"]

        log_probs = model(images)
        input_lengths = model.output_lengths(widths)
        preds = ctc_greedy_decode(log_probs, input_lengths, id_to_label, blank_id=0)
        if args.smooth_window and args.smooth_window > 1:
            preds = [smooth_token_sequence(p, window=args.smooth_window) for p in preds]

        for sid, p, r in zip(sample_ids, preds, refs):
            e = edit_distance(r, p)
            cer = e / max(1, len(r))
            rows.append(
                {
                    "id": sid,
                    "gt_fonts": "".join(r),
                    "pred_fonts": "".join(p),
                    "edits": e,
                    "gt_len": len(r),
                    "font_cer": cer,
                }
            )
            dom = dominant_font(r)
            per_dom[dom]["edits"] += e
            per_dom[dom]["chars"] += max(1, len(r))
            per_dom[dom]["lines"] += 1
        preds_all.extend(preds)
        refs_all.extend(refs)

    overall_cer = font_cer(preds_all, refs_all)
    per_dom_out: Dict[str, dict] = {}
    for k, v in sorted(per_dom.items()):
        per_dom_out[k] = {
            "lines": v["lines"],
            "font_cer": v["edits"] / max(1, v["chars"]),
            "chars": v["chars"],
        }

    metrics = {
        "font_cer": overall_cer,
        "num_lines": len(rows),
        "checkpoint": str(Path(args.checkpoint)),
        "manifest_csv": str(Path(args.manifest_csv)),
        "arch": arch,
        "smooth_window": int(args.smooth_window),
        "per_dominant_font": per_dom_out,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gt_fonts", "pred_fonts", "edits", "gt_len", "font_cer"])
        w.writeheader()
        w.writerows(rows)

    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['id']}\t{r['pred_fonts']}\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
