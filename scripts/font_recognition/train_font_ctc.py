#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from model_ctc import build_font_ctc_model, ctc_greedy_decode, font_cer
from rec_loader import (
    FontAlignDataset,
    collate_font_batch,
    dominant_font,
    load_align_items_from_csv,
    load_font_vocab,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="configs/train_clean.csv")
    ap.add_argument("--val_csv", default="configs/valid_clean.csv")
    ap.add_argument("--font_vocab", default="configs/font_vocab.json")
    ap.add_argument("--data_root", default="dataset", help="Local dataset root containing train/valid folders.")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="valid")
    ap.add_argument("--out_dir", default="runs/font-ctc-baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--image_height", type=int, default=48)
    ap.add_argument("--max_width", type=int, default=1536)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", type=str, default="crnn", choices=["crnn", "transformer"])
    ap.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint (.pt) to resume training.")
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--no_weighted_sampler", action="store_true")
    ap.add_argument("--save_every_epoch", action="store_true", help="Also save epoch_XXXX.pt each epoch.")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    items,
    stoi: Dict[str, int],
    image_height: int,
    max_width: int,
    batch_size: int,
    num_workers: int,
    weighted_sampler: bool,
):
    ds = FontAlignDataset(items, stoi, image_height=image_height, max_width=max_width)
    if weighted_sampler:
        dom = [dominant_font(x.gt_fonts) for x in items]
        counts: Dict[str, int] = defaultdict(int)
        for d in dom:
            counts[d] += 1
        weights = [1.0 / counts[d] for d in dom]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_font_batch)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_font_batch)


@torch.no_grad()
def evaluate(model, loader, id_to_label, device):
    model.eval()
    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []
    for batch in loader:
        images = batch["images"].to(device)
        widths = batch["input_widths"].to(device)
        log_probs = model(images)
        input_lengths = model.output_lengths(widths)
        preds = ctc_greedy_decode(log_probs, input_lengths, id_to_label, blank_id=0)
        all_preds.extend(preds)
        all_refs.extend(batch["target_tokens"])
    return font_cer(all_preds, all_refs), all_preds, all_refs


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, stoi = load_font_vocab(args.font_vocab)
    id_to_label = {i + 1: lab for i, lab in enumerate(labels)}

    train_items = load_align_items_from_csv(
        args.train_csv,
        stoi,
        data_root=args.data_root,
        split_hint=args.train_split,
        only_ok=True,
        limit=args.train_limit,
    )
    val_items = load_align_items_from_csv(
        args.val_csv,
        stoi,
        data_root=args.data_root,
        split_hint=args.val_split,
        only_ok=True,
        limit=args.val_limit,
    )
    if not train_items:
        raise RuntimeError("No train samples found. Check --data_root and --train_csv.")
    if not val_items:
        raise RuntimeError("No val samples found. Check --data_root and --val_csv.")

    train_loader = build_loader(
        train_items, stoi, args.image_height, args.max_width, args.batch_size, args.num_workers, (not args.no_weighted_sampler)
    )
    val_loader = DataLoader(
        FontAlignDataset(val_items, stoi, image_height=args.image_height, max_width=args.max_width),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_font_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_font_ctc_model(args.arch, num_labels=len(labels)).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_cer = 1e9
    history = []
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_cer = float(ckpt.get("best_val_font_cer", best_val_cer))
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed from {resume_path} | next_epoch={start_epoch} | "
            f"best_val_font_cer={best_val_cer:.6f}"
        )
        if start_epoch > args.epochs:
            print("Resume checkpoint already reached/exceeded --epochs. Nothing to train.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["images"].to(device)
            widths = batch["input_widths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs = model(images)
            input_lengths = model.output_lengths(widths)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bsz = images.shape[0]
            running_loss += loss.item() * bsz
            seen += bsz

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_cer, _, _ = evaluate(model, val_loader, id_to_label, device)

        log = {"epoch": epoch, "train_loss": train_loss, "val_font_cer": val_cer, "lr": scheduler.get_last_lr()[0]}
        history.append(log)
        print(json.dumps(log))

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "labels": labels,
            "arch": args.arch,
            "args": vars(args),
            "best_val_font_cer": best_val_cer,
            "history": history,
        }
        torch.save(last_ckpt, out_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(last_ckpt, out_dir / f"epoch_{epoch:04d}.pt")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "labels": labels,
                "arch": args.arch,
                "args": vars(args),
                "best_val_font_cer": best_val_cer,
                "history": history,
            }
            torch.save(ckpt, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "best_val_font_cer": best_val_cer,
        "num_train_samples": len(train_items),
        "num_val_samples": len(val_items),
        "labels": labels,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
