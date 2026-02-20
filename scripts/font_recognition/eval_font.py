#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from font_dataset import AlignJsonlFontDataset, collate_font_batch
from rec_loader import load_rec_model_with_features, extract_rec_features
from train_font import RangePooler, FontHead  # reuse your exact modules


def levenshtein(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # DP (O(min(n,m)) memory)
    if m < n:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # del
                cur[j - 1] + 1,     # ins
                prev[j - 1] + cost  # sub
            )
        prev = cur
    return prev[m]


def rle(seq: List[int]) -> List[int]:
    """Run-length encode (collapse consecutive duplicates)."""
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


@paddle.no_grad()
def eval_on_jsonl(
    rec_model: nn.Layer,
    pooler: RangePooler,
    head: FontHead,
    dl: paddle.io.DataLoader,
    feat_source: str,
    min_range_len: int,
    viterbi_lambda: float,
) -> Tuple[float, float, float, float, int]:
    head.eval()
    pooler.eval()

    total_tok = 0
    correct_tok = 0

    # CER counters
    tok_edits = 0
    tok_len = 0

    seg_edits = 0
    seg_len = 0

    # macro acc
    num_fonts = head.mlp[-1].weight.shape[0]
    per_c_tot = np.zeros((num_fonts,), dtype=np.int64)
    per_c_cor = np.zeros((num_fonts,), dtype=np.int64)

    for x, ranges, y, mask in dl:
        feats = extract_rec_features(rec_model, x)

        if feat_source == "im2seq":
            F_btd = feats["im2seq"] if "im2seq" in feats else feats["ctc_neck"]
        else:
            F_btd = feats["ctc_neck"]

        rlen = (ranges[:, :, 1] - ranges[:, :, 0] + 1).astype("int64")
        mask2 = mask * (rlen >= min_range_len).astype("float32")
        m = (mask2 > 0.5).astype("int64")

        pooled = pooler(F_btd, ranges)
        logits = head(pooled)

        if viterbi_lambda > 1e-8:
            logp = F.log_softmax(logits, axis=-1).numpy()
            lam = float(viterbi_lambda)
            pred_list = []
            for lp, mrow in zip(logp, mask2.numpy()):
                G, K = lp.shape
                valid = (mrow > 0.5)
                if not valid.any():
                    pred_list.append(np.zeros((G,), dtype=np.int64))
                    continue
                dp = np.full((G, K), np.inf, dtype=np.float64)
                back = np.zeros((G, K), dtype=np.int16)
                dp[0] = -lp[0]
                for g in range(1, G):
                    if not valid[g]:
                        dp[g] = dp[g-1]
                        back[g] = np.argmin(dp[g-1])
                        continue
                    prev = dp[g-1][:, None] + lam * (np.arange(K)[None, :] != np.arange(K)[:, None])
                    best_prev = prev.min(axis=0)
                    back[g] = prev.argmin(axis=0)
                    dp[g] = best_prev - lp[g]
                seq = np.zeros((G,), dtype=np.int64)
                seq[-1] = dp[-1].argmin()
                for g in range(G-2, -1, -1):
                    seq[g] = back[g+1, seq[g+1]]
                pred_list.append(seq)
            pred = paddle.to_tensor(np.stack(pred_list, axis=0), place=logits.place)
        else:
            pred = paddle.argmax(logits, axis=-1)  # [B,G]

        # token acc
        eq = ((pred == y).astype("int64") * m)
        total_tok += int(m.sum().item())
        correct_tok += int(eq.sum().item())

        # numpy for CER + macro
        y_np = y.numpy().astype(np.int64)
        p_np = pred.numpy().astype(np.int64)
        m_np = (m.numpy() > 0)

        # macro acc
        for yy, pp, mm in zip(y_np.reshape(-1), p_np.reshape(-1), m_np.reshape(-1)):
            if not mm:
                continue
            per_c_tot[yy] += 1
            if pp == yy:
                per_c_cor[yy] += 1

        # CER per sample
        B = y_np.shape[0]
        for b in range(B):
            gt_seq = y_np[b][m_np[b]].tolist()
            pr_seq = p_np[b][m_np[b]].tolist()
            if len(gt_seq) == 0:
                continue

            # Token CER
            tok_edits += levenshtein(pr_seq, gt_seq)
            tok_len += len(gt_seq)

            # Segment/Group CER (collapsed runs)
            gt_seg = rle(gt_seq)
            pr_seg = rle(pr_seq)
            seg_edits += levenshtein(pr_seg, gt_seg)
            seg_len += len(gt_seg)

    acc = correct_tok / max(1, total_tok)

    accs = []
    for c in range(len(per_c_tot)):
        if per_c_tot[c] > 0:
            accs.append(per_c_cor[c] / per_c_tot[c])
    macro = float(np.mean(accs)) if accs else 0.0

    tok_cer = tok_edits / max(1, tok_len)
    seg_cer = seg_edits / max(1, seg_len)

    return float(acc), float(macro), float(tok_cer), float(seg_cer), int(total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-jsonl", type=Path, required=True)
    ap.add_argument("--font-vocab", type=Path, required=True)

    ap.add_argument("--rec-config", type=str, required=True)
    ap.add_argument("--rec-checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="gpu",
                    choices=["gpu", "cpu"])

    ap.add_argument("--pooling", type=str, default="attnmax",
                    choices=["mean", "max", "meanmax", "attn", "attnmax"])
    ap.add_argument("--context", type=str, default="conv",
                    choices=["none", "conv", "gru"])
    ap.add_argument("--context-hidden", type=int, default=128)
    ap.add_argument("--context-layers", type=int, default=1)
    ap.add_argument("--viterbi-lambda", type=float, default=0.0,
                    help="Inference-time Potts transition penalty; >0 enables Viterbi decoding that penalizes font changes.")
    ap.add_argument("--disable-cudnn-rnn", action="store_true",
                    help="Disable cuDNN RNN kernels (workaround for GRU segfaults on some GPUs).")

    ap.add_argument("--feat-source", type=str, default="im2seq",
                    choices=["im2seq", "ctc_neck"])
    ap.add_argument("--min-range-len", type=int, default=1)
    ap.add_argument("--max-graphemes", type=int, default=120)
    ap.add_argument("--rec-image-shape", type=str, default=None,
                    help="C,H,W for recognizer preprocess. If omitted, auto-read Global.image_shape from rec config.")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--head-ckpt", type=Path, required=True)
    ap.add_argument("--pooler-ckpt", type=Path, required=True)

    args = ap.parse_args()

    if args.context == "gru" and args.disable_cudnn_rnn:
        import os
        os.environ["FLAGS_use_cudnn"] = "0"
        print("[INFO] Set env FLAGS_use_cudnn=0 for GRU; restart the process to ensure cuDNN is off.")

    paddle.set_device("cpu" if args.device == "cpu" else "gpu")

    vocab = json.loads(args.font_vocab.read_text(encoding="utf-8"))
    # quick check: run inside eval before building dataset
    print("num_fonts:", vocab["num_fonts"])
    print("font2id:", vocab["font2id"])
    font2id: Dict[str, int] = vocab["font2id"]
    num_fonts: int = int(vocab["num_fonts"])

    if args.rec_image_shape is not None:
        rec_shape = tuple(int(x.strip()) for x in args.rec_image_shape.split(","))
    else:
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(Path(args.rec_config).read_text(encoding="utf-8"))
            gimg = cfg.get("Global", {}).get("image_shape")
            if isinstance(gimg, (list, tuple)) and len(gimg) == 3:
                rec_shape = tuple(int(x) for x in gimg)
            else:
                raise ValueError
            print(f"[INFO] rec_image_shape auto-set from config: {rec_shape}")
        except Exception:
            rec_shape = (3, 48, 320)
            print("[WARN] Could not read Global.image_shape from rec config; defaulting to (3,48,320). "
                  "Pass --rec-image-shape explicitly if different.")

    # dataset
    ds = AlignJsonlFontDataset(
        align_jsonl=args.align_jsonl,
        font2id=font2id,
        rec_image_shape=rec_shape,
        max_graphemes=args.max_graphemes,
    )
    dl = paddle.io.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_font_batch,
        return_list=True,
    )

    # frozen OCR model
    rec_model = load_rec_model_with_features(
        args.rec_config, args.rec_checkpoint, device=args.device)
    rec_model.eval()
    for p in rec_model.parameters():
        p.stop_gradient = True

    # infer feature dim once
    x0, ranges0, y0, mask0 = next(iter(dl))
    feats0 = extract_rec_features(rec_model, x0)
    if args.feat_source == "im2seq":
        F0 = feats0["im2seq"] if "im2seq" in feats0 else feats0["ctc_neck"]
    else:
        F0 = feats0["ctc_neck"]
    D = int(F0.shape[-1])
    pool_dim = D if args.pooling in ("mean", "max", "attn") else (2 * D)

    pooler = RangePooler(D=D, pooling=args.pooling)
    head = FontHead(
        in_dim=pool_dim,
        hidden=4096,
        num_fonts=num_fonts,
        dropout=0.0,
        context=args.context,
        context_hidden=args.context_hidden,
        context_layers=args.context_layers,
    )

    pooler.set_state_dict(paddle.load(str(args.pooler_ckpt)))
    head.set_state_dict(paddle.load(str(args.head_ckpt)))

    acc, macro, tok_cer, seg_cer, ntok = eval_on_jsonl(
        rec_model=rec_model,
        pooler=pooler,
        head=head,
        dl=dl,
        feat_source=args.feat_source,
        min_range_len=args.min_range_len,
        viterbi_lambda=args.viterbi_lambda,
    )

    print("==== FONT GROUP TEST EVAL ====")
    print(f"tokens: {ntok}")
    print(f"token_acc: {acc:.6f}")
    print(f"macro_acc: {macro:.6f}")
    print(
        f"token_font_CER: {tok_cer:.6f}   (edit distance over per-grapheme font ids)")
    print(
        f"group_font_CER: {seg_cer:.6f}   (edit distance over font RUNS / groups)")


if __name__ == "__main__":
    main()
