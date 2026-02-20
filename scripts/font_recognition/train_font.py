#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# :contentReference[oaicite:5]{index=5}
from font_dataset import AlignJsonlFontDataset, collate_font_batch
# :contentReference[oaicite:6]{index=6}
from rec_loader import load_rec_model_with_features, extract_rec_features


# -----------------------
# Utilities: Levenshtein + RLE (for CER)
# -----------------------
def levenshtein(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    if m < n:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def rle(seq: List[int]) -> List[int]:
    """Collapse consecutive duplicates."""
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


# -----------------------
# Small wrapper for stable oversampling
# -----------------------
class IndexDataset(paddle.io.Dataset):
    def __init__(self, base: paddle.io.Dataset, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def build_contains_oversampled_indices(
    ds: AlignJsonlFontDataset,
    boost_ids: List[int],
    target_mult: float = 2.0,
    seed: int = 42,
) -> List[int]:
    rng = np.random.default_rng(seed)
    base_idx = np.arange(len(ds), dtype=np.int64)

    boosted = []
    for i in range(len(ds)):
        _, _, y = ds[i]
        y_arr = np.asarray(y).astype(np.int64)
        if np.isin(y_arr, boost_ids).any():
            boosted.append(i)

    boosted = np.asarray(boosted, dtype=np.int64)
    if len(boosted) == 0:
        print(
            "[WARN] contains-oversample: no boosted samples found. Using base indices.")
        return base_idx.tolist()

    extra = int(max(0.0, (target_mult - 1.0)) * len(boosted))
    extra_idx = rng.choice(boosted, size=extra, replace=True)

    idx_list = np.concatenate([base_idx, extra_idx]).tolist()
    rng.shuffle(idx_list)

    print(
        f"[INFO] contains-oversample: boosted_unique={len(boosted)}/{len(ds)} "
        f"target_mult={target_mult} extra={len(extra_idx)} total_indices={len(idx_list)} boost_ids={boost_ids}"
    )
    return idx_list


def build_transition_oversampled_indices(
    ds: AlignJsonlFontDataset,
    target_mult: float = 2.0,
    min_transitions: int = 1,
    seed: int = 42,
) -> List[int]:
    """
    Oversample examples that have font changes (multi-font / transitions).
    This is what your current training is missing most.
    """
    rng = np.random.default_rng(seed)
    base_idx = np.arange(len(ds), dtype=np.int64)

    boosted = []
    for i in range(len(ds)):
        _, _, y = ds[i]
        y_arr = np.asarray(y).astype(np.int64)
        if y_arr.size <= 1:
            continue
        transitions = int(np.sum(y_arr[1:] != y_arr[:-1]))
        if transitions >= min_transitions:
            boosted.append(i)

    boosted = np.asarray(boosted, dtype=np.int64)
    if len(boosted) == 0:
        print(
            "[WARN] transition-oversample: no boosted samples found. Using base indices.")
        return base_idx.tolist()

    extra = int(max(0.0, (target_mult - 1.0)) * len(boosted))
    extra_idx = rng.choice(boosted, size=extra, replace=True)

    idx_list = np.concatenate([base_idx, extra_idx]).tolist()
    rng.shuffle(idx_list)

    print(
        f"[INFO] transition-oversample: boosted_unique={len(boosted)}/{len(ds)} "
        f"target_mult={target_mult} extra={len(extra_idx)} total_indices={len(idx_list)} "
        f"min_transitions={min_transitions}"
    )
    return idx_list


# -----------------------
# Token-level class counts from align jsonl
# -----------------------
def token_counts_from_align_jsonl(
    align_jsonl: Path,
    font2id: Dict[str, int],
    max_graphemes: int,
) -> np.ndarray:
    K = len(font2id)
    counts = np.zeros((K,), dtype=np.int64)

    with align_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok_align", False):
                continue
            gt_fonts = row.get("gt_fonts")
            if not isinstance(gt_fonts, list):
                continue

            gt_fonts = gt_fonts[:max_graphemes]
            for name in gt_fonts:
                if name is None:
                    continue
                name = str(name)
                if name not in font2id:
                    continue
                counts[font2id[name]] += 1
    return counts


def class_weights_from_token_counts(token_counts: np.ndarray) -> paddle.Tensor:
    c = token_counts.astype(np.float32)
    c = np.maximum(c, 1.0)
    max_c = np.max(c)
    w = np.sqrt(max_c / c)  # inverse-sqrt
    w = w / np.mean(w)
    w = np.clip(w, 0.5, 5.0)
    return paddle.to_tensor(w, dtype="float32")


# -----------------------
# Boundary loss (aligns with group CER by punishing extra font flips)
# -----------------------
def boundary_loss_from_logits(
    logits_bgk: paddle.Tensor,
    y_bg: paddle.Tensor,
    m_bg: paddle.Tensor,
    temp: float = 1.0,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    probs = F.softmax(logits_bgk / max(1e-6, float(temp)), axis=-1)
    sim = paddle.sum(probs[:, :-1, :] * probs[:, 1:, :], axis=-1)  # [B,G-1]
    p_b = 1.0 - sim  # boundary prob ~ dissimilarity

    b_gt = (y_bg[:, :-1] != y_bg[:, 1:]).astype("float32")
    mb = (m_bg[:, :-1] * m_bg[:, 1:]).astype("float32")

    eps = 1e-6
    p_b = paddle.clip(p_b, eps, 1.0 - eps)
    bce = -(b_gt * paddle.log(p_b) + (1.0 - b_gt) * paddle.log(1.0 - p_b))
    loss = paddle.sum(bce * mb) / (paddle.sum(mb) + eps)

    pred_trans = paddle.sum(p_b * mb, axis=-1) / (paddle.sum(mb, axis=-1) + eps)
    gt_trans = paddle.sum(b_gt * mb, axis=-1) / (paddle.sum(mb, axis=-1) + eps)
    return loss, paddle.mean(pred_trans), paddle.mean(gt_trans)


def neighbor_tv_loss(
    logits_bgk: paddle.Tensor,
    y_bg: paddle.Tensor,
    m_bg: paddle.Tensor,
) -> paddle.Tensor:
    """
    Total-variation style loss over probability distributions.
    Encourages consecutive tokens with the same GT font (and valid masks)
    to share similar distributions, reducing spurious font flips.
    """
    if logits_bgk.shape[1] <= 1:
        return paddle.to_tensor(0.0, dtype=logits_bgk.dtype)

    prob = F.softmax(logits_bgk, axis=-1)  # [B,G,K]
    p1 = prob[:, :-1, :]
    p2 = prob[:, 1:, :]

    same = ((y_bg[:, :-1] == y_bg[:, 1:]).astype("float32")
            * m_bg[:, :-1] * m_bg[:, 1:])  # [B,G-1]
    if float(paddle.sum(same)) < 1e-6:
        return paddle.to_tensor(0.0, dtype=logits_bgk.dtype)

    diff = paddle.abs(p1 - p2).sum(axis=-1)  # [B,G-1]
    tv = paddle.sum(diff * same) / (paddle.sum(same) + 1e-6)
    return tv


# -----------------------
# Range Pooler (learnable attention pooling)
# -----------------------
class RangePooler(nn.Layer):
    def __init__(self, D: int, pooling: str):
        super().__init__()
        assert pooling in ("mean", "max", "meanmax", "attn", "attnmax")
        self.pooling = pooling
        self.D = D
        if pooling in ("attn", "attnmax"):
            self.attn_w = self.create_parameter(
                [D], default_initializer=nn.initializer.Normal(std=0.02))

    def forward(self, F_btd: paddle.Tensor, ranges_bG2: paddle.Tensor) -> paddle.Tensor:
        B, T, D = F_btd.shape
        _, G, _ = ranges_bG2.shape

        pooled_rows = []
        for b in range(B):
            Fb = F_btd[b]       # [T,D]
            rb = ranges_bG2[b]  # [G,2]
            feats = []
            for i in range(G):
                t0 = int(rb[i, 0].item())
                t1 = int(rb[i, 1].item())

                t0 = max(0, min(T - 1, t0))
                t1 = max(0, min(T - 1, t1))
                if t1 < t0:
                    t0, t1 = t1, t0

                seg = Fb[t0: t1 + 1, :]  # [L,D]
                if seg.shape[0] == 0:
                    seg = Fb[t0: t0 + 1, :]

                if self.pooling == "mean":
                    v = paddle.mean(seg, axis=0)
                elif self.pooling == "max":
                    v = paddle.max(seg, axis=0)
                elif self.pooling == "meanmax":
                    v = paddle.concat(
                        [paddle.mean(seg, axis=0), paddle.max(seg, axis=0)], axis=0)
                elif self.pooling in ("attn", "attnmax"):
                    scores = paddle.matmul(
                        seg, self.attn_w.unsqueeze(-1)).squeeze(-1)  # [L]
                    alpha = F.softmax(scores, axis=0)
                    attn_v = paddle.sum(
                        seg * alpha.unsqueeze(-1), axis=0)  # [D]
                    if self.pooling == "attn":
                        v = attn_v
                    else:
                        max_v = paddle.max(seg, axis=0)
                        v = paddle.concat([attn_v, max_v], axis=0)  # [2D]
                else:
                    raise ValueError(self.pooling)

                feats.append(v)

            pooled_rows.append(paddle.stack(feats, axis=0))  # [G,?]

        return paddle.stack(pooled_rows, axis=0)  # [B,G,?]


# -----------------------
# Head (stable local context + MLP)
# -----------------------
class FontHead(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_fonts: int,
        dropout: float,
        context: str = "none",
        context_hidden: int = 128,
        context_layers: int = 1,
    ):
        super().__init__()
        assert context in ("none", "conv", "gru")
        self.context = context
        self.ln = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)

        if context == "conv":
            self.dw = nn.Conv1D(
                in_channels=in_dim,
                out_channels=in_dim,
                kernel_size=3,
                padding=1,
                groups=in_dim,
            )
            self.pw = nn.Conv1D(in_channels=in_dim,
                                out_channels=in_dim, kernel_size=1)
            self.conv_act = nn.GELU()

        elif context == "gru":
            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=context_hidden,
                num_layers=context_layers,
                direction="bidirect",
                dropout=0.0 if context_layers == 1 else dropout,
            )
            in_dim = context_hidden * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_fonts),
        )

    def forward(self, x_bgd: paddle.Tensor) -> paddle.Tensor:
        x = self.ln(x_bgd)
        x = self.drop(x)

        if self.context == "conv":
            x = paddle.transpose(x, [0, 2, 1])         # [B,D,G]
            x = self.pw(self.conv_act(self.dw(x)))     # [B,D,G]
            x = paddle.transpose(x, [0, 2, 1])         # [B,G,D]
        elif self.context == "gru":
            x_gbd = paddle.transpose(x, [1, 0, 2])     # [G,B,D]
            y_gbd, _ = self.gru(x_gbd)
            x = paddle.transpose(y_gbd, [1, 0, 2])     # [B,G,D]

        return self.mlp(x)


# -----------------------
# Loss (focal + smoothing + class weights + sample weights)
# -----------------------
def font_token_loss(
    logits: paddle.Tensor,                  # (N, K)
    y: paddle.Tensor,                       # (N,)
    class_weight: Optional[paddle.Tensor],  # (K,)
    focal_gamma: float,
    label_smoothing: float,
    sample_weight: Optional[paddle.Tensor] = None,  # (N,)
) -> paddle.Tensor:
    logp = F.log_softmax(logits, axis=-1)
    nll = -paddle.take_along_axis(logp, y.unsqueeze(-1),
                                  axis=-1).squeeze(-1)  # (N,)

    if label_smoothing and label_smoothing > 0:
        uniform_ce = -paddle.mean(logp, axis=-1)
        nll = (1.0 - label_smoothing) * nll + label_smoothing * uniform_ce

    if class_weight is not None:
        nll = nll * paddle.gather(class_weight, y)

    if focal_gamma and focal_gamma > 0:
        pt = paddle.exp(-nll)
        nll = ((1.0 - pt) ** focal_gamma) * nll

    if sample_weight is not None:
        nll = nll * sample_weight

    return paddle.mean(nll)


# -----------------------
# LR schedule: warmup + cosine
# -----------------------
def make_warmup_cosine_lr(base_lr: float, total_steps: int, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr = base_lr * min_lr_ratio

    cosine = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr,
        T_max=total_steps,
        eta_min=min_lr,
    )

    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=cosine,
        warmup_steps=warmup_steps,
        start_lr=base_lr * 0.1,
        end_lr=base_lr,
    )
    return lr


# -----------------------
# Validation
# -----------------------
@paddle.no_grad()
def eval_metrics(
    rec_model: nn.Layer,
    pooler: RangePooler,
    head: nn.Layer,
    dl: paddle.io.DataLoader,
    feat_source: str,
    min_range_len: int,
    boundary_weight: float,
    viterbi_lambda: float = 0.0,
    debug_batches: int = 0,
    num_fonts: int = 0,
) -> Tuple[float, float, float, float, int]:
    head.eval()
    pooler.eval()

    total = 0
    correct = 0

    per_c_tot = np.zeros(
        (num_fonts,), dtype=np.int64) if num_fonts > 0 else None
    per_c_cor = np.zeros(
        (num_fonts,), dtype=np.int64) if num_fonts > 0 else None

    dbg_true = Counter()
    dbg_pred = Counter()
    dbg_taken = 0

    tok_edits = 0
    tok_len = 0
    seg_edits = 0
    seg_len = 0

    for x, ranges, y, mask in dl:
        feats = extract_rec_features(rec_model, x)

        if feat_source == "im2seq":
            if "im2seq" not in feats:
                F_btd = feats["ctc_neck"]
            else:
                F_btd = feats["im2seq"]
        else:
            F_btd = feats["ctc_neck"]

        # Range-length mask
        rlen = (ranges[:, :, 1] - ranges[:, :, 0] + 1).astype("int64")
        mask2 = mask * (rlen >= min_range_len).astype("float32")

        pooled = pooler(F_btd, ranges)
        logits = head(pooled)

        if viterbi_lambda > 1e-8:
            # Decode each sample with Potts-model Viterbi to penalize transitions
            logp = F.log_softmax(logits, axis=-1).numpy()  # [B,G,K]
            pred_list = []
            lam = float(viterbi_lambda)
            for lp, mrow in zip(logp, mask2.numpy()):
                G, K = lp.shape
                valid = (mrow > 0.5)
                T = lp  # [G,K]
                if not valid.any():
                    pred_list.append(np.zeros((G,), dtype=np.int64))
                    continue
                # DP
                dp = np.full((G, K), np.inf, dtype=np.float64)
                back = np.zeros((G, K), dtype=np.int16)
                dp[0] = -T[0]
                for g in range(1, G):
                    if not valid[g]:
                        dp[g] = dp[g-1]  # copy
                        back[g] = np.argmin(dp[g-1])
                        continue
                    prev = dp[g-1][:, None] + lam * (np.arange(K)[None, :] != np.arange(K)[:, None])
                    best_prev = prev.min(axis=0)
                    back[g] = prev.argmin(axis=0)
                    dp[g] = best_prev - T[g]
                seq = np.zeros((G,), dtype=np.int64)
                seq[-1] = dp[-1].argmin()
                for g in range(G-2, -1, -1):
                    seq[g] = back[g+1, seq[g+1]]
                pred_list.append(seq)
            pred = paddle.to_tensor(np.stack(pred_list, axis=0), place=logits.place)
        else:
            pred = paddle.argmax(logits, axis=-1)

        m = (mask2 > 0.5).astype("int64")
        eq = (pred == y).astype("int64") * m

        y_np = y.numpy().astype(np.int64)
        p_np = pred.numpy().astype(np.int64)
        m_np = (m.numpy() > 0)

        total += int(m.sum().item())
        correct += int(eq.sum().item())

        if per_c_tot is not None:
            for yy, pp, mm in zip(y_np.reshape(-1), p_np.reshape(-1), m_np.reshape(-1)):
                if not mm:
                    continue
                per_c_tot[yy] += 1
                if pp == yy:
                    per_c_cor[yy] += 1

        # CER
        B = y_np.shape[0]
        for b in range(B):
            gt_seq = y_np[b][m_np[b]].tolist()
            pr_seq = p_np[b][m_np[b]].tolist()
            if not gt_seq:
                continue
            tok_edits += levenshtein(pr_seq, gt_seq)
            tok_len += len(gt_seq)
            gt_seg = rle(gt_seq)
            pr_seg = rle(pr_seq)
            seg_edits += levenshtein(pr_seg, gt_seg)
            seg_len += len(gt_seg)

        if debug_batches and dbg_taken < debug_batches:
            yt = (y * m).numpy().reshape([-1]).tolist()
            yp = (pred * m).numpy().reshape([-1]).tolist()
            dbg_true.update(yt)
            dbg_pred.update(yp)
            dbg_taken += 1

    if debug_batches:
        print(f"[DBG] val dist aggregated over {dbg_taken} batches")
        print("[DBG] val true:", dict(dbg_true))
        print("[DBG] val pred:", dict(dbg_pred))

    acc = correct / max(1, total)

    macro = 0.0
    if per_c_tot is not None:
        accs = []
        for c in range(num_fonts):
            if per_c_tot[c] > 0:
                accs.append(per_c_cor[c] / per_c_tot[c])
        macro = float(np.mean(accs)) if accs else 0.0

    tok_cer = tok_edits / max(1, tok_len)
    grp_cer = seg_edits / max(1, seg_len)

    return float(acc), float(macro), float(tok_cer), float(grp_cer), total


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-align-jsonl", type=Path, required=True)
    ap.add_argument("--val-align-jsonl", type=Path, default=None)
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
    ap.add_argument("--disable-cudnn-rnn", action="store_true",
                    help="Disable cuDNN RNN kernels (useful if GRU segfaults on some GPUs like Colab T4).")

    # NEW: feature source
    ap.add_argument("--feat-source", type=str, default="im2seq", choices=["im2seq", "ctc_neck"],
                    help="Font should train on im2seq (pre-language) for best accuracy.")

    # NEW: boundary + range filtering
    ap.add_argument("--min-range-len", type=int, default=1,
                    help="Drop tokens whose t_range length < this (reduces boundary/noise). "
                         "Use 1 for most alignments where t-ranges are 1–2 steps.")
    ap.add_argument("--boundary-weight", type=float, default=0.20,
                    help="Weight for boundary loss (penalizes spurious transitions).")
    ap.add_argument("--boundary-downweight", type=float, default=1.0,
                    help="Optional: down-weight tokens at boundaries (<1.0 reduces boundary tokens).")
    ap.add_argument("--transition-temp", type=float, default=1.0,
                    help="Softmax temperature when computing transition probabilities.")
    ap.add_argument("--print-tran-stats", action="store_true",
                    help="Print predicted vs GT transition rate during train.")
    ap.add_argument("--tv-loss-weight", type=float, default=0.0,
                    help="Neighbor consistency loss weight (TV over probs when GT labels match).")
    ap.add_argument("--viterbi-lambda", type=float, default=0.0,
                    help="Inference-time Potts transition penalty; >0 enables Viterbi decoding that penalizes font changes.")

    # training
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.10)

    ap.add_argument("--focal-gamma", type=float, default=1.1)
    ap.add_argument("--label-smoothing", type=float, default=0.01)

    ap.add_argument("--rec-image-shape", type=str, default=None,
                    help="C,H,W for recognizer preprocess. If omitted, auto-read from rec config (Global.image_shape).")
    ap.add_argument("--max-graphemes", type=int, default=120)

    # oversampling
    ap.add_argument("--oversample", type=str, default="transitions",
                    choices=["none", "contains", "transitions"])
    ap.add_argument("--oversample-ids", type=str, default="",
                    help="Comma-separated font IDs for 'contains' oversample.")
    ap.add_argument("--oversample-mult", type=float, default=2.0)
    ap.add_argument("--min-transitions", type=int, default=1,
                    help="For oversample=transitions, minimum transitions per sample.")

    # stability / speed
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--use-shared-memory", action="store_true")

    # debug
    ap.add_argument("--debug-val-batches", type=int, default=10)
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="If >0, run eval_metrics every N steps (uses full val_dl, costly).")
    ap.add_argument("--eval-every-epoch", action="store_true",
                    help="Run an additional eval_metrics pass at end of each epoch (separate from best-saving eval).")

    # checkpoints
    ap.add_argument("--init-head", type=Path, default=None)
    ap.add_argument("--save-best", action="store_true")
    ap.add_argument("--main-metric", type=str, default="token_cer",
                    choices=["acc", "token_cer", "group_cer"])
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="Resume training from a checkpoint directory (expects train_state.pd + model files).")
    ap.add_argument("--resume", action="store_true",
                    help="Auto-resume from <out_dir>/train_state.pd if it exists (ignored when --resume-from is set).")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/font_v3"))

    args = ap.parse_args()

    if args.context == "gru" and args.disable_cudnn_rnn:
        # cuDNN RNN kernels can segfault on some Colab GPUs (e.g., T4).
        # Use env var only to avoid ValueError in some Paddle builds.
        import os
        os.environ["FLAGS_use_cudnn"] = "0"
        print("[INFO] Set env FLAGS_use_cudnn=0 for GRU; restart the process to guarantee cuDNN is disabled.")

    paddle.set_device("cpu" if args.device == "cpu" else "gpu")

    vocab = json.loads(args.font_vocab.read_text(encoding="utf-8"))
    font2id: Dict[str, int] = vocab["font2id"]
    num_fonts: int = int(vocab["num_fonts"])

    print(f"[INFO] num_fonts={num_fonts} pooling={args.pooling} context={args.context} "
          f"feat_source={args.feat_source} oversample={args.oversample} main_metric={args.main_metric} "
          f"boundary_weight={args.boundary_weight} boundary_downweight={args.boundary_downweight}")

    # Resolve recognizer image shape
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
            # fallback to common PP-OCRv5 height
            rec_shape = (3, 48, 320)
            print("[WARN] Could not read Global.image_shape from rec config; defaulting to (3,48,320). "
                  "Pass --rec-image-shape explicitly to override.")

    base_train_ds = AlignJsonlFontDataset(
        align_jsonl=args.train_align_jsonl,
        font2id=font2id,
        rec_image_shape=rec_shape,
        max_graphemes=args.max_graphemes,
    )

    train_ds = base_train_ds
    if args.oversample == "contains" and args.oversample_ids.strip():
        boost_ids = [int(x.strip())
                     for x in args.oversample_ids.split(",") if x.strip()]
        idx_list = build_contains_oversampled_indices(
            base_train_ds, boost_ids=boost_ids, target_mult=args.oversample_mult, seed=42
        )
        train_ds = IndexDataset(base_train_ds, idx_list)
    elif args.oversample == "transitions":
        idx_list = build_transition_oversampled_indices(
            base_train_ds, target_mult=args.oversample_mult, min_transitions=args.min_transitions, seed=42
        )
        train_ds = IndexDataset(base_train_ds, idx_list)

    train_dl = paddle.io.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_font_batch,
        drop_last=True,
        num_workers=args.num_workers,
        use_shared_memory=args.use_shared_memory,
    )

    val_dl = None
    if args.val_align_jsonl is not None:
        val_ds = AlignJsonlFontDataset(
            align_jsonl=args.val_align_jsonl,
            font2id=font2id,
            rec_image_shape=rec_shape,
            max_graphemes=args.max_graphemes,
        )
        val_dl = paddle.io.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_font_batch,
            drop_last=False,
            num_workers=args.num_workers,
            use_shared_memory=args.use_shared_memory,
        )

    # Load frozen OCR model
    rec_model = load_rec_model_with_features(
        config_path=args.rec_config,
        checkpoint_path=args.rec_checkpoint,
        device=args.device,
    )
    rec_model.eval()
    for p in rec_model.parameters():
        p.stop_gradient = True

    # Infer encoder dim from chosen feature
    x0, ranges0, y0, mask0 = next(iter(train_dl))
    with paddle.no_grad():
        feats0 = extract_rec_features(rec_model, x0)
        if args.feat_source == "im2seq" and "im2seq" in feats0:
            F0 = feats0["im2seq"]
        else:
            F0 = feats0["ctc_neck"]
    if len(F0.shape) != 3:
        raise RuntimeError(f"Expected [B,T,D] feature, got shape={F0.shape}. "
                           f"Check rec_loader.extract_rec_features().")
    D = int(F0.shape[-1])

    pooler = RangePooler(D=D, pooling=args.pooling)
    pool_dim = D if args.pooling in ("mean", "max", "attn") else (2 * D)
    print(f"[INFO] encoder feature dim D={D}, pooled dim={pool_dim}")

    head = FontHead(
        in_dim=pool_dim,
        hidden=args.hidden,
        num_fonts=num_fonts,
        dropout=args.dropout,
        context=args.context,
        context_hidden=args.context_hidden,
        context_layers=args.context_layers,
    )

    if args.init_head is not None and args.init_head.exists():
        st = paddle.load(str(args.init_head))
        head.set_state_dict(st)
        print(f"[OK] initialized head from {args.init_head}")

    # Class weights from token counts
    tok_counts = token_counts_from_align_jsonl(
        args.train_align_jsonl, font2id, max_graphemes=args.max_graphemes)
    cw = class_weights_from_token_counts(tok_counts)
    print("[INFO] token counts (by id):", tok_counts.tolist())
    print("[INFO] class weights (by id):", [float(x)
          for x in cw.numpy().tolist()])

    # LR schedule
    steps_per_epoch = len(train_dl)
    total_steps = max(1, args.epochs * steps_per_epoch)
    lr_sched = make_warmup_cosine_lr(
        args.lr, total_steps=total_steps, warmup_ratio=0.05, min_lr_ratio=0.1)

    params = list(pooler.parameters()) + list(head.parameters())
    opt = paddle.optimizer.AdamW(
        learning_rate=lr_sched,
        parameters=params,
        weight_decay=args.weight_decay,
        grad_clip=paddle.nn.ClipGradByNorm(
            args.grad_clip) if args.grad_clip and args.grad_clip > 0 else None,
    )

    scaler = paddle.amp.GradScaler(
        init_loss_scaling=1024) if args.amp else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Convenience: --resume uses <out_dir>/train_state.pd unless an explicit path was provided
    if args.resume and args.resume_from is None:
        cand = args.out_dir / "train_state.pd"
        if cand.exists():
            args.resume_from = cand
            print(f"[INFO] --resume enabled, using checkpoint: {cand}")
        else:
            raise FileNotFoundError(
                f"--resume was set but no train_state.pd found in {args.out_dir}")
    if args.main_metric == "acc":
        best_score = -1.0
    else:
        best_score = float("inf")
    best_path = args.out_dir / "font_head_best.pdparams"
    best_pool = args.out_dir / "pooler_best.pdparams"
    state_path = args.out_dir / "train_state.pd"

    global_step = 0
    start_epoch = 1

    # Resume support
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume-from {args.resume_from} does not exist")
        # expect train_state.pd in resume dir if a directory is given
        if args.resume_from.is_dir():
            load_dir = args.resume_from
            state_ckpt = load_dir / "train_state.pd"
        else:
            load_dir = args.resume_from.parent
            state_ckpt = args.resume_from

        # If user didn't override out_dir (still default) but is resuming from a different folder,
        # keep outputs inside the checkpoint directory to avoid scattering checkpoints.
        if args.out_dir == Path("runs/font_v3") and args.out_dir.resolve() != load_dir.resolve():
            print(f"[INFO] Switching out_dir to resume directory: {load_dir}")
            args.out_dir = load_dir
            state_path = args.out_dir / "train_state.pd"
            best_path = args.out_dir / "font_head_best.pdparams"
            best_pool = args.out_dir / "pooler_best.pdparams"

        if not state_ckpt.exists():
            raise FileNotFoundError(f"Cannot find state checkpoint at {state_ckpt}")

        st = paddle.load(str(state_ckpt))
        head_state = paddle.load(str(load_dir / "font_head_last.pdparams"))
        pool_state = paddle.load(str(load_dir / "pooler_last.pdparams"))
        head.set_state_dict(head_state)
        pooler.set_state_dict(pool_state)

        opt.set_state_dict(st["opt"])
        lr_sched.set_state_dict(st["lr_sched"])
        if scaler is not None and st.get("scaler") is not None:
            try:
                scaler.load_state_dict(st["scaler"])
            except Exception as e1:
                try:
                    scaler.set_state_dict(st["scaler"])
                except Exception as e2:
                    print(f"[WARN] GradScaler restore skipped (load/set failed): {e1} / {e2}")

        global_step = int(st.get("global_step", 0))
        best_score = float(st.get("best_score", best_score))
        start_epoch = int(st.get("epoch", 1)) + 1
        print(f"[OK] resumed from {state_ckpt}, start_epoch={start_epoch}, global_step={global_step}, best_score={best_score}")

    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        pooler.train()

        running_loss = 0.0
        running_tokens = 0
        epoch_kept_tokens = 0
        boundary_losses = []
        tv_losses = []
        pred_tr_list = []
        gt_tr_list = []

        for x, ranges, y, mask in train_dl:
            global_step += 1

            # 1) Frozen OCR forward in no_grad
            with paddle.no_grad():
                feats = extract_rec_features(rec_model, x)
                if args.feat_source == "im2seq" and "im2seq" in feats:
                    F_btd = feats["im2seq"]
                else:
                    F_btd = feats["ctc_neck"]

                if len(F_btd.shape) != 3:
                    raise RuntimeError(f"Feature must be [B,T,D], got shape={F_btd.shape}")

                # Noise reduction: filter short ranges
                rlen = (ranges[:, :, 1] - ranges[:, :, 0] + 1).astype("int64")  # [B,G]
                mask2 = mask * (rlen >= args.min_range_len).astype("float32")

                # Optional boundary down-weighting (sample weights only)
                yw = y.astype("int64")
                B, G = yw.shape
                w = paddle.ones([B, G], dtype="float32")
                if G >= 2 and args.boundary_downweight < 0.999:
                    left_diff = (yw[:, 1:] != yw[:, :-1]).astype("float32")  # [B,G-1]
                    boundary = paddle.zeros([B, G], dtype="float32")
                    boundary[:, 1:] = paddle.maximum(boundary[:, 1:], left_diff)
                    boundary[:, :-1] = paddle.maximum(boundary[:, :-1], left_diff)
                    dw = float(args.boundary_downweight)
                    w = w * (1.0 - boundary) + w * boundary * dw

            # 2) IMPORTANT: pooler/head must run WITH grads (NO no_grad here)
            pooled = pooler(F_btd, ranges)   # [B,G,pool_dim]
            logits = head(pooled)            # [B,G,K]

            B, G, K = logits.shape
            logits2 = logits.reshape([B * G, K])
            y2 = y.reshape([B * G])
            m2 = mask2.reshape([B * G])
            w2 = w.reshape([B * G])

            # how many tokens survive masking for train stability diagnostics
            idx = paddle.nonzero(m2 > 0.5).reshape([-1])
            if idx.shape[0] == 0:
                lr_sched.step()
                continue

            logits_v = paddle.gather(logits2, idx)
            y_v = paddle.gather(y2, idx)
            sw_v = paddle.gather(w2, idx)

            if args.amp:
                with paddle.amp.auto_cast():
                    loss = font_token_loss(
                        logits_v, y_v,
                        class_weight=cw,
                        focal_gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        sample_weight=sw_v,
                    )
                    bnd_loss, pred_tr, gt_tr = boundary_loss_from_logits(
                        logits, y, mask2, temp=args.transition_temp)
                    loss = loss + float(args.boundary_weight) * bnd_loss
                    if args.tv_loss_weight > 1e-8:
                        tv_loss = neighbor_tv_loss(logits, y, mask2)
                        loss = loss + float(args.tv_loss_weight) * tv_loss
                    else:
                        tv_loss = paddle.to_tensor(0.0)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.clear_grad()
            else:
                loss = font_token_loss(
                    logits_v, y_v,
                    class_weight=cw,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    sample_weight=sw_v,
                )
                bnd_loss, pred_tr, gt_tr = boundary_loss_from_logits(
                    logits, y, mask2, temp=args.transition_temp)
                loss = loss + float(args.boundary_weight) * bnd_loss
                if args.tv_loss_weight > 1e-8:
                    tv_loss = neighbor_tv_loss(logits, y, mask2)
                    loss = loss + float(args.tv_loss_weight) * tv_loss
                else:
                    tv_loss = paddle.to_tensor(0.0)

                loss.backward()
                opt.step()
                opt.clear_grad()

            lr_sched.step()

            n_tok = int(idx.shape[0])
            running_loss += float(loss.item()) * n_tok
            running_tokens += n_tok
            epoch_kept_tokens += n_tok
            boundary_losses.append(float(bnd_loss.item()))
            tv_losses.append(float(tv_loss.item()))
            pred_tr_list.append(float(pred_tr.item()))
            gt_tr_list.append(float(gt_tr.item()))

            if global_step % 50 == 0:
                avg = running_loss / max(1, running_tokens)
                print(
                    f"[train] epoch={epoch} step={global_step} avg_loss={avg:.4f} tokens={running_tokens}")

            # Optional on-the-fly evaluation
            if args.eval_every_steps > 0 and val_dl is not None and global_step % args.eval_every_steps == 0:
                head.eval()
                pooler.eval()
                acc, macro, tok_cer, grp_cer, ntok = eval_metrics(
                    rec_model=rec_model,
                    pooler=pooler,
                    head=head,
                    dl=val_dl,
                    feat_source=args.feat_source,
                    min_range_len=args.min_range_len,
                    boundary_weight=args.boundary_weight,
                    viterbi_lambda=args.viterbi_lambda,
                    debug_batches=args.debug_val_batches,
                    num_fonts=num_fonts,
                )
                print(
                    f"[eval@step] epoch={epoch} step={global_step} tokens={ntok} "
                    f"acc={acc:.4f} macro_acc={macro:.4f} token_CER={tok_cer:.4f} group_CER={grp_cer:.4f}"
                )
                head.train()
                pooler.train()

        msg = f"[train] epoch={epoch} avg_loss={running_loss / max(1, running_tokens):.6f}"
        if boundary_losses:
            msg += f" boundary_loss={float(np.mean(boundary_losses)):.6f}"
        if tv_losses:
            msg += f" tv_loss={float(np.mean(tv_losses)):.6f}"
        if args.print_tran_stats and pred_tr_list:
            msg += f" pred_tr={float(np.mean(pred_tr_list)):.4f} gt_tr={float(np.mean(gt_tr_list)):.4f}"
        msg += f" kept_tokens={epoch_kept_tokens}"
        print(msg)

        print(f"[OK] saved epoch={epoch} checkpoints")
        paddle.save(head.state_dict(), str(
            args.out_dir / f"font_head_epoch{epoch}.pdparams"))
        paddle.save(pooler.state_dict(), str(
            args.out_dir / f"pooler_epoch{epoch}.pdparams"))
        paddle.save(head.state_dict(), str(
            args.out_dir / "font_head_last.pdparams"))
        paddle.save(pooler.state_dict(), str(
            args.out_dir / "pooler_last.pdparams"))

        if val_dl is not None:
            acc, macro, tok_cer, grp_cer, ntok = eval_metrics(
                rec_model=rec_model,
                pooler=pooler,
                head=head,
                dl=val_dl,
                feat_source=args.feat_source,
                min_range_len=args.min_range_len,
                boundary_weight=args.boundary_weight,
                viterbi_lambda=args.viterbi_lambda,
                debug_batches=args.debug_val_batches,
                num_fonts=num_fonts,
            )
            print(
                f"[val] epoch={epoch} acc={acc:.4f} macro_acc={macro:.4f} "
                f"token_font_CER={tok_cer:.4f} group_font_CER={grp_cer:.4f} tokens={ntok}")

            if args.main_metric == "acc":
                cand = acc
                better = cand > best_score
            elif args.main_metric == "token_cer":
                cand = tok_cer
                better = cand < best_score
            else:
                cand = grp_cer
                better = cand < best_score

            if args.save_best and better:
                best_score = cand
                paddle.save(head.state_dict(), str(best_path))
                paddle.save(pooler.state_dict(), str(best_pool))
                print(f"[OK] saved BEST {args.main_metric}={best_score:.4f}")

        # Optional additional eval after each epoch (same metrics; for monitoring even when save_best is off)
        if args.eval_every_epoch and val_dl is not None:
            acc, macro, tok_cer, grp_cer, ntok = eval_metrics(
                rec_model=rec_model,
                pooler=pooler,
                head=head,
                dl=val_dl,
                feat_source=args.feat_source,
                min_range_len=args.min_range_len,
                boundary_weight=args.boundary_weight,
                viterbi_lambda=args.viterbi_lambda,
                debug_batches=args.debug_val_batches,
                num_fonts=num_fonts,
            )
            print(
                f"[val_epoch] epoch={epoch} acc={acc:.4f} macro_acc={macro:.4f} "
                f"token_font_CER={tok_cer:.4f} group_font_CER={grp_cer:.4f} tokens={ntok}")

        # Save training state for resume
        paddle.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "opt": opt.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            str(state_path),
        )


if __name__ == "__main__":
    main()
