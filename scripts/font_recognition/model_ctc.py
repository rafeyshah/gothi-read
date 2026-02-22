#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn


class FontCRNNCTCModel(nn.Module):
    def __init__(self, num_labels: int, lstm_hidden: int = 256):
        super().__init__()
        self.blank_id = 0
        self.num_labels = num_labels
        self.num_classes = num_labels + 1  # + CTC blank

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, self.num_classes)

    def output_lengths(self, input_widths: torch.Tensor) -> torch.Tensor:
        # Two 2x2 max-pooling layers.
        return torch.clamp(input_widths // 4, min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)  # [B, C, H', W']
        x = x.mean(dim=2)         # collapse height -> [B, C, W']
        x = x.permute(2, 0, 1)    # [T, B, C]
        x, _ = self.rnn(x)
        logits = self.classifier(x)  # [T, B, classes]
        return logits.log_softmax(dim=-1)


class FontTransformerCTCModel(nn.Module):
    def __init__(self, num_labels: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.blank_id = 0
        self.num_labels = num_labels
        self.num_classes = num_labels + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, self.num_classes)

    def output_lengths(self, input_widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input_widths // 4, min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = x.mean(dim=2)       # [B, C, W']
        x = x.permute(2, 0, 1)  # [T, B, C]
        x = self.transformer(x)
        logits = self.classifier(x)
        return logits.log_softmax(dim=-1)


def build_font_ctc_model(arch: str, num_labels: int) -> nn.Module:
    arch = (arch or "crnn").lower()
    if arch == "transformer":
        return FontTransformerCTCModel(num_labels=num_labels)
    if arch == "crnn":
        return FontCRNNCTCModel(num_labels=num_labels)
    raise ValueError(f"Unsupported arch: {arch}. Use one of: crnn, transformer")


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    id_to_label: Dict[int, str],
    blank_id: int = 0,
) -> List[List[str]]:
    # log_probs: [T, B, C]
    pred_ids = log_probs.argmax(dim=-1).transpose(0, 1)  # [B, T]
    outputs: List[List[str]] = []
    for b in range(pred_ids.shape[0]):
        L = int(input_lengths[b].item())
        seq = pred_ids[b, :L].tolist()
        decoded: List[str] = []
        prev = None
        for idx in seq:
            if idx == blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            if idx in id_to_label:
                decoded.append(id_to_label[idx])
            prev = idx
        outputs.append(decoded)
    return outputs


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            cost = 0 if ai == bj else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def font_cer(preds: Sequence[Sequence[str]], refs: Sequence[Sequence[str]]) -> float:
    total_edits = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_edits += edit_distance(r, p)
        total_len += max(1, len(r))
    return total_edits / max(1, total_len)


def smooth_token_sequence(tokens: Sequence[str], window: int = 3) -> List[str]:
    if window <= 1 or not tokens:
        return list(tokens)
    radius = window // 2
    out: List[str] = []
    for i in range(len(tokens)):
        l = max(0, i - radius)
        r = min(len(tokens), i + radius + 1)
        votes: Dict[str, int] = {}
        for t in tokens[l:r]:
            votes[t] = votes.get(t, 0) + 1
        # tie-break: keep original token
        top = max(votes.items(), key=lambda kv: (kv[1], kv[0] == tokens[i]))[0]
        out.append(top)
    return out
