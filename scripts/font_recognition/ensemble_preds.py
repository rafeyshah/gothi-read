#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import regex as re

_GRAPHEME_RE = re.compile(r"\X", re.UNICODE)


def parse_font_seq(raw: str) -> List[str]:
    s = (raw or "").strip()
    return list(s) if s else []


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


def load_preds(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    out: Dict[str, List[str]] = {}
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = (row.get("id") or row.get("img_id") or "").strip()
                pred = row.get("pred_fonts") or row.get("pred") or ""
                if sid:
                    out[sid] = parse_font_seq(pred)
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if not s:
                    continue
                if "\t" in s:
                    sid, pred = s.split("\t", 1)
                else:
                    parts = s.split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    sid, pred = parts[0], parts[1]
                out[sid.strip()] = parse_font_seq(pred)
    return out


def _resolve_from_content(path_str: str, data_root: Optional[Path]) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if data_root is None:
        return p
    norm = path_str.replace("\\", "/")
    marker = "/dataset/"
    if marker in norm:
        rel = norm.split(marker, 1)[1]
        return data_root / Path(rel)
    return p


def _load_font_tokens_from_path(font_path: Path) -> List[str]:
    if not font_path.exists():
        return []
    s = font_path.read_text(encoding="utf-8").strip()
    return list(s) if s else []


def load_gt_fonts_from_csv(path: Optional[str], data_root: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    out: Dict[str, List[str]] = {}
    root = Path(data_root) if data_root else None
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ok_val = str(row.get("ok", "")).strip().upper()
            if ok_val not in {"TRUE", "1", "YES", "Y"}:
                continue
            sid = str(row.get("id", "")).strip()
            fp_raw = str(row.get("font_path", "")).strip()
            if not sid or not fp_raw:
                continue
            fp = _resolve_from_content(fp_raw, root)
            toks = _load_font_tokens_from_path(fp)
            if toks:
                out[sid] = toks
    return out


def load_expected_lens(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    p = Path(path)
    out: Dict[str, int] = {}
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = str(row.get("id", "")).strip()
                if not sid:
                    continue
                gt_units_len = str(row.get("gt_units_len", "")).strip()
                if gt_units_len.isdigit():
                    out[sid] = int(gt_units_len)
                    continue
                txt = row.get("gt_text") or row.get("text") or ""
                if isinstance(txt, str) and txt:
                    out[sid] = len(_GRAPHEME_RE.findall(txt))
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if not s or "\t" not in s:
                    continue
                sid, txt = s.split("\t", 1)
                sid = sid.strip()
                if sid:
                    out[sid] = len(_GRAPHEME_RE.findall(txt))
    return out


def build_priors(prior_csv: Optional[str], data_root: Optional[str]) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    if not prior_csv:
        return {}, {}

    root = Path(data_root) if data_root else None
    uni = Counter()
    bi = Counter()
    with Path(prior_csv).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ok_val = str(row.get("ok", "")).strip().upper()
            if ok_val not in {"TRUE", "1", "YES", "Y"}:
                continue
            fp_raw = str(row.get("font_path", "")).strip()
            if not fp_raw:
                continue
            fp = _resolve_from_content(fp_raw, root)
            toks = _load_font_tokens_from_path(fp)
            if not toks:
                continue
            uni.update(toks)
            bi.update(zip(toks[:-1], toks[1:]))

    total_uni = sum(uni.values())
    labs = sorted(uni.keys()) if uni else []
    if not labs or total_uni == 0:
        return {}, {}

    # Additive smoothing.
    k = 1.0
    uni_prob = {lab: (uni[lab] + k) / (total_uni + k * len(labs)) for lab in labs}

    bi_prob: Dict[Tuple[str, str], float] = {}
    out_counts: Dict[str, int] = Counter()
    for a, _ in bi:
        out_counts[a] += bi[(a, _)]
    for a in labs:
        denom = out_counts.get(a, 0) + k * len(labs)
        for b in labs:
            bi_prob[(a, b)] = (bi.get((a, b), 0) + k) / denom
    return uni_prob, bi_prob


def seq_risk(
    seq: Sequence[str],
    expected_len: Optional[int],
    uni_prob: Dict[str, float],
    bi_prob: Dict[Tuple[str, str], float],
) -> float:
    if not seq:
        return 1e9

    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    switch_ratio = switches / max(1, len(seq) - 1)

    p_uni = 0.0
    if uni_prob:
        for t in seq:
            p = uni_prob.get(t, 1e-9)
            p_uni += -math.log(max(p, 1e-9))
        p_uni /= len(seq)

    p_bi = 0.0
    if bi_prob and len(seq) > 1:
        acc = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            p = bi_prob.get((a, b), 1e-9)
            acc += -math.log(max(p, 1e-9))
        p_bi = acc / (len(seq) - 1)

    len_pen = 0.0
    if expected_len is not None and expected_len > 0:
        len_pen = abs(len(seq) - expected_len) / expected_len

    # Lower is better.
    return 1.2 * switch_ratio + 0.4 * p_uni + 0.6 * p_bi + 1.5 * len_pen


def compute_font_cer(preds: Iterable[Sequence[str]], refs: Iterable[Sequence[str]]) -> float:
    total_edits = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_edits += edit_distance(r, p)
        total_len += max(1, len(r))
    return total_edits / max(1, total_len)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", required=True, help="CRNN/first model predictions (preds.txt or per_line.csv)")
    ap.add_argument("--pred_b", required=True, help="Transformer/second model predictions (preds.txt or per_line.csv)")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--prior_csv", default=None, help="Optional train_clean.csv for unigram/bigram prior.")
    ap.add_argument("--expected_len_from", default=None, help="Optional clean.csv or TSV with id->text.")
    ap.add_argument("--ref_csv", default=None, help="Optional valid/test clean.csv for CER reporting.")
    ap.add_argument("--data_root", default="dataset", help="Used to resolve /content/dataset/... paths.")
    ap.add_argument("--out_dir", default="runs/font-ensemble")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_a = load_preds(args.pred_a)
    pred_b = load_preds(args.pred_b)
    ids = sorted(set(pred_a.keys()) | set(pred_b.keys()))

    uni_prob, bi_prob = build_priors(args.prior_csv, args.data_root)
    exp_len = load_expected_lens(args.expected_len_from)
    refs = load_gt_fonts_from_csv(args.ref_csv, args.data_root)

    chosen: Dict[str, List[str]] = {}
    rows = []
    pick_counts = Counter()

    for sid in ids:
        a = pred_a.get(sid, [])
        b = pred_b.get(sid, [])
        target_len = exp_len.get(sid)

        ra = seq_risk(a, target_len, uni_prob, bi_prob) if a else 1e9
        rb = seq_risk(b, target_len, uni_prob, bi_prob) if b else 1e9

        if ra <= rb:
            seq = a
            source = args.name_a
            risk = ra
        else:
            seq = b
            source = args.name_b
            risk = rb

        chosen[sid] = list(seq)
        pick_counts[source] += 1

        row = {
            "id": sid,
            "pred_ensemble": "".join(seq),
            "source": source,
            "risk": risk,
            "len_pred": len(seq),
            "len_expected": target_len if target_len is not None else "",
            "pred_a": "".join(a),
            "pred_b": "".join(b),
        }
        if sid in refs:
            r = refs[sid]
            row["gt_fonts"] = "".join(r)
            row["edits"] = edit_distance(r, seq)
            row["font_cer"] = row["edits"] / max(1, len(r))
        rows.append(row)

    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\t{''.join(chosen[sid])}\n")

    fieldnames = [
        "id",
        "source",
        "risk",
        "len_pred",
        "len_expected",
        "pred_ensemble",
        "pred_a",
        "pred_b",
        "gt_fonts",
        "edits",
        "font_cer",
    ]
    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    metrics = {
        "num_lines": len(ids),
        "picked_from": dict(pick_counts),
        "has_refs": bool(refs),
    }
    if refs:
        pred_list = [chosen[sid] for sid in ids if sid in refs]
        ref_list = [refs[sid] for sid in ids if sid in refs]
        metrics["font_cer"] = compute_font_cer(pred_list, ref_list)
        metrics["num_eval_lines"] = len(ref_list)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
