from typing import List, Optional
from jiwer import cer, wer

def normalize_text(s: str) -> str:
    # Define a single, consistent normalization for your project.
    # You may want to NFC-normalize and strip extra whitespace.
    return s

def compute_ocr_metrics(preds: List[str], refs: List[str]) -> dict:
    preds_n = [normalize_text(p) for p in preds]
    refs_n = [normalize_text(r) for r in refs]
    return {
        "CER": cer(refs_n, preds_n),
        "WER": wer(refs_n, preds_n),
    }

def compute_font_cer(pred_font: List[str], ref_font: List[str]) -> float:
    # pred_font/ref_font are strings like "abfG..." (lists or strings both fine)
    return cer(["".join(ref_font)], ["".join(pred_font)])
