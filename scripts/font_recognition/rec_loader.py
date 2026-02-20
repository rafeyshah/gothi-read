#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# -----------------------------
# Make PaddleOCR importable
# -----------------------------
PADDLEOCR_ROOT = Path(os.environ.get(
    "PADDLEOCR_ROOT", "/content/PaddleOCR")).resolve()
if not (PADDLEOCR_ROOT / "ppocr").exists():
    raise RuntimeError(
        f"Cannot find ppocr/ under PADDLEOCR_ROOT={PADDLEOCR_ROOT}")
sys.path.insert(0, str(PADDLEOCR_ROOT))

import paddle  # noqa: E402


def _load_yaml(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config did not parse as dict: {config_path}")
    return cfg


def _read_dict_tokens(dict_path: Path) -> List[str]:
    tokens: List[str] = []
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n\r")
            if tok == "":
                continue
            tokens.append(tok)
    return tokens


def _resolve_dict_path(cfg: Dict[str, Any], config_path: str) -> Path:
    g = cfg.get("Global", {}) or {}
    p = g.get("character_dict_path", None)
    if not p:
        raise RuntimeError("Global.character_dict_path is missing in config")

    p = str(p)
    base = Path(config_path).resolve().parent
    dict_path = (
        base / p).resolve() if p.startswith(".") or not Path(p).is_absolute() else Path(p)

    if not dict_path.exists():
        alt = (PADDLEOCR_ROOT / p.lstrip("./")).resolve()
        if alt.exists():
            dict_path = alt

    if not dict_path.exists():
        raise FileNotFoundError(f"character_dict_path not found: {dict_path}")

    return dict_path


def _infer_decoders_used(cfg: Dict[str, Any]) -> Set[str]:
    arch = cfg.get("Architecture", {}) or {}
    head = arch.get("Head", {}) or {}
    head_list = head.get("head_list", []) or []

    need: Set[str] = set()
    for item in head_list:
        if not isinstance(item, dict) or not item:
            continue
        name = list(item.keys())[0]
        if name == "CTCHead":
            need.add("CTCLabelDecode")
        elif name == "NRTRHead":
            need.add("NRTRLabelDecode")
        elif name == "SARHead":
            need.add("SARLabelDecode")

    pp = cfg.get("PostProcess", {}) or {}
    if pp.get("name") == "CTCLabelDecode":
        need.add("CTCLabelDecode")
    return need


def _inject_out_channels_list(cfg: Dict[str, Any], config_path: str) -> None:
    g = cfg.get("Global", {}) or {}
    use_space_char = bool(g.get("use_space_char", False))

    dict_path = _resolve_dict_path(cfg, config_path)
    tokens = _read_dict_tokens(dict_path)

    if use_space_char and " " not in tokens:
        tokens.append(" ")

    # CTC classes = len(tokens) + 1 (blank)
    num_classes = len(tokens) + 1

    needed = _infer_decoders_used(cfg)
    out_channels_list = {k: num_classes for k in needed}

    cfg.setdefault("Architecture", {})
    cfg["Architecture"].setdefault("Head", {})
    cfg["Architecture"]["Head"]["out_channels_list"] = out_channels_list


def _load_weights(model: paddle.nn.Layer, cfg: Dict[str, Any], checkpoint_path: str) -> None:
    try:
        from ppocr.utils.save_load import load_model  # type: ignore
        load_model(cfg, model, checkpoint_path)
        return
    except Exception:
        pass

    state = paddle.load(checkpoint_path)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format at {checkpoint_path}")

    # Shape-safe loading: keep only parameters whose shapes match.
    cur = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in cur and list(cur[k].shape) == list(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    cur.update(filtered)
    model.set_state_dict(cur)

    if skipped:
        print(f"[WARN] Skipped loading {len(skipped)} params due to shape mismatch: {skipped[:6]}{'...' if len(skipped)>6 else ''}")


def _pick_main_tensor(x):
    if isinstance(x, dict):
        # best effort: prefer known keys, else first value
        for k in ["neck_out", "backbone_out", "x", "out"]:
            if k in x:
                return x[k]
        return next(iter(x.values()))
    return x


@paddle.no_grad()
def extract_rec_features(rec_model: paddle.nn.Layer, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
    """
    Extract recognition features for font classification.

    Returns:
      - pre_ctc_map: tensor after backbone+neck, typically [B,C,H,W]
      - im2seq:      tensor after Im2Seq reshape, [B,T,C]  (BEST for font style)
      - ctc_neck:    tensor after SequenceEncoder, [B,T,D] (more language-biased)
    """
    m = rec_model

    # Transform
    if getattr(m, "use_transform", False):
        x = m.transform(x)

    # Backbone
    if getattr(m, "use_backbone", False):
        x = m.backbone(x)
        x = _pick_main_tensor(x)

    # Neck
    if getattr(m, "use_neck", False):
        x = m.neck(x)
        x = _pick_main_tensor(x)

    pre_ctc_map = x

    head = getattr(m, "head", None)
    if head is None:
        raise RuntimeError("Cannot extract features: model.head not found.")

    # --- im2seq (preferred) ---
    im2seq = None
    if hasattr(head, "encoder_reshape"):
        # MultiHead defines encoder_reshape = Im2Seq(in_channels) :contentReference[oaicite:2]{index=2}
        try:
            im2seq = head.encoder_reshape(pre_ctc_map)  # [B,T,C]
        except Exception:
            im2seq = None

    # --- ctc neck (fallback / optional) ---
    ctc_neck = None
    if hasattr(head, "ctc_encoder"):
        try:
            ctc_neck = head.ctc_encoder(pre_ctc_map)  # should yield [B,T,D]
        except Exception:
            ctc_neck = None

    feats: Dict[str, paddle.Tensor] = {"pre_ctc_map": pre_ctc_map}
    if im2seq is not None:
        feats["im2seq"] = im2seq
    if ctc_neck is not None:
        feats["ctc_neck"] = ctc_neck

    if "im2seq" not in feats and "ctc_neck" not in feats:
        raise RuntimeError(
            "Failed to extract both im2seq and ctc_neck. "
            "Check that your rec model uses MultiHead/CTCHead (encoder_reshape/ctc_encoder)."
        )

    return feats


def load_rec_model_with_features(
    config_path: str,
    checkpoint_path: str,
    device: str = "gpu",
):
    paddle.set_device("cpu" if device == "cpu" else "gpu")

    cfg = _load_yaml(config_path)
    _inject_out_channels_list(cfg, config_path)

    # Guardrail: ensure dictionary size matches checkpoint embedding rows.
    try:
        dict_path = _resolve_dict_path(cfg, config_path)
        tokens = _read_dict_tokens(dict_path)
        use_space_char = bool(cfg.get("Global", {}).get("use_space_char", False))
        dict_tokens = len(tokens) + (1 if use_space_char and " " not in tokens else 0)

        state = paddle.load(checkpoint_path)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        emb_key = "head.gtc_head.embedding.embedding.weight"
        if isinstance(state, dict) and emb_key in state:
            ckpt_tokens = state[emb_key].shape[0]
            # NRTR-style decoders append BOS/EOS; allow either exact match or +2.
            expected = {dict_tokens, dict_tokens + 2}
            if ckpt_tokens not in expected:
                print(
                    f"[WARN] Dict/token mismatch: checkpoint expects {ckpt_tokens} tokens but dict has {dict_tokens} "
                    f"(or {dict_tokens + 2} incl. BOS/EOS). Dropping GTC/CTC head weights; they will be re-initialized."
                )
                for k in list(state.keys()):
                    if k.startswith("head.gtc_head.") or k.startswith("head.ctc_head."):
                        state.pop(k, None)
    except Exception as e:
        raise

    from ppocr.modeling.architectures.base_model import BaseModel  # type: ignore

    model = BaseModel(cfg["Architecture"])
    _load_weights(model, cfg, checkpoint_path)

    model.eval()
    return model
