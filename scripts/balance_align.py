import json
from pathlib import Path
from typing import List, Dict
import random


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_mixed(rows: List[Dict]):
    mixed, single = [], []
    for r in rows:
        if not r.get("ok_align", False):
            continue
        fonts = (
            r.get("gt_fonts")
            or r.get("fonts")
            or r.get("font_ids")
            or r.get("font_id")
            or []
        )
        if len(set(fonts)) > 1:
            mixed.append(r)
        else:
            single.append(r)
    return mixed, single


def main():
    root = Path(__file__).resolve().parents[1]  # repo root
    train_in = root / "configs" / "train_align.jsonl"
    val_in = root / "configs" / "valid_align.jsonl"
    train_out = root / "configs" / "train_align_balanced.jsonl"
    val_out = root / "configs" / "valid_align_balanced.jsonl"

    train_rows = load_jsonl(train_in)
    val_rows = load_jsonl(val_in)

    train_mixed, train_single = split_mixed(train_rows)
    val_mixed, val_single = split_mixed(val_rows)

    # Deterministic shuffle for stable sampling.
    rng = random.Random(42)
    rng.shuffle(train_mixed)
    rng.shuffle(train_single)
    rng.shuffle(val_mixed)
    rng.shuffle(val_single)

    # Keep validation split isolated. Upsample mixed-font rows to improve transition learning.
    # x12 means each row is repeated 12 times (original + 11 copies).
    mixed_oversampled = []
    for r in train_mixed:
        mixed_oversampled.extend([r] * 12)

    # Downsample train single-font rows by 50% to reduce skew.
    train_single_down = train_single[::2]

    balanced_train = mixed_oversampled + train_single_down
    rng.shuffle(balanced_train)

    # Balanced validation remains leakage-free and uses only ok_align rows.
    # Keep all mixed rows and downsample single rows by 50% for stress-testing transitions.
    val_single_down = val_single[::2]
    balanced_val = val_mixed + val_single_down
    rng.shuffle(balanced_val)

    dump_jsonl(train_out, balanced_train)
    dump_jsonl(val_out, balanced_val)

    print(f"Train balanced: {len(balanced_train)} lines "
          f"(ok_align mixed {len(train_mixed)} x12 => {len(mixed_oversampled)}, "
          f"ok_align single kept {len(train_single_down)}/{len(train_single)})")
    print(f"Val balanced:   {len(balanced_val)} lines "
          f"(ok_align mixed kept {len(val_mixed)}, "
          f"ok_align single kept {len(val_single_down)}/{len(val_single)})")


if __name__ == "__main__":
    main()
