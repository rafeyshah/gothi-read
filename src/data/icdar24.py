import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from PIL import Image
import regex as re
import glob
from pathlib import Path

# Default to Unicode grapheme cluster splitting.
# Replace with the competition-provided regex later for a perfect match.
_GRAPHEME_RE = re.compile(r"\X", re.UNICODE)
NESTED_ALLOWED_SUBSETS = {"single", "multiple"}  # what we expect under split/

FONT_GROUPS = ["a", "b", "f", "G", "i", "r", "s", "t"]
FONT_TO_ID = {c:i for i, c in enumerate(FONT_GROUPS)}
ID_TO_FONT = {i:c for c, i in FONT_TO_ID.items()}

def split_into_chars(text: str) -> List[str]:
    """
    Split text into *user-perceived characters* (grapheme clusters).
    This is a robust default for Unicode; the competition publishes a specific regex you can paste here later.
    """
    return _GRAPHEME_RE.findall(text)

def _collect_triplets_nested(split_dir: Path):
    """
    Walk nested folders and collect (img, txt, font) triplets that share a stem.
    Accepts: split_dir / {single|multiple} / <class_or_any> / <stem>.{jpg,txt,font}
    Also works if {single|multiple} are missing—will just walk everything under split_dir.
    """
    triplets = []
    # search all jpgs
    jpgs = glob.glob(str(split_dir / "**" / "*.jpg"), recursive=True)
    for jp in sorted(jpgs):
        stem = Path(jp).stem
        base = Path(jp).parent
        tp = base / f"{stem}.txt"
        fp = base / f"{stem}.font"
        if tp.exists() and fp.exists():
            triplets.append((str(jp), str(tp), str(fp)))
    return triplets

@dataclass
class LineItem:
    image_path: str
    text: str
    font_seq: Optional[str] = None  # same length as text, e.g., "aafGs..."

class LineDataset:
    """
    Expects a folder with image files and two sibling folders/files:
    - images/: 0000.jpg, ...
    - texts/:  0000.txt, ...
    - fonts/:  0000.font, ...   (optional at inference time)
    """
    def __init__(self, root: str, split: str, load_fonts: bool = True):
        self.root = root
        self.split = split
        self.images_dir = os.path.join(root, split, "images")
        self.texts_dir = os.path.join(root, split, "texts")
        self.fonts_dir = os.path.join(root, split, "fonts")
        self.load_fonts = load_fonts

        # Discover ids by scanning texts (or images)
        ids = []
        if os.path.isdir(self.texts_dir):
            for fn in sorted(os.listdir(self.texts_dir)):
                if fn.endswith(".txt"):
                    ids.append(os.path.splitext(fn)[0])
        else:
            for fn in sorted(os.listdir(self.images_dir)):
                if fn.endswith(".jpg"):
                    ids.append(os.path.splitext(fn)[0])

        self.items: List[LineItem] = []
        for id_ in ids:
            img = os.path.join(self.images_dir, f"{id_}.jpg")
            txt = os.path.join(self.texts_dir, f"{id_}.txt")
            fnt = os.path.join(self.fonts_dir, f"{id_}.font")
            if not (os.path.isfile(img) and os.path.isfile(txt)):
                continue
            with open(txt, "r", encoding="utf-8") as f:
                text = f.read().strip()
            font_seq = None
            if self.load_fonts and os.path.isfile(fnt):
                with open(fnt, "r", encoding="utf-8") as f:
                    font_seq = f.read().strip()
                # Optional sanity check
                tchars = split_into_chars(text)
                if len(tchars) != len(font_seq):
                    print(f"[WARN] length mismatch for {id_}: {len(tchars)} chars vs {len(font_seq)} fonts")
            self.items.append(LineItem(img, text, font_seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, Optional[str], str]:
        it = self.items[idx]
        img = Image.open(it.image_path).convert("RGB")
        return img, it.text, it.font_seq, os.path.basename(it.image_path)


class FlexibleLineDataset:
    """
    Supports TWO layouts:

    1) FLAT:
        root/data/<split>/{images,texts,fonts}/0000.{jpg,txt,font}
    2) NESTED (FAUBox):
        root/dataset/<split>/{single|multiple}/<class>/
            0000.jpg, 0000.txt, 0000.font (same folder)

    Args:
        root: base path, e.g. "/content/icdar24-multifont"
        split: "train", "val"/"valid", or "test_public"
        base_dirname: "data" (flat) or "dataset" (nested). If None, auto-detect.
        include_single / include_multiple: filter subsets in nested mode.
        include_unknown_classes: keep class folders that are not in the 8 Latin groups (e.g., greek/hebrew)
    """
    def __init__(
        self,
        root: str,
        split: str,
        base_dirname: str | None = None,
        load_fonts: bool = True,
        include_single: bool = True,
        include_multiple: bool = True,
        include_unknown_classes: bool = True,
    ):
        self.root = Path(root)
        split = {"val": "valid", "validation": "valid"}.get(split, split)
        self.split = split
        self.load_fonts = load_fonts

        # Try to detect layout if not specified
        if base_dirname is None:
            if (self.root / "data" / split).exists():
                base_dirname = "data"
            elif (self.root / "dataset" / split).exists():
                base_dirname = "dataset"
            else:
                raise FileNotFoundError(
                    f"Could not find '{self.root}/data/{split}' or '{self.root}/dataset/{split}'."
                )
        self.base_dirname = base_dirname

        self.items: list[LineItem] = []

        if base_dirname == "data":
            # --- FLAT layout (old) ---
            images_dir = self.root / "data" / split / "images"
            texts_dir  = self.root / "data" / split / "texts"
            fonts_dir  = self.root / "data" / split / "fonts"
            if not images_dir.exists():
                raise FileNotFoundError(f"Missing folder: {images_dir}")
            ids = [Path(p).stem for p in sorted(glob.glob(str(texts_dir / "*.txt")))]
            for id_ in ids:
                img = images_dir / f"{id_}.jpg"
                txt = texts_dir  / f"{id_}.txt"
                fnt = fonts_dir  / f"{id_}.font"
                if img.exists() and txt.exists():
                    text = txt.read_text(encoding="utf-8").strip()
                    font_seq = None
                    if load_fonts and fnt.exists():
                        font_seq = fnt.read_text(encoding="utf-8").strip()
                        tchars = split_into_chars(text)
                        if len(tchars) != len(font_seq):
                            print(f"[WARN] {id_}: len(chars)={len(tchars)} != len(fonts)={len(font_seq)}")
                    self.items.append(LineItem(str(img), text, font_seq))
        else:
            # --- NESTED layout (FAUBox) ---
            split_dir = self.root / "dataset" / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing folder: {split_dir}")

            # Optionally filter by subset (single/multiple)
            subset_dirs = []
            if include_single and (split_dir / "single").exists():
                subset_dirs.append(split_dir / "single")
            if include_multiple and (split_dir / "multiple").exists():
                subset_dirs.append(split_dir / "multiple")
            if not subset_dirs:
                subset_dirs = [split_dir]  # walk everything

            triplets = []
            for sd in subset_dirs:
                triplets.extend(_collect_triplets_nested(sd))

            for img, txt, fnt in triplets:
                text = Path(txt).read_text(encoding="utf-8").strip()
                font_seq = None
                if load_fonts and Path(fnt).exists():
                    font_seq = Path(fnt).read_text(encoding="utf-8").strip()

                # Optional: skip classes outside the 8 Latin groups (e.g., greek/hebrew), if desired
                if not include_unknown_classes:
                    # Class folder name is parent of the file
                    cls = Path(img).parent.name.lower()
                    latin8 = {"antiqua", "bastarda", "fraktur", "gotico-antiqua",
                              "italic", "rotunda", "schwabacher", "textura"}
                    if cls not in latin8:
                        continue

                if font_seq is not None:
                    tchars = split_into_chars(text)
                    if len(tchars) != len(font_seq):
                        stem = Path(img).stem
                        print(f"[WARN] {stem}: len(chars)={len(tchars)} != len(fonts)={len(font_seq)}")
                self.items.append(LineItem(img, text, font_seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it.image_path).convert("RGB")
        return img, it.text, it.font_seq, os.path.basename(it.image_path)
