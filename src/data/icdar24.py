import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from PIL import Image
import regex as re

# Default to Unicode grapheme cluster splitting.
# Replace with the competition-provided regex later for a perfect match.
_GRAPHEME_RE = re.compile(r"\X", re.UNICODE)

FONT_GROUPS = ["a", "b", "f", "G", "i", "r", "s", "t"]
FONT_TO_ID = {c:i for i, c in enumerate(FONT_GROUPS)}
ID_TO_FONT = {i:c for c, i in FONT_TO_ID.items()}

def split_into_chars(text: str) -> List[str]:
    """
    Split text into *user-perceived characters* (grapheme clusters).
    This is a robust default for Unicode; the competition publishes a specific regex you can paste here later.
    """
    return _GRAPHEME_RE.findall(text)

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
