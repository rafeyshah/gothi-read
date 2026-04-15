#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "ocr_recognition" / "prepare_paddleocr_specialists.py"),
        run_name="__main__",
    )
