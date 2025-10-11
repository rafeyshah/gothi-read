# GothiRead

## What’s here
- **src/data/icdar24.py** — dataset + unicode-safe char splitting (grapheme clusters as a solid default; replace with competition regex later).
- **src/eval/metrics.py** — CER/WER helpers via `jiwer`.
- **src/models/multitask_trocr.py** — skeleton class for the multi-task TrOCR (font head placeholder).
- **scripts/zero_shot_trocr.py** — minimal zero-shot inference stub.
- **scripts/install_paddle.sh** — optional script to install PaddleOCR (CPU) to avoid CUDA mismatch on Day 1.
- **Day_1_Colab.ipynb** — one-click setup notebook for Colab (installs deps, verifies imports, creates folders).
