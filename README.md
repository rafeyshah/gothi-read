# ICDAR 2024 — Multi-Font Group Recognition & OCR (Track B)

Day 1 scaffold for Google Colab.

## What’s here
- **src/data/icdar24.py** — dataset + unicode-safe char splitting (grapheme clusters as a solid default; replace with competition regex later).
- **src/eval/metrics.py** — CER/WER helpers via `jiwer`.
- **src/models/multitask_trocr.py** — skeleton class for the multi-task TrOCR (font head placeholder).
- **scripts/zero_shot_trocr.py** — minimal zero-shot inference stub.
- **scripts/install_paddle.sh** — optional script to install PaddleOCR (CPU) to avoid CUDA mismatch on Day 1.
- **Day_1_Colab.ipynb** — one-click setup notebook for Colab (installs deps, verifies imports, creates folders).

## Suggested workflow (Day 1)
1. Open **Day_1_Colab.ipynb** in Colab.
2. Run cells in order: install, verify, mount Drive (optional), prepare folders.
3. (Optional) run the PaddleOCR installer cell at the end (CPU version for now).

> Tip: Later in Day 2–3, paste the *official* competition regex into `split_into_chars()` to match evaluation exactly.
