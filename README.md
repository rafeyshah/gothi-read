# 🏛️ Gothi-Read

**Track B:** *OCR + Font Group Recognition (Per-Character Multi-Task)*  
**Author:** Abdul Rafey  
**Repository:** https://github.com/rafeyshah/gothi-read  

---

## 🚀 Overview

**Gothi-Read** is an end-to-end OCR + font-group recognition framework developed for **Pattern Recognition Lab**.  
The goal is to build and benchmark models capable of:

1. **Optical Character Recognition** — text transcription from scanned lines.  
2. **Font Group Recognition** — predicting the font category for every character.

The repository now provides:
- A verified, Unicode-safe data pipeline  
- Manifest generation and integrity checks  
- Visualization of font annotations  
- Evaluation scripts with unified model harness  
- Metrics computation for CER/WER and font accuracy

## ⚙️ Environment Setup

- Configured **Python 3 + PyTorch + Hugging Face + CUDA**.  
- Verified GPU availability and reproducibility across Colab and VS Code.  
- Clear modular directory layout: `scripts/`, `src/`, `notebooks/,` `runs/`.  

Main dependencies:
```bash
pip install torch torchvision torchaudio transformers jiwer pillow regex matplotlib
```

## 🧾 Dataset Handling and Validation

- `build_manifest.py` – scans dataset folders to create manifest CSVs listing `.jpg`, `.txt`, and `.font` triplets.  
- `check_integrity.py` – summarizes file presence & alignment health.  
- `make_test_split.py` – builds reproducible test subsets.  
- Verified 100 % length alignment between text and font sequences.

**Validation Integrity Summary**
- Total lines : 4040
- Clean (ok=True) : 3827 (94.73 %)
- Missing txt : 213
- Length mismatches : 0
- ✅ 94.7 % of validation lines are clean — ready for evaluation.

## 🔡 Unicode-Safe Data Loader and Alignment

- Implemented in `src/icdar24.py`.  
- Uses `regex \X` to split Unicode grapheme clusters — accurate for ligatures and diacritics.  
- Includes two dataset classes:  
  - `LineDataset` for flat folder structures.  
  - `FlexibleLineDataset` for nested FAUBox-style layouts.  
- Strict assertions ensure `len(characters) == len(font_labels)` for every sample.  
- Optional filters for Latin-8 font groups and subset selection.

## 🖼 Visualization

`visualize_line.py` renders each line image with colored font labels per character.

**Features**
- Shows image + text with per-char color coding by font group.  
- Legend auto-maps font labels → colors.  
- Trims length mismatches safely.  
- Saves visualizations to `exp/viz/`.

**Example command**
```bash
python scripts/visualize_line.py --manifest manifests/train.csv --num 8
```

## 🧮 Metrics Computation

- `src/metrics.py` computes:
  - **CER** (Character Error Rate)
  - **WER** (Word Error Rate)
- Uses **JiWER** for standard evaluation.  
- Normalization pipeline prepared for future expansion to font-CER.  
- Outputs aggregated JSON metrics for consistency across models.

## 🧠 Unified Model Evaluation Harness

`harness.py` provides a single interface to evaluate any OCR model.

**Functions**
```python
predict_lines(model_name, images) → List[str]  
evaluate(text_preds, text_gts) → {CER, WER, per_book, per_fontmix}
```

**Image Preprocessing**  
grayscale → resize(height = 64) → pad to max width

**Outputs saved to**
```
runs/<model>/<date>/
  preds.txt  
  metrics.json  
  per_line.csv  # img_id, gt, pred, CER
```

Example:
```bash
python scripts/day3_harness.py   --manifest manifests/valid.csv   --model microsoft/trocr-base-printed   --height 64   --limit 1000
```

## 📁 Repository Structure

```text
gothi-read/
├── notebooks/  
│  
├── scripts/  
│   ├── build_manifest.py  
│   ├── check_integrity.py  
│   ├── make_test_split.py  
│   ├── visualize_line.py  
│   ├── build_vocab.py  
│   ├── zero_shot_trocr.py  
│   ├── harness.py  
│  
├── src/  
│   ├── icdar24.py  
│   ├── metrics.py  
│  
└── runs/  
    └── microsoft_trocr-base-printed/  
```

## ✅ Achievements

- Environment and GPU setup completed  
- Dataset manifests validated (0 length mismatches)  
- Unicode-safe data loader implemented  
- Visualization utility verified  
- Metric computation (CER/WER) operational  
- Unified evaluation harness tested successfully  
- Zero-shot TrOCR baseline benchmarked with beam vs greedy  
- Decoding comparison (Day 4). **Greedy decoding** (num_beams = 1) **gave slightly better average CER/WER overall**, while **beam search** (num_beams = 5) **performed better on difficult or ambiguous lines.**  
- Zero-shot **PaddleOCR (PP-OCRv4 English)** baseline benchmarked on the same validation split.

## 🔜 Next Steps

- Add font-classification head to OCR encoder for multi-task learning.  
- Extend PaddleOCR experiments and integrate additional models: Donut, MMOCR, docTR.  
- Benchmark all models on the same validation split.  
- Compute joint **text CER + font-CER**.  
- Build a leaderboard under `/runs/` for cross-model comparisons.

## 🏁 Summary

**Gothi-Read** now includes a validated data pipeline, visualization system, and unified model evaluation framework.  
All data integrity, alignment, and evaluation steps are complete.  
The project is ready for multi-model benchmarking and fine-tuning experiments for Pattern Recognition Lab.

Current zero-shot OCR baselines on the validation split:

| run           | CER       | WER       |
|---------------|-----------|-----------|
| **Paddle-OCRv4**     | **0.203298** | **0.755115** |
| trocr-greedy  | 1.255461  | 1.598408  |
| trocr-beam    | 1.361015  | 1.732157  |

PaddleOCR (PP-OCRv4, English, detection disabled, line-crop recognizer) currently achieves the best CER/WER among the evaluated models.
