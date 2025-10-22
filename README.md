# 🏛️ Gothi-Read
S
**Track B:** *OCR + Font Group Recognition (Per-Character Multi-Task)*  
**Author:** Abdul Rafey  
**Repository:** https://github.com/rafeyshah/gothi-read  

---

## 🚀 Overview

This repository implements the foundation of an end-to-end OCR pipeline for **Track B**.

The objective is to evaluate and fine-tune OCR models such as **TrOCR**, **PaddleOCR**, **Donut**, **MMOCR**, and **docTR** to perform both:
1. Optical Character Recognition (text transcription), and  
2. Font Group Recognition (per-character classification).

So far, the complete **data preparation and verification pipeline** has been developed and validated.  
This includes environment setup, dataset scanning, manifest building, Unicode-safe character alignment, data visualization, and metric computation.

---

## ⚙️ Implemented Components

### 🧩 Environment Setup
- Configured **Python**, **PyTorch**, **Hugging Face**, and **CUDA** environment.
- Verified GPU availability and package compatibility in Colab/VSCode.
- Organized the repository structure with clear directories for scripts, data, notebooks, and experiments.

### 🧾 Dataset Handling and Manifest Creation
- Implemented `build_manifest.py` to automatically generate CSV manifests listing `.jpg`, `.txt`, and `.font` triplets.
- Each entry includes Unicode grapheme counts using `regex \X` for accurate multi-byte character handling.
- Added missing-file and length-mismatch detection logic.
- Created `check_integrity.py` to summarize data health and integrity statistics.
- Designed `make_test_split.py` for reproducible train/test splits with configurable ratios and random seeds.

**Example manifest summary output:**
```
== train.csv ==
Total lines         : 163023
Clean (ok=True)     : 100%
Missing image/txt/font : 0
Length mismatches   : 0
Issues total        : 0
```

### 🔡 Unicode-Safe Loader and Alignment Checks
- Developed `icdar24.py` containing two robust dataset classes:
  - `LineDataset` – simple directory layout.  
  - `FlexibleLineDataset` – supports FAUBox-style nested structure.
- Implemented Unicode-safe text splitting (`split_into_chars`) using **grapheme clusters** to handle ligatures and diacritics correctly.
- Added verification ensuring `len(characters) == len(font_labels)` for all lines.
- Built integration checks and metric computations (CER/WER) using **JiWER** in `metrics.py`.

### 🖼 Visualization
- Implemented `visualize_line.py` to display each image alongside its recognized text and font labels.
- Each character is colored according to its font group, with a legend mapping label → color.
- Automatically handles length mismatches by truncating to the shorter sequence.
- Outputs saved to `exp/viz/` for quick inspection.

**Command example:**
```
python scripts/visualize_line.py --manifest manifests/train.csv --num 8
```

### 🧮 Metrics
- Added `metrics.py` for computing **Character Error Rate (CER)** and **Word Error Rate (WER)** using JiWER.
- Placeholder normalization function for consistent text processing before evaluation.

### 🔤 Vocabulary Preparation
- Implemented `build_vocab.py` to build character-level vocabularies directly from `.txt` files using the Unicode-safe splitter.

### 🧠 Zero-Shot Baseline (TrOCR)
- Added `zero_shot_trocr.py` to perform inference using **microsoft/trocr-base-printed** as a baseline OCR model.
- Outputs predictions and placeholder metrics for further evaluation.

---

## 📁 Repository Structure

```
gothi-read/
├── notebooks/
│   ├── 01_Environment_&_Repo_Setup.ipynb
│   ├── 02_Data_loader,_alignment_checks,_and_metrics.ipynb
│
├── scripts/
│   ├── build_manifest.py
│   ├── check_integrity.py
│   ├── make_test_split.py
│   ├── visualize_line.py
│   ├── build_vocab.py
│   ├── zero_shot_trocr.py
│
├── src/
│   ├── icdar24.py
│   ├── metrics.py
│
└── exp/
    └── viz/ (generated visualizations)
```

---

## ✅ Achievements So Far

- [x] Verified Python, PyTorch, Hugging Face, CUDA environment  
- [x] Built manifest generator and dataset integrity validator  
- [x] Implemented Unicode-safe data loading and character alignment  
- [x] Developed visualizer for per-character font labels  
- [x] Computed OCR metrics (CER/WER) using JiWER  
- [x] Built vocabulary extraction and zero-shot TrOCR baseline  

---

## 🔜 Next Steps

- Integrate **font classification head** into OCR encoder (multi-task fine-tuning).  
- Evaluate **zero-shot** and **fine-tuned** models on train/validation splits.  
- Benchmark multiple OCR architectures (TrOCR, PaddleOCR, Donut, MMOCR, docTR).  
- Compute both **text** and **font-group** CER for comparison.  
- Begin model fine-tuning experiments (Day 3+).

---

## 🏁 Summary

The repository now has a **fully verified data pipeline** Track B.  
All dataset integrity, alignment, and visualization issues have been resolved.  
The project is ready to transition into **model benchmarking and fine-tuning**.

