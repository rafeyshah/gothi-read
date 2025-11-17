#!/usr/bin/env bash
set -e
python - << 'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable, "-m", "pip"]+list(args))
# CPU install (safe on Colab Day 1). Swap to GPU later if needed.
pipi("install", "--upgrade", "pip")
pipi("install", "paddlepaddle==3.0.0")
pipi("install", "paddleocr")
print("Installed CPU PaddleOCR. For GPU later, see Paddle docs for a CUDA-matched wheel.")
PY
