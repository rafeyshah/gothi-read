import argparse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os, json, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="microsoft/trocr-base-printed")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proc = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.eval()

    results = {}
    for img_path in sorted(glob.glob(os.path.join(args.images_dir, "*.jpg"))):
        img = Image.open(img_path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        out_ids = model.generate(**inputs, max_length=args.max_length)
        text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(args.out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        results[stem] = text

    with open(os.path.join(args.out_dir, "metrics_placeholder.json"), "w") as f:
        json.dump({"note": "compute metrics later with GT texts using src/eval/metrics.py"}, f, indent=2)

if __name__ == "__main__":
    main()
