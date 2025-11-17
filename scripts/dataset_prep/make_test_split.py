
import os, json, random, shutil, argparse
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

def find_leaf_dirs_with_images(root: Path):
    """Recursively find all *leaf* directories under root that contain images.
    A leaf dir is a directory that has images and no image subdirs."""
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # images in current dir
        imgs = [f for f in filenames if Path(f).suffix.lower() in IMG_EXTS]
        # any subdir that contains images?
        has_img_subdir = False
        for d in dirnames:
            sub = p / d
            for _, _, files in os.walk(sub):
                if any(Path(f).suffix.lower() in IMG_EXTS for f in files):
                    has_img_subdir = True
                    break
            if has_img_subdir:
                break
        if imgs and not has_img_subdir:
            leaf_dirs.append(p)
    return sorted(leaf_dirs)

def collect_triplets(dir_path: Path):
    """Collect (stem, jpg, txt, font) triplets within a directory."""
    triplets = []
    for img in sorted(dir_path.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        stem = img.stem
        txt = dir_path / f"{stem}.txt"
        fnt = dir_path / f"{stem}.font"
        if txt.exists() and fnt.exists():
            triplets.append((stem, img, txt, fnt))
    return triplets

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path, move: bool):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help="Repo root with 'dataset/train' present")
    ap.add_argument('--split_ratio', type=float, default=0.10, help='Fraction per leaf directory')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--move', action='store_true', help='Move instead of copy (removes from train)')
    ap.add_argument('--include_single', type=lambda s: s.lower() in {'true','1','yes'}, default=True)
    ap.add_argument('--include_multiple', type=lambda s: s.lower() in {'true','1','yes'}, default=True)
    ap.add_argument('--dry_run', action='store_true', help='Plan only; do not copy/move files')
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    train_root = root / 'dataset' / 'train'
    test_root  = root / 'dataset' / 'test'

    if not train_root.exists():
        raise SystemExit(f"Train folder not found: {train_root}")

    subtrees = []
    if args.include_single and (train_root / 'single').exists():
        subtrees.append(('single', train_root / 'single'))
    if args.include_multiple and (train_root / 'multiple').exists():
        subtrees.append(('multiple', train_root / 'multiple'))
    if not subtrees:
        subtrees.append(('train', train_root))

    plan = []
    total_src = total_sel = 0

    for subset_name, subset_path in subtrees:
        leaf_dirs = find_leaf_dirs_with_images(subset_path)
        for leaf in leaf_dirs:
            triplets = collect_triplets(leaf)
            if not triplets:
                continue
            total_src += len(triplets)
            k = max(1, int(round(len(triplets) * args.split_ratio)))
            selected = random.sample(triplets, k=k)
            total_sel += len(selected)

            for stem, jpg, txt, fnt in selected:
                rel = leaf.relative_to(train_root)  # e.g., single/antiqua
                dst_leaf = test_root / rel
                dst_jpg = dst_leaf / jpg.name
                dst_txt = dst_leaf / txt.name
                dst_fnt = dst_leaf / fnt.name
                plan.append({
                    'subset': subset_name,
                    'leaf': str(leaf),
                    'dst_leaf': str(dst_leaf),
                    'stem': stem,
                    'jpg_src': str(jpg),
                    'txt_src': str(txt),
                    'font_src': str(fnt),
                    'jpg_dst': str(dst_jpg),
                    'txt_dst': str(dst_txt),
                    'font_dst': str(dst_fnt),
                })

    moved = 0
    if not args.dry_run:
        for item in plan:
            copy_or_move(Path(item['jpg_src']), Path(item['jpg_dst']), args.move)
            copy_or_move(Path(item['txt_src']), Path(item['txt_dst']), args.move)
            copy_or_move(Path(item['font_src']), Path(item['font_dst']), args.move)
            moved += 1

    report = {
        'root': str(root),
        'train_root': str(train_root),
        'test_root': str(test_root),
        'include_single': args.include_single,
        'include_multiple': args.include_multiple,
        'split_ratio': args.split_ratio,
        'seed': args.seed,
        'mode': 'move' if args.move else 'copy',
        'dry_run': args.dry_run,
        'leaf_dirs_processed': len(plan),
        'triplets_considered': total_src,
        'triplets_selected': total_sel,
    }
    print(json.dumps(report, indent=2))

    out_dir = root / 'exp' / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'make_test_split_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(out_dir / 'make_test_split_plan.json', 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)

    if args.dry_run:
        print('\n[DRY RUN] No files were copied/moved. Re-run WITHOUT --dry_run to apply.')
    else:
        print(f"\nDone. {'Moved' if args.move else 'Copied'} {moved} triplet(s) to test.")
        print(f"Test set available at: {test_root}")

if __name__ == '__main__':
    main()
