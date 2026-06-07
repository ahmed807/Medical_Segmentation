"""
extract_test_bundle.py
----------------------
Build a minimal data bundle containing only the files referenced by the
per-(image, annotation) test split. For seg-only ablations the only required
fields are `image`, `annotation_mask`, and `annotation` (the GT segmap used
for IoU and BF1). `original_clean_image` is included by default because the
files are small and harmless, but can be excluded with --no-originals to
save bandwidth.

The script preserves the original directory layout under the source dataset
so the existing test.json paths resolve unchanged on the vast.ai side.

Usage:
    python3 extract_test_bundle.py \\
        --sam3-dataset /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset \\
        --out-dir inference_ablation_test_set/sam_finetuning_dataset
"""
import argparse
import json
import os
import shutil
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-dataset", required=True,
                    help="Path to sam_finetuning_dataset (source)")
    ap.add_argument("--out-dir", required=True,
                    help="Destination directory for the bundle")
    ap.add_argument("--no-originals", action="store_true",
                    help="Skip original_clean_image files (saves ~30%% bandwidth)")
    args = ap.parse_args()

    src_root = os.path.abspath(args.sam3_dataset)
    dst_root = os.path.abspath(args.out_dir)
    test_json = os.path.join(src_root, "test.json")

    if not os.path.exists(test_json):
        raise FileNotFoundError(f"No test.json at {test_json}")

    with open(test_json) as f:
        entries = json.load(f)
    print(f"[load] {len(entries):,} test entries from {test_json}")

    keys = ["image", "annotation_mask", "annotation"]
    if not args.no_originals:
        keys.append("original_clean_image")

    needed = set()
    missing_keys = Counter()
    for e in entries:
        for k in keys:
            v = e.get(k)
            if v:
                needed.add(v)
            else:
                missing_keys[k] += 1

    print(f"[collect] {len(needed):,} unique files referenced "
          f"({len(needed) / max(len(entries), 1):.2f} per entry)")
    for k, n in missing_keys.items():
        if n:
            print(f"  missing key {k!r}: {n} entries")

    os.makedirs(dst_root, exist_ok=True)
    shutil.copy(test_json, os.path.join(dst_root, "test.json"))

    copied = 0
    not_found = 0
    for rel in sorted(needed):
        src = os.path.join(src_root, rel)
        dst = os.path.join(dst_root, rel)
        if not os.path.exists(src):
            not_found += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        copied += 1
        if copied % 2000 == 0:
            print(f"[copy] {copied:,}/{len(needed):,}")

    print(f"[done] copied {copied:,} files to {dst_root}")
    if not_found:
        print(f"  WARNING: {not_found:,} referenced files not found on disk")

    total_bytes = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(dst_root)
        for f in files
    )
    print(f"[size] bundle = {total_bytes / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
