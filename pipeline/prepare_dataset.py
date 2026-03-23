"""
prepare_dataset.py
------------------
Prepares the SAM3 fine-tuning dataset from the raw `output/` folder structure.
Optimized for ~50,000 images: parallel processing, progress tracking, resume support.

Expected input structure:
    output/
    ├── 0/
    │   ├── annotated_0.png
    │   ├── annotation_mask_0_0.png
    │   ├── annotation_segmap_0_0.png
    │   ├── labels.json
    │   └── original.png
    ├── 2/
    │   ├── annotated_0.png
    │   ├── annotation_mask_0_0.png  ...  annotation_mask_0_4.png
    │   ├── annotation_segmap_0_0.png ... annotation_segmap_0_4.png
    │   ├── labels.json
    │   └── original.png
    └── ...

Output structure:
    sam_finetuning_dataset/
    ├── images/           <- annotated images (with arrows/annotations)
    ├── masks/            <- segmentation maps (GT object masks for SAM)
    ├── prompt_masks/     <- annotation masks (arrow pixels -> SAM bbox prompt + Flux inpaint)
    ├── originals/        <- clean original images (GT for Flux evaluation)
    ├── train.json
    ├── val.json
    └── test.json

Usage:
    python prepare_dataset.py            # fresh build
    python prepare_dataset.py --resume   # skip already-processed files
"""

import os
import json
import shutil
import argparse
import cv2
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
RAW_DATA_DIR = "/home/ahma/unannotate/output"
OUTPUT_DIR   = "sam_finetuning_dataset"
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # Train / Val / Test
RANDOM_SEED  = 42
BBOX_PADDING = 10                 # Pixels of padding around annotation bbox
NUM_WORKERS  = 16                 # Parallel folder-processing workers
# ───────────────────────────────────────────────────────────────────────────────


def fix_extension(folder_path: str, filename: str):
    """Resolve actual file path regardless of extension (.png / .jpg / .jpeg)."""
    base = os.path.splitext(filename)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder_path, base + ext)
        if os.path.exists(path):
            return path
    return None


def get_prompt_box(annot_np: np.ndarray, padding: int = 10) -> list:
    """
    Derives a SAM bounding-box prompt from any annotation type:
      - Arrows / dashed lines -> single or elongated contour
      - Numbers / Letters     -> one or more small blobs
      - Rectangular bboxes    -> large rectangular contour
      - Freeform bboxes       -> irregular polygon contour

    Takes the union bbox of ALL contours + padding so multi-part
    annotations like "2a" are treated as a single prompt.
    """
    contours, _ = cv2.findContours(annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = annot_np.shape
        return [0, 0, w, h]

    all_pts      = np.vstack(contours)
    x, y, w, h   = cv2.boundingRect(all_pts)
    img_h, img_w = annot_np.shape
    return [
        max(0,     x - padding),
        max(0,     y - padding),
        min(img_w, x + w + padding),
        min(img_h, y + h + padding),
    ]


def coarse_annotation_type(description: str) -> str:
    """
    Maps free-text annotation description from labels.json to one of four
    coarse categories used for per-type evaluation metrics.

    Examples:
        "red arrow"          -> "arrow"
        "black dashed line"  -> "freeform_bbox"
        "purple 't'"         -> "number_letter"
        "bounding box"       -> "rect_bbox"
    """
    desc = description.lower()
    if "arrow" in desc:
        return "arrow"
    elif "dashed" in desc or "line" in desc:
        return "freeform_bbox"
    elif "box" in desc or "rect" in desc or "rectangle" in desc:
        return "rect_bbox"
    else:
        return "number_letter"


def process_folder(args: tuple) -> tuple:
    """
    Process a single folder from the raw dataset.
    Returns (entries: list, warnings: list).
    Safe to call in parallel workers.
    """
    (folder_name, folder_path,
     images_dir, masks_dir, prompt_masks_dir, originals_dir,
     resume) = args

    entries  = []
    warnings = []

    labels_path = os.path.join(folder_path, "labels.json")
    if not os.path.exists(labels_path):
        return entries, warnings

    try:
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
    except json.JSONDecodeError as e:
        warnings.append(f"[{folder_name}] Invalid JSON: {e}")
        return entries, warnings

    # Copy original clean image once per folder (used for Flux GT evaluation)
    original_path = os.path.join(folder_path, "original.png")
    if os.path.exists(original_path):
        new_original_filename = f"{folder_name}_original.png"
        dest_original = os.path.join(originals_dir, new_original_filename)
        if not (resume and os.path.exists(dest_original)):
            shutil.copy(original_path, dest_original)
        original_ref = f"originals/{new_original_filename}"
    else:
        original_ref = None

    for key, value in labels_data.items():
        target_img_name = value.get("annotated_img_name")
        if not target_img_name:
            warnings.append(f"[{folder_name}/{key}] Missing annotated_img_name")
            continue

        image_path = fix_extension(folder_path, target_img_name)
        if image_path is None:
            warnings.append(f"[{folder_name}/{key}] Image not found: {target_img_name}")
            continue

        prompt_path = os.path.join(folder_path, f"annotation_mask_{key}.png")
        gt_path     = os.path.join(folder_path, f"annotation_segmap_{key}.png")

        if not os.path.exists(prompt_path):
            warnings.append(f"[{folder_name}/{key}] Missing: {prompt_path}")
            continue
        if not os.path.exists(gt_path):
            warnings.append(f"[{folder_name}/{key}] Missing: {gt_path}")
            continue

        unique_base = f"{folder_name}_{key}"

        # ── Save annotated image (WITH annotation -> SAM + Flux input) ─────────
        new_image_filename = f"{unique_base}.jpg"
        dest_image = os.path.join(images_dir, new_image_filename)
        if not (resume and os.path.exists(dest_image)):
            img = cv2.imread(image_path)
            if img is None:
                warnings.append(f"[{folder_name}/{key}] Cannot read: {image_path}")
                continue
            cv2.imwrite(dest_image, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Save segmentation map (TARGET object GT mask for SAM training) ─────
        new_mask_filename = f"{unique_base}_segmap.png"
        dest_mask = os.path.join(masks_dir, new_mask_filename)
        if not (resume and os.path.exists(dest_mask)):
            shutil.copy(gt_path, dest_mask)

        # ── Save annotation mask (arrow pixels -> SAM bbox + Flux inpaint) ────
        new_prompt_mask_filename = f"{unique_base}_annot_mask.png"
        dest_prompt = os.path.join(prompt_masks_dir, new_prompt_mask_filename)
        if not (resume and os.path.exists(dest_prompt)):
            shutil.copy(prompt_path, dest_prompt)

        # ── Derive SAM bounding-box prompt from annotation mask ────────────────
        annot_np = cv2.imread(prompt_path, cv2.IMREAD_GRAYSCALE)
        if annot_np is None:
            warnings.append(f"[{folder_name}/{key}] Cannot read mask: {prompt_path}")
            continue
        prompt_box = get_prompt_box(annot_np, padding=BBOX_PADDING)

        annotation_description = value.get("annotation", "unknown")
        object_label           = value.get("object",     "unknown")

        entries.append({
            "image":                  f"images/{new_image_filename}",
            "annotation":             f"masks/{new_mask_filename}",
            "annotation_mask":        f"prompt_masks/{new_prompt_mask_filename}",
            "original_clean_image":   original_ref,
            "prompt_box":             prompt_box,
            "prompt_text":            object_label,            # SAM text prompt = object label
            "label":                  object_label,
            "annotation_description": annotation_description,  # "red arrow", "black dashed line"
            "annotation_type":        coarse_annotation_type(annotation_description),
        })

    return entries, warnings


def prepare_dataset(resume: bool = False):
    # ── 1. Setup output directories ────────────────────────────────────────────
    images_dir       = os.path.join(OUTPUT_DIR, "images")
    masks_dir        = os.path.join(OUTPUT_DIR, "masks")
    prompt_masks_dir = os.path.join(OUTPUT_DIR, "prompt_masks")
    originals_dir    = os.path.join(OUTPUT_DIR, "originals")

    if not resume and os.path.exists(OUTPUT_DIR):
        print(f"Removing existing '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)

    for d in [images_dir, masks_dir, prompt_masks_dir, originals_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: '{RAW_DATA_DIR}' not found.")
        return

    folder_names = sorted([
        f for f in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, f))
    ])
    print(f"Found {len(folder_names):,} folders in '{RAW_DATA_DIR}'")
    print(f"Processing with {NUM_WORKERS} parallel workers...\n")

    # ── 2. Parallel folder processing ──────────────────────────────────────────
    all_entries  = []
    all_warnings = []

    worker_args = [
        (name, os.path.join(RAW_DATA_DIR, name),
         images_dir, masks_dir, prompt_masks_dir, originals_dir, resume)
        for name in folder_names
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_folder, a): a[0] for a in worker_args}
        pbar    = tqdm(as_completed(futures), total=len(futures),
                       desc="Processing folders", unit="folder")
        for future in pbar:
            entries, warnings = future.result()
            all_entries.extend(entries)
            all_warnings.extend(warnings)
            pbar.set_postfix(entries=len(all_entries), warnings=len(all_warnings))

    # ── 3. Print warnings ──────────────────────────────────────────────────────
    if all_warnings:
        print(f"\n{len(all_warnings)} warnings (first 30):")
        for w in all_warnings[:30]:
            print(f"  {w}")
        if len(all_warnings) > 30:
            print(f"  ... and {len(all_warnings) - 30} more")
        # Save all warnings to file for inspection
        with open(os.path.join(OUTPUT_DIR, "warnings.txt"), "w") as f:
            f.write("\n".join(all_warnings))
        print(f"  Full warnings saved to {OUTPUT_DIR}/warnings.txt")

    # ── 4. Shuffle and split ───────────────────────────────────────────────────
    print(f"\nShuffling {len(all_entries):,} entries and splitting...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_entries)

    total   = len(all_entries)
    n_train = int(total * SPLIT_RATIOS[0])
    n_val   = int(total * SPLIT_RATIOS[1])

    train_data = all_entries[:n_train]
    val_data   = all_entries[n_train : n_train + n_val]
    test_data  = all_entries[n_train + n_val :]

    # ── 5. Save split JSONs ────────────────────────────────────────────────────
    for split_name, split_data in [("train", train_data),
                                    ("val",   val_data),
                                    ("test",  test_data)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    type_counts  = Counter(e["annotation_type"] for e in all_entries)
    label_counts = Counter(e["label"]           for e in all_entries)

    print(f"\nDataset ready in '{OUTPUT_DIR}/'")
    print(f"  Total    {total:>8,}")
    print(f"  Train    {len(train_data):>8,}  ({100*SPLIT_RATIOS[0]:.0f}%)")
    print(f"  Val      {len(val_data):>8,}  ({100*SPLIT_RATIOS[1]:.0f}%)")
    print(f"  Test     {len(test_data):>8,}  ({100*SPLIT_RATIOS[2]:.0f}%)")

    print(f"\n  Annotation type breakdown:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<22} {c:>7,}  ({100*c/total:.1f}%)")

    print(f"\n  Top-10 object labels:")
    for label, count in label_counts.most_common(10):
        print(f"    {label:<22} {count:>7,}  ({100*count/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed files (safe to re-run)")
    args = parser.parse_args()
    prepare_dataset(resume=args.resume)
