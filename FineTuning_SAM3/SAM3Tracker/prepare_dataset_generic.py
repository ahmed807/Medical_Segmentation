"""
prepare_dataset_generic.py
--------------------------
Generic-prompt-routing variant of prepare_dataset.py.

Identical to prepare_dataset.py in every respect EXCEPT one line:
the annotation type is forced to "rect_bbox" for every entry,
so the existing box-prompt routing in train.py and inference.py
applies uniformly to all annotations regardless of their actual
visual category (arrow, freeform_bbox, etc.).

This produces the "generic prompt baseline" used in the ablation
that defends type-specific routing in the thesis. The ablation
compares:

  - production routing : type-specific paths
                         (arrow tip, letter centroids, freeform
                          filled-contour mask)
  - generic baseline   : a single bounding box around the
                         annotation pixels, regardless of type
                         (the simplest naive baseline — what a
                         person would do without thinking about
                         prompt routing)

Both runs share the same training script, the same loss
composition, the same 30-epoch schedule, the same model
architecture, and the same dataset images. Only the prompt
geometry per annotation changes.

Usage:
    python prepare_dataset_generic.py
    python prepare_dataset_generic.py --resume

Output:
    sam_finetuning_dataset_generic/
        images/, masks/, prompt_masks/, originals/,
        train.json, val.json, test.json

The annotation_type field in every JSON entry is "rect_bbox".
The original (true) annotation type is preserved in a new field
"true_annotation_type" for downstream analysis only — train.py
and inference.py do not consume this field.
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
OUTPUT_DIR   = "sam_finetuning_dataset_generic"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED  = 42                          # Same seed as production for split parity
BBOX_PADDING = 10
NUM_WORKERS  = 16
MAX_ENTRIES  = None
# ───────────────────────────────────────────────────────────────────────────────


def fix_extension(folder_path, filename):
    base = os.path.splitext(filename)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder_path, base + ext)
        if os.path.exists(path):
            return path
    return None


def get_prompt_box(annot_np, padding=10):
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = annot_np.shape
        return [0, 0, w, h]
    all_pts      = np.vstack(contours)
    x, y, w, h   = cv2.boundingRect(all_pts)
    img_h, img_w = annot_np.shape
    return [
        max(0, x - padding),
        max(0, y - padding),
        min(img_w, x + w + padding),
        min(img_h, y + h + padding),
    ]


def true_annotation_type(description: str) -> str:
    """Reports the actual annotation visual category for downstream analysis."""
    desc = description.lower()
    if "arrow" in desc:
        return "arrow"
    if "dashed" in desc or "line" in desc:
        return "freeform_bbox"
    if "box" in desc or "rect" in desc or "rectangle" in desc:
        return "rect_bbox"
    return "number_letter"


def process_folder(args):
    (folder_name, folder_path,
     images_dir, masks_dir, prompt_masks_dir, originals_dir,
     resume) = args

    entries  = []
    warnings = []

    labels_path = os.path.join(folder_path, "labels.json")
    if not os.path.exists(labels_path):
        return entries, warnings

    try:
        with open(labels_path) as f:
            labels_data = json.load(f)
    except json.JSONDecodeError as e:
        warnings.append(f"[{folder_name}] Invalid JSON: {e}")
        return entries, warnings

    original_path = os.path.join(folder_path, "original.png")
    if os.path.exists(original_path):
        new_orig = f"{folder_name}_original.png"
        dest_orig = os.path.join(originals_dir, new_orig)
        if not (resume and os.path.exists(dest_orig)):
            shutil.copy(original_path, dest_orig)
        original_ref = f"originals/{new_orig}"
    else:
        original_ref = None

    for key, value in labels_data.items():
        target_img_name = value.get("annotated_img_name")
        if not target_img_name:
            warnings.append(f"[{folder_name}/{key}] Missing annotated_img_name")
            continue

        image_path  = fix_extension(folder_path, target_img_name)
        prompt_path = os.path.join(folder_path, f"annotation_mask_{key}.png")
        gt_path     = os.path.join(folder_path, f"annotation_segmap_{key}.png")

        if image_path is None:
            warnings.append(f"[{folder_name}/{key}] Image not found: {target_img_name}")
            continue
        if not os.path.exists(prompt_path):
            warnings.append(f"[{folder_name}/{key}] Missing: {prompt_path}")
            continue
        if not os.path.exists(gt_path):
            warnings.append(f"[{folder_name}/{key}] Missing: {gt_path}")
            continue

        unique_base = f"{folder_name}_{key}"

        new_image = f"{unique_base}.jpg"
        dest_image = os.path.join(images_dir, new_image)
        if not (resume and os.path.exists(dest_image)):
            img = cv2.imread(image_path)
            if img is None:
                warnings.append(f"[{folder_name}/{key}] Cannot read: {image_path}")
                continue
            cv2.imwrite(dest_image, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        new_mask = f"{unique_base}_segmap.png"
        dest_mask = os.path.join(masks_dir, new_mask)
        if not (resume and os.path.exists(dest_mask)):
            shutil.copy(gt_path, dest_mask)

        new_prompt_mask = f"{unique_base}_annot_mask.png"
        dest_prompt = os.path.join(prompt_masks_dir, new_prompt_mask)
        if not (resume and os.path.exists(dest_prompt)):
            shutil.copy(prompt_path, dest_prompt)

        annot_np = cv2.imread(prompt_path, cv2.IMREAD_GRAYSCALE)
        if annot_np is None:
            warnings.append(f"[{folder_name}/{key}] Cannot read mask: {prompt_path}")
            continue

        annotation_description = value.get("annotation", "unknown")
        object_label           = value.get("object", "unknown")
        prompt_box             = get_prompt_box(annot_np, padding=BBOX_PADDING)

        # ─── KEY DIFFERENCE FROM PRODUCTION prepare_dataset.py ───
        # The annotation_type is uniformly "rect_bbox" so the
        # existing box-prompt routing applies to every entry
        # regardless of what the annotation actually is. The true
        # category is preserved separately for downstream analysis.
        # ─────────────────────────────────────────────────────────
        forced_type = "rect_bbox"
        actual_type = true_annotation_type(annotation_description)

        entries.append({
            "image":                  f"images/{new_image}",
            "annotation":             f"masks/{new_mask}",
            "annotation_mask":        f"prompt_masks/{new_prompt_mask}",
            "original_clean_image":   original_ref,
            "prompt_box":             prompt_box,
            "prompt_text":            object_label,
            "label":                  object_label,
            "annotation_description": annotation_description,
            "annotation_type":        forced_type,
            "true_annotation_type":   actual_type,   # for analysis only
        })

    return entries, warnings


def prepare_dataset(resume=False):
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

    all_entries  = []
    all_warnings = []

    worker_args = [
        (name, os.path.join(RAW_DATA_DIR, name),
         images_dir, masks_dir, prompt_masks_dir, originals_dir, resume)
        for name in folder_names
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_folder, a): a[0] for a in worker_args}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Processing folders", unit="folder")
        for future in pbar:
            entries, warnings = future.result()
            all_entries.extend(entries)
            all_warnings.extend(warnings)
            pbar.set_postfix(entries=len(all_entries), warnings=len(all_warnings))

    if all_warnings:
        print(f"\n{len(all_warnings)} warnings (first 30):")
        for w in all_warnings[:30]:
            print(f"  {w}")
        with open(os.path.join(OUTPUT_DIR, "warnings.txt"), "w") as f:
            f.write("\n".join(all_warnings))

    random.seed(RANDOM_SEED)
    random.shuffle(all_entries)
    if MAX_ENTRIES is not None:
        all_entries = all_entries[:MAX_ENTRIES]

    total   = len(all_entries)
    n_train = int(total * SPLIT_RATIOS[0])
    n_val   = int(total * SPLIT_RATIOS[1])

    train_data = all_entries[:n_train]
    val_data   = all_entries[n_train : n_train + n_val]
    test_data  = all_entries[n_train + n_val :]

    for split_name, split_data in [("train", train_data),
                                   ("val",   val_data),
                                   ("test",  test_data)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)

    type_counts = Counter(e["true_annotation_type"] for e in all_entries)

    print(f"\nGeneric-prompt dataset ready in '{OUTPUT_DIR}/'")
    print(f"  Total    {total:>8,}")
    print(f"  Train    {len(train_data):>8,}")
    print(f"  Val      {len(val_data):>8,}")
    print(f"  Test     {len(test_data):>8,}")
    print(f"\n  All entries report annotation_type='rect_bbox' (forced).")
    print(f"  This routes through the box-prompt code path uniformly.")
    print(f"\n  True visual-category breakdown (from descriptions):")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<22} {c:>7,}  ({100*c/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed files")
    args = parser.parse_args()
    prepare_dataset(resume=args.resume)
