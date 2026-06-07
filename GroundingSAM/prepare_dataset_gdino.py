"""
prepare_dataset_gdino.py
------------------------
Prepares the dataset for fine-tuning Grounding DINO on annotation detection,
AND for evaluating Grounded SAM 2 on the same task (masks needed for eval).

Key properties:
  - One entry per ANNOTATED IMAGE (not per annotation). All annotations on
    a given image are grouped into a list.
  - Folder-level train/val/test split (no scene leakage across splits).
  - Annotation MASKS ARE COPIED so that downstream eval can compute mask IoU.
    GDINO training ignores them; eval uses them.
  - 3-class taxonomy: arrow / freeform_bbox / number_letter

Output structure:
    gdino_finetuning_dataset/
    ├── images/
    │   ├── 9988_0.jpg
    │   └── ...
    ├── masks/
    │   ├── 9988_0_0_0.png         <- annotation 0_0 on image 9988_0
    │   └── ...
    ├── train.json
    ├── val.json
    └── test.json

Each JSON entry:
    {
      "image":  "images/9988_0.jpg",
      "width":  500,
      "height": 333,
      "annotations": [
        {
          "box":         [x1, y1, x2, y2],
          "class":       "freeform_bbox",
          "description": "blue dashed line",
          "mask":        "masks/9988_0_0_0.png"
        }, ...
      ]
    }

Usage:
    python prepare_dataset_gdino.py
    python prepare_dataset_gdino.py --resume     # skip already-copied files
"""

import os
import json
import shutil
import argparse
import cv2
import numpy as np
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
RAW_DATA_DIR = "/home/ahma/unannotate/output"
OUTPUT_DIR   = "gdino_finetuning_dataset"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED  = 42
BBOX_PADDING = 10
NUM_WORKERS  = 16
MAX_FOLDERS  = None
# ───────────────────────────────────────────────────────────────────────────────


def fix_extension(folder_path, filename):
    base = os.path.splitext(filename)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder_path, base + ext)
        if os.path.exists(path):
            return path
    return None


def get_box_from_mask(mask, padding=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(pts)
    H, W = mask.shape
    return [
        int(max(0, x - padding)),
        int(max(0, y - padding)),
        int(min(W, x + w + padding)),
        int(min(H, y + h + padding)),
    ]


def coarse_annotation_type(description):
    desc = description.lower()
    if "arrow" in desc:
        return "arrow"
    if "dashed" in desc or "line" in desc:
        return "freeform_bbox"
    return "number_letter"


def parse_image_variant(annotated_img_name):
    base = os.path.splitext(os.path.basename(annotated_img_name))[0]
    parts = base.split("_")
    return parts[-1] if parts else base


def process_folder(args):
    folder_name, folder_path, images_dir, masks_dir, resume = args

    entries = []
    warnings = []

    labels_path = os.path.join(folder_path, "labels.json")
    if not os.path.exists(labels_path):
        return folder_name, entries, warnings

    try:
        with open(labels_path) as f:
            labels_data = json.load(f)
    except json.JSONDecodeError as e:
        warnings.append(f"[{folder_name}] Invalid JSON: {e}")
        return folder_name, entries, warnings

    by_variant = defaultdict(list)
    for key, value in labels_data.items():
        target_img_name = value.get("annotated_img_name")
        if not target_img_name:
            warnings.append(f"[{folder_name}/{key}] Missing annotated_img_name")
            continue
        variant_id = parse_image_variant(target_img_name)
        by_variant[variant_id].append((key, value, target_img_name))

    for variant_id, items in by_variant.items():
        target_img_name = items[0][2]
        image_path = fix_extension(folder_path, target_img_name)
        if image_path is None:
            warnings.append(f"[{folder_name}/v{variant_id}] Image not found: {target_img_name}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            warnings.append(f"[{folder_name}/v{variant_id}] Cannot read: {image_path}")
            continue
        H, W = img.shape[:2]

        new_image_name = f"{folder_name}_{variant_id}.jpg"
        dest_image = os.path.join(images_dir, new_image_name)
        if not (resume and os.path.exists(dest_image)):
            cv2.imwrite(dest_image, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        annotations = []
        copied_mask_paths = []
        for key, value, _ in items:
            src_mask_path = os.path.join(folder_path, f"annotation_mask_{key}.png")
            if not os.path.exists(src_mask_path):
                warnings.append(f"[{folder_name}/{key}] Missing mask: {src_mask_path}")
                continue

            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                warnings.append(f"[{folder_name}/{key}] Cannot read mask: {src_mask_path}")
                continue

            mask_bin = (mask > 127).astype(np.uint8)
            box = get_box_from_mask(mask_bin, padding=BBOX_PADDING)
            if box is None:
                warnings.append(f"[{folder_name}/{key}] Empty mask, skipping")
                continue

            new_mask_name = f"{folder_name}_{variant_id}_{key}.png"
            dest_mask_path = os.path.join(masks_dir, new_mask_name)
            if not (resume and os.path.exists(dest_mask_path)):
                shutil.copy(src_mask_path, dest_mask_path)
            copied_mask_paths.append(dest_mask_path)

            description = value.get("annotation", "unknown")
            cls = coarse_annotation_type(description)

            annotations.append({
                "box":         box,
                "class":       cls,
                "description": description,
                "mask":        f"masks/{new_mask_name}",
            })

        if not annotations:
            warnings.append(f"[{folder_name}/v{variant_id}] No valid annotations, skipping image")
            if os.path.exists(dest_image):
                os.remove(dest_image)
            for p in copied_mask_paths:
                if os.path.exists(p):
                    os.remove(p)
            continue

        entries.append({
            "image":       f"images/{new_image_name}",
            "width":       W,
            "height":      H,
            "folder":      folder_name,
            "annotations": annotations,
        })

    return folder_name, entries, warnings


def split_by_folder(entries_by_folder, ratios, seed):
    folders = sorted(entries_by_folder.keys())
    rng = random.Random(seed)
    rng.shuffle(folders)

    n_total = len(folders)
    n_train = int(n_total * ratios[0])
    n_val   = int(n_total * ratios[1])

    train_folders = set(folders[:n_train])
    val_folders   = set(folders[n_train : n_train + n_val])
    test_folders  = set(folders[n_train + n_val :])

    train, val, test = [], [], []
    for fname, fentries in entries_by_folder.items():
        if fname in train_folders:
            train.extend(fentries)
        elif fname in val_folders:
            val.extend(fentries)
        else:
            test.extend(fentries)

    return train, val, test, (len(train_folders), len(val_folders), len(test_folders))


def prepare_dataset(resume=False):
    images_dir = os.path.join(OUTPUT_DIR, "images")
    masks_dir  = os.path.join(OUTPUT_DIR, "masks")

    if not resume and os.path.exists(OUTPUT_DIR):
        print(f"Removing existing '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: '{RAW_DATA_DIR}' not found.")
        return

    folder_names = sorted([
        f for f in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, f))
    ])
    if MAX_FOLDERS is not None:
        folder_names = folder_names[:MAX_FOLDERS]
        print(f"Limiting to {MAX_FOLDERS} folders (MAX_FOLDERS is set)")

    print(f"Found {len(folder_names):,} folders in '{RAW_DATA_DIR}'")
    print(f"Processing with {NUM_WORKERS} parallel workers...\n")

    entries_by_folder = defaultdict(list)
    all_warnings = []

    worker_args = [
        (name, os.path.join(RAW_DATA_DIR, name), images_dir, masks_dir, resume)
        for name in folder_names
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_folder, a): a[0] for a in worker_args}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Processing folders", unit="folder")
        for future in pbar:
            fname, entries, warnings = future.result()
            if entries:
                entries_by_folder[fname].extend(entries)
            all_warnings.extend(warnings)
            pbar.set_postfix(images=sum(len(v) for v in entries_by_folder.values()),
                             warnings=len(all_warnings))

    if all_warnings:
        with open(os.path.join(OUTPUT_DIR, "warnings.txt"), "w") as f:
            f.write("\n".join(all_warnings))
        print(f"\n{len(all_warnings)} warnings written to {OUTPUT_DIR}/warnings.txt "
              f"(showing first 20):")
        for w in all_warnings[:20]:
            print(f"  {w}")

    train, val, test, folder_counts = split_by_folder(
        entries_by_folder, SPLIT_RATIOS, RANDOM_SEED
    )

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)

    total_imgs = len(train) + len(val) + len(test)
    total_anns = sum(len(e["annotations"]) for e in train + val + test)
    type_counts_imgs = Counter()
    type_counts_anns = Counter()
    anns_per_img = []
    for e in train + val + test:
        anns_per_img.append(len(e["annotations"]))
        types_on_img = set()
        for a in e["annotations"]:
            type_counts_anns[a["class"]] += 1
            types_on_img.add(a["class"])
        for t in types_on_img:
            type_counts_imgs[t] += 1

    print(f"\nDataset ready in '{OUTPUT_DIR}/'")
    print(f"  Folders   {sum(folder_counts):>8,}  "
          f"(train {folder_counts[0]:,} / val {folder_counts[1]:,} / test {folder_counts[2]:,})")
    print(f"  Images    {total_imgs:>8,}  "
          f"(train {len(train):,} / val {len(val):,} / test {len(test):,})")
    print(f"  Annot.    {total_anns:>8,}  "
          f"(avg {total_anns/total_imgs:.2f} per image, "
          f"max {max(anns_per_img)}, min {min(anns_per_img)})")
    print(f"\n  Annotation count per image:")
    apc = Counter(anns_per_img)
    for n in sorted(apc):
        print(f"    {n} annot/img : {apc[n]:>7,} images")
    print(f"\n  Class breakdown (annotations):")
    for t, c in sorted(type_counts_anns.items()):
        print(f"    {t:<16} {c:>8,}  ({100*c/total_anns:.1f}%)")
    print(f"\n  Class breakdown (images that contain at least one of class):")
    for t, c in sorted(type_counts_imgs.items()):
        print(f"    {t:<16} {c:>8,}  ({100*c/total_imgs:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed image and mask files")
    args = parser.parse_args()
    prepare_dataset(resume=args.resume)
