"""
build_pipeline_b_input.py
-------------------------
Convert GroundedSAM 2 predictions into the per-(image, annotation) sam3-style
JSON format that the existing SAM3Tracker inference pipeline consumes.

Input  : predictions.json + pred_masks/   (produced by zero_shot_eval_v2_with_persistence.py)
Output : pipeline_b_input.json

Each output entry corresponds to ONE detected annotation in ONE image. Per
image with N detections, N output entries are emitted — matching how
sam3_test.json / test.json were structured for training and inference.

Per-class score thresholding is applied here: only detections whose score
exceeds the per-class threshold survive into the output. Defaults match the
recommendations from the threshold sweep:

    --thresh-arrow         0.25
    --thresh-freeform      0.30   (raises precision 33% -> 53%, drops recall ~7pp)
    --thresh-number_letter 0.25   (no operating point usefully recovers this class)

Override any of these on the CLI for the per-class-tuned variant.

Output format (matches sam3_test.json contract):

    {
        "image":                  "images/1002_0.jpg",
        "annotation_mask":        "<predictions_dir>/pred_masks/1002_0_3_sam.png",
        "original_clean_image":   "originals/1002_original.png",
        "prompt_box":             [x1, y1, x2, y2],
        "prompt_text":            "arrow",       # mapped from class for SAM3Tracker
        "label":                  "",
        "annotation_description": "predicted",
        "annotation_type":        "arrow",
        "score":                  0.297,
        "source_image":           "images/1002_0.jpg",
        "detection_idx":          3
    }

Two extra fields are added beyond the GT format ("score" and "detection_idx")
to support per-detection error analysis later. They are ignored by the
existing inference pipeline.

Usage
-----
    python3 build_pipeline_b_input.py \
        --predictions-dir /path/to/zero_shot_v2_full_promptD \
        --gdino-dataset-root /path/to/GroundingSAM/gdino_finetuning_dataset \
        --out pipeline_b_input.json

    # Per-class tuned thresholds (precision-leaning):
    python3 build_pipeline_b_input.py \
        --predictions-dir /path/to/zero_shot_v2_full_promptD \
        --gdino-dataset-root /path/to/GroundingSAM/gdino_finetuning_dataset \
        --thresh-arrow 0.25 \
        --thresh-freeform 0.30 \
        --thresh-number_letter 0.45 \
        --out pipeline_b_input_tuned.json
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


# Mapping from the predicted class label to the prompt_text expected by
# SAM3Tracker's inference code. Keep in sync with how sam3_test.json was built.
CLASS_TO_PROMPT_TEXT = {
    "arrow":         "arrow",
    "freeform_bbox": "dashed line",
    "number_letter": "letter or number",
}


def derive_original_clean_image(folder):
    """
    Derive original_clean_image from the gdino test entry's folder field.
    Confirmed deterministic over all 4814 unique folders in sam3_test.json:
    every original is `originals/{folder}_original.png`.
    """
    return f"originals/{folder}_original.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", required=True,
                    help="Directory containing predictions.json and pred_masks/")
    ap.add_argument("--gdino-dataset-root", required=True,
                    help="Path to gdino_finetuning_dataset (used to resolve absolute "
                         "paths for the annotated input images).")
    ap.add_argument("--sam3-dataset-root", required=True,
                    help="Path to sam_finetuning_dataset (used to resolve absolute "
                         "paths for the originals/ clean reference images, which "
                         "only exist under the sam3 dataset).")
    ap.add_argument("--gdino-test-json", default=None,
                    help="Optional fallback: gdino_test.json to look up folder by image")
    ap.add_argument("--out", required=True,
                    help="Output sam3-format JSON path")
    ap.add_argument("--thresh-arrow",         type=float, default=0.25)
    ap.add_argument("--thresh-freeform",      type=float, default=0.30)
    ap.add_argument("--thresh-number_letter", type=float, default=0.25)
    args = ap.parse_args()

    pred_dir = Path(args.predictions_dir).resolve()
    gdino_root = Path(args.gdino_dataset_root).resolve()
    sam3_root  = Path(args.sam3_dataset_root).resolve()
    preds_json = pred_dir / "predictions.json"
    if not preds_json.exists():
        raise FileNotFoundError(
            f"Expected predictions.json at {preds_json}. Did you run the patched "
            f"zero_shot_eval_v2_with_persistence.py with --save-predictions?"
        )
    with open(preds_json) as f:
        predictions = json.load(f)

    # Folder fallback if some entries lack the field
    folder_lookup = {}
    if args.gdino_test_json and os.path.exists(args.gdino_test_json):
        with open(args.gdino_test_json) as f:
            gdino_test = json.load(f)
        for g in gdino_test:
            folder_lookup[g["image"]] = g.get("folder")
        print(f"[fallback] loaded folder lookup for {len(folder_lookup)} images "
              f"from {args.gdino_test_json}")

    thresh = {
        "arrow":         args.thresh_arrow,
        "freeform_bbox": args.thresh_freeform,
        "number_letter": args.thresh_number_letter,
    }
    print(f"[thresholds] {thresh}")

    out_entries = []
    kept_per_class = Counter()
    dropped_per_class = Counter()
    skipped_no_folder = 0
    skipped_unknown_class = 0

    for img_entry in predictions:
        folder = img_entry.get("folder")
        if folder is None:
            folder = folder_lookup.get(img_entry["image"])
        if folder is None:
            skipped_no_folder += 1
            continue

        # Absolute paths - dataset_dir at inference time is a no-op for these.
        # image lives under gdino dataset; originals live ONLY under sam3 dataset.
        abs_image    = str(gdino_root / img_entry["image"])
        abs_original = str(sam3_root / f"originals/{folder}_original.png")

        for det in img_entry["predictions"]:
            cls = det["class"]
            if cls not in CLASS_TO_PROMPT_TEXT:
                skipped_unknown_class += 1
                continue

            score = det["score"]
            if score < thresh[cls]:
                dropped_per_class[cls] += 1
                continue

            # mask_sam_path is stored relative to predictions-dir; absolutize it.
            mask_rel  = det["mask_sam_path"]
            mask_full = (str(pred_dir / mask_rel)
                         if not os.path.isabs(mask_rel) else mask_rel)

            out_entries.append({
                "image":                  abs_image,
                "annotation_mask":        mask_full,
                "original_clean_image":   abs_original,
                "prompt_box":             [int(round(v)) for v in det["box"]],
                "prompt_text":            CLASS_TO_PROMPT_TEXT[cls],
                "label":                  "",
                "annotation_description": "predicted",
                "annotation_type":        cls,
                "score":                  score,
                "source_image":           img_entry["image"],
                "detection_idx":          det["idx"],
            })
            kept_per_class[cls] += 1

    print()
    print("=" * 70)
    print(f"Pipeline B input built : {args.out}")
    print(f"Total entries          : {len(out_entries)}")
    print("-" * 70)
    print("Per-class kept (passes threshold):")
    for c in ["arrow", "freeform_bbox", "number_letter"]:
        k = kept_per_class[c]
        d = dropped_per_class[c]
        total = k + d
        rate = (k / total * 100) if total else 0.0
        print(f"  {c:<16} kept={k:5d}  dropped={d:5d}  ({rate:.1f}% of detections kept)")
    if skipped_no_folder:
        print(f"  Skipped (no folder field, no fallback): {skipped_no_folder}")
    if skipped_unknown_class:
        print(f"  Skipped (unknown class label):          {skipped_unknown_class}")
    print("=" * 70)

    with open(args.out, "w") as f:
        json.dump(out_entries, f, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
