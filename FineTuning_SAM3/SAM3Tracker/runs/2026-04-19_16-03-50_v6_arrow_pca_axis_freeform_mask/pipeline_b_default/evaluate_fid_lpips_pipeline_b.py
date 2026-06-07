"""
evaluate_fid_lpips_pipeline_b.py
================================

FID + LPIPS evaluation for the Pipeline B outputs.

WHY A SEPARATE SCRIPT
---------------------
Pipeline B differs from Pipeline A in two ways that break the original
evaluate_fid_lpips.py:

  1. Entry-ID convention. Pipeline A writes <stem>_clean.png where stem
     is the test.json image-field stem. Pipeline B writes
     <source_stem>_det<detection_idx>_clean.png, because several
     detections can come from the same source image and would otherwise
     collide on disk (see inference_pipeline_b.py run_batch, the
     "source_image"/"detection_idx" branch).

  2. No ground-truth object mask. Pipeline B detections have no matching
     GT segmap, so IoU / Boundary F1 against GT are undefined. Only the
     inpainting metrics (FID, LPIPS) are well-defined, because they
     compare the predicted clean image against the original clean
     reference, which every entry still carries.

This script therefore computes ONLY FID + LPIPS, and reconstructs the
Pipeline B entry_id exactly as the inference script did so the predicted
clean files are found.

It reuses the same crop-and-compare logic as evaluate_fid_lpips.py
(annotation-bbox crop + margin, resize, per-sample LPIPS, crops saved
for the torchvision-FID pass) so the numbers are directly comparable to
Pipeline A as long as compute_fid_from_crops.py is used for FID on both.

DEPENDENCIES
------------
pip install numpy opencv-python pillow torch lpips tqdm

USAGE
-----
python evaluate_fid_lpips_pipeline_b.py \\
    --input-json   /abs/path/pipeline_b_input_default.json \\
    --results-dir  /abs/.../pipeline_b_default \\
    --out-dir      /abs/.../pipeline_b_default/eval_fid_lpips \\
    --crop-size    256 \\
    --margin-px    16

Then (same as Pipeline A, network-safe FID):
python compute_fid_from_crops.py \\
    --eval-dir /abs/.../pipeline_b_default/eval_fid_lpips
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Crop helpers (identical semantics to evaluate_fid_lpips.py)
# ---------------------------------------------------------------------------

def annotation_bbox(mask: np.ndarray, margin: int):
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 127
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    H, W = binary.shape
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, W)
    y1 = min(int(ys.max()) + 1 + margin, H)
    return (x0, y0, x1, y1)


def crop_and_resize(img, bbox, size):
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Pipeline B entry_id — reconstruct exactly as inference_pipeline_b.py did
# ---------------------------------------------------------------------------

def pipeline_b_entry_id(entry: dict) -> str:
    """
    Mirrors run_batch():
      if source_image and detection_idx present:
          entry_id = <source_image stem>_det<detection_idx>
      else:
          entry_id = <image basename stem>
    """
    if "source_image" in entry and "detection_idx" in entry:
        src_stem = Path(entry["source_image"]).stem
        return f"{src_stem}_det{entry['detection_idx']}"
    return Path(entry["image"]).stem


def resolve(path_field: str, base: str) -> str:
    """
    The Pipeline B converter writes absolute paths, so os.path.join with a
    base is a no-op for absolute inputs (Python keeps the absolute side).
    This matches the inference script's os.path.join(dataset_dir, ...).
    """
    if path_field is None:
        return None
    return os.path.join(base, path_field)


def apply_prefix_replace(p: str, rules: list) -> str:
    """
    Apply --orig-root-replace OLD:NEW substitutions to an absolute path.
    Matches as a leading-prefix replacement (not arbitrary substring) so a
    typo in OLD does not silently rewrite the middle of a path.
    """
    if not p or not rules:
        return p
    for old, new in rules:
        if p.startswith(old):
            return new + p[len(old):]
    return p


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def init_lpips(device: str):
    import lpips
    print(f"Loading LPIPS (AlexNet) on {device}...")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb * 2.0 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    @torch.no_grad()
    def score(a, b):
        return float(loss_fn(to_tensor(a), to_tensor(b)).item())

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-json", required=True,
                    help="pipeline_b_input_*.json fed to inference")
    ap.add_argument("--results-dir", required=True,
                    help="pipeline_b_default dir (contains <type>/ subdirs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-dir", default="",
                    help="Base for any relative paths in the JSON. "
                         "Pipeline B paths are absolute so this is "
                         "normally left empty.")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--margin-px", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--orig-root-replace", action="append", default=[],
                    metavar="OLD:NEW",
                    help="Rewrite a leading path prefix on the reference "
                         "fields (original_clean_image, annotation_mask). "
                         "Use when the JSON was written with a stale "
                         "dataset root. May be passed multiple times.")
    args = ap.parse_args()

    # Parse OLD:NEW substitutions.
    replace_rules = []
    for spec in args.orig_root_replace:
        if ":" not in spec:
            raise SystemExit(
                f"--orig-root-replace expects OLD:NEW, got {spec!r}")
        old, new = spec.split(":", 1)
        replace_rules.append((old, new))
    if replace_rules:
        print("Prefix rewrites:")
        for o, n in replace_rules:
            print(f"  {o}  ->  {n}")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.input_json) as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} Pipeline B entries from "
          f"{Path(args.input_json).name}")
    print(f"Crop size {args.crop_size}px, margin {args.margin_px}px")

    lpips_fn = init_lpips(device)

    by_type = defaultdict(list)
    skipped_missing = 0
    skipped_empty = 0
    skipped_no_clean = 0

    per_sample_csv = out_dir / "lpips_per_sample.csv"
    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_id", "annotation_type", "lpips"])

        for entry in tqdm(entries, desc="Cropping + LPIPS", unit="img"):
            ann_type = entry.get("annotation_type", "arrow")
            eid = pipeline_b_entry_id(entry)

            pred_path = results_dir / ann_type / f"{eid}_clean.png"
            orig_path = apply_prefix_replace(
                resolve(entry.get("original_clean_image"), args.dataset_dir),
                replace_rules)
            mask_path = apply_prefix_replace(
                resolve(entry.get("annotation_mask"), args.dataset_dir),
                replace_rules)

            # _clean.png only exists if FLUX ran for this entry.
            if not pred_path.exists():
                skipped_no_clean += 1
                continue
            if not (orig_path and os.path.exists(orig_path)
                    and mask_path and os.path.exists(mask_path)):
                skipped_missing += 1
                continue

            pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
            orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if pred is None or orig is None or mask is None:
                skipped_missing += 1
                continue

            H, W = pred.shape[:2]
            if orig.shape[:2] != (H, W):
                orig = cv2.resize(orig, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            bbox = annotation_bbox(mask, margin=args.margin_px)
            if bbox is None:
                skipped_empty += 1
                continue
            pc = crop_and_resize(pred, bbox, args.crop_size)
            oc = crop_and_resize(orig, bbox, args.crop_size)
            if pc is None or oc is None:
                skipped_empty += 1
                continue

            (crops_dir / ann_type / "predicted").mkdir(
                parents=True, exist_ok=True)
            (crops_dir / ann_type / "original").mkdir(
                parents=True, exist_ok=True)
            cv2.imwrite(
                str(crops_dir / ann_type / "predicted" / f"{eid}.png"), pc)
            cv2.imwrite(
                str(crops_dir / ann_type / "original" / f"{eid}.png"), oc)

            s = lpips_fn(pc, oc)
            by_type[ann_type].append(s)
            writer.writerow([eid, ann_type, f"{s:.4f}"])

    print()
    if skipped_no_clean:
        print(f"Note: {skipped_no_clean} entries had no _clean.png "
              f"(FLUX did not run / entry not inpainted).")
    if skipped_missing:
        print(f"Note: {skipped_missing} entries skipped (missing "
              f"original/mask file or unreadable).")
    if skipped_empty:
        print(f"Note: {skipped_empty} entries skipped (empty mask "
              f"or empty crop).")

    # summary CSV (FID column left blank; fill via compute_fid_from_crops.py)
    summary_csv = out_dir / "fid_lpips_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "n", "lpips_mean", "lpips_median",
                    "lpips_std", "fid"])
        for t in sorted(by_type):
            sc = by_type[t]
            w.writerow([
                t, len(sc),
                f"{statistics.mean(sc):.4f}",
                f"{statistics.median(sc):.4f}",
                f"{statistics.stdev(sc):.4f}" if len(sc) > 1 else "",
                "",
            ])
        allsc = [s for v in by_type.values() for s in v]
        if allsc:
            w.writerow([
                "OVERALL", len(allsc),
                f"{statistics.mean(allsc):.4f}",
                f"{statistics.median(allsc):.4f}",
                f"{statistics.stdev(allsc):.4f}" if len(allsc) > 1 else "",
                "",
            ])

    print()
    print("=" * 70)
    print("Pipeline B  —  LPIPS (region-cropped). FID pending compute_fid.")
    print("=" * 70)
    print(f"| {'Type':<16} | {'N':>6} | {'LPIPS mean':>10} | "
          f"{'LPIPS med':>9} |")
    print("|" + "-" * 52 + "|")
    for t in sorted(by_type):
        sc = by_type[t]
        print(f"| {t:<16} | {len(sc):>6} | "
              f"{statistics.mean(sc):>10.4f} | "
              f"{statistics.median(sc):>9.4f} |")
    if allsc:
        print(f"| {'OVERALL':<16} | {len(allsc):>6} | "
              f"{statistics.mean(allsc):>10.4f} | "
              f"{statistics.median(allsc):>9.4f} |")
    print()
    print(f"Per-sample LPIPS : {per_sample_csv}")
    print(f"Summary          : {summary_csv}")
    print(f"Crops            : {crops_dir}")
    print()
    print("Next: python compute_fid_from_crops.py --eval-dir "
          f"{out_dir}")


if __name__ == "__main__":
    main()
