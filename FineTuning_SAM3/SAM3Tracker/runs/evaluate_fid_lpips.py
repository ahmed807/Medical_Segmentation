"""
evaluate_fid_lpips.py
=====================

Inpainting-quality evaluation restricted to the annotation region.

WHY REGION-CROPPED EVALUATION
-----------------------------
inference_tracker.py uses a hard composite: the FLUX output is taken only
inside the dilated annotation mask, and original-image pixels are kept
elsewhere. Two consequences for evaluation:

1. Outside the annotation mask, the predicted clean image is byte-identical
   to the input. Comparing whole images would swamp the inpainting signal
   under thousands of identical background pixels, and FID/LPIPS would
   reflect input similarity rather than inpainting quality.

2. A test image may contain other annotations besides the one being cleaned
   in this inference call. The original_clean_image has no annotations at
   all. Comparing the whole predicted image against the whole original
   would penalise the predictor for the residual annotations the current
   inference call did not touch.

The fix is to crop both the predicted clean image and the original clean
reference to the bounding box of the *current* annotation mask (with a
margin so the surrounding context is included), then compute FID and LPIPS
on those crops.

PIPELINE
--------
For every test entry:
  1. Load predicted clean = results_dir/<type>/<basename>_clean.png
  2. Load original clean  = data_root/original_clean_image
  3. Load annotation mask = data_root/annotation_mask
  4. Compute bounding box of annotation_mask + `--margin-px` padding,
     clipped to image bounds.
  5. Crop both predicted and original to that bbox.
  6. Resize both crops to (--crop-size, --crop-size) for fair FID/LPIPS.
  7. Save crops under out_dir/<type>/{predicted,original}/<basename>.png.

Per-sample LPIPS is computed on each crop pair as we go.
FID is computed at the end per annotation type (and overall) using
clean-fid on the saved crop folders.

DEPENDENCIES
------------
pip install numpy opencv-python pillow torch lpips clean-fid tqdm

USAGE
-----
python evaluate_fid_lpips.py \\
    --test-json    /path/to/test.json \\
    --data-root    /path/to/dataset_root \\
    --results-dir  /path/to/inference_results_full \\
    --out-dir      ./eval_fid_lpips \\
    --crop-size    256 \\
    --margin-px    16

OUTPUTS
-------
- out_dir/crops/<type>/predicted/<basename>.png
- out_dir/crops/<type>/original/<basename>.png
- out_dir/lpips_per_sample.csv
- out_dir/fid_lpips_summary.csv
- markdown table printed to stdout
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def annotation_bbox(mask: np.ndarray, margin: int) -> tuple:
    """
    Bounding box (x0, y0, x1, y1) of all foreground pixels in `mask`,
    expanded by `margin` pixels and clipped to image bounds.

    Returns None if the mask is empty.
    """
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


def crop_and_resize(img: np.ndarray, bbox: tuple, size: int) -> np.ndarray:
    """Crop `img` to `bbox` and resize to (size, size) with bilinear interp."""
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def basename_from_image_path(image_rel_path: str) -> str:
    return Path(image_rel_path).stem


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def init_lpips(device: str):
    """Returns a callable f(img_a, img_b) -> float for HxWx3 uint8 BGR inputs."""
    import lpips
    print(f"Loading LPIPS (AlexNet) on {device}...")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    def to_tensor(bgr_img: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb * 2.0 - 1.0  # LPIPS expects [-1, 1]
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        return t

    @torch.no_grad()
    def score(a: np.ndarray, b: np.ndarray) -> float:
        ta = to_tensor(a)
        tb = to_tensor(b)
        return float(loss_fn(ta, tb).item())

    return score


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(folder_a: Path, folder_b: Path, device: str) -> float:
    """Wrapper around clean-fid. Returns FID or NaN on failure."""
    try:
        from cleanfid import fid
    except ImportError:
        print("[ERROR] clean-fid not installed. Run: pip install clean-fid")
        return float("nan")

    a_imgs = list(folder_a.glob("*.png"))
    b_imgs = list(folder_b.glob("*.png"))
    if len(a_imgs) < 2 or len(b_imgs) < 2:
        return float("nan")

    return float(fid.compute_fid(
        str(folder_a), str(folder_b),
        mode="clean",
        device=device,
        num_workers=0,
        verbose=False,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    test_json_path: str,
    data_root: str,
    results_dir: str,
    out_dir: str,
    crop_size: int = 256,
    margin_px: int = 16,
    device: str = None,
    skip_fid: bool = False,
) -> None:
    test_json_path = Path(test_json_path)
    data_root = Path(data_root)
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(test_json_path) as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} test entries")
    print(f"Crop size: {crop_size}px, margin: {margin_px}px")
    print()

    lpips_fn = init_lpips(device)

    by_type = defaultdict(list)
    skipped_missing = 0
    skipped_empty_mask = 0

    per_sample_csv = out_dir / "lpips_per_sample.csv"
    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["basename", "annotation_type", "lpips"])

        for entry in tqdm(entries, desc="Cropping + LPIPS", unit="img"):
            ann_type = entry["annotation_type"]
            basename = basename_from_image_path(entry["image"])

            pred_path = results_dir / ann_type / f"{basename}_clean.png"
            orig_path = data_root / entry["original_clean_image"]
            mask_path = data_root / entry["annotation_mask"]

            if not (pred_path.exists() and orig_path.exists() and mask_path.exists()):
                skipped_missing += 1
                continue

            pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
            orig = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if pred is None or orig is None or mask is None:
                skipped_missing += 1
                continue

            # Align resolutions to the predicted clean (the canonical output).
            H, W = pred.shape[:2]
            if orig.shape[:2] != (H, W):
                orig = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LINEAR)
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            bbox = annotation_bbox(mask, margin=margin_px)
            if bbox is None:
                skipped_empty_mask += 1
                continue

            pred_crop = crop_and_resize(pred, bbox, crop_size)
            orig_crop = crop_and_resize(orig, bbox, crop_size)
            if pred_crop is None or orig_crop is None:
                skipped_empty_mask += 1
                continue

            # Save crops (used later by clean-fid).
            (crops_dir / ann_type / "predicted").mkdir(parents=True, exist_ok=True)
            (crops_dir / ann_type / "original").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crops_dir / ann_type / "predicted" / f"{basename}.png"), pred_crop)
            cv2.imwrite(str(crops_dir / ann_type / "original" / f"{basename}.png"), orig_crop)

            # Per-sample LPIPS.
            score = lpips_fn(pred_crop, orig_crop)
            by_type[ann_type].append(score)
            writer.writerow([basename, ann_type, f"{score:.4f}"])

    print()
    if skipped_missing:
        print(f"Note: skipped {skipped_missing} entries (missing files).")
    if skipped_empty_mask:
        print(f"Note: skipped {skipped_empty_mask} entries (empty annotation mask).")
    print()

    # ------------------- FID per type + overall -------------------
    fid_per_type = {}
    if not skip_fid:
        # Build "all" folders by symlinking everything together.
        all_pred = crops_dir / "_all" / "predicted"
        all_orig = crops_dir / "_all" / "original"
        all_pred.mkdir(parents=True, exist_ok=True)
        all_orig.mkdir(parents=True, exist_ok=True)
        for ann_type in by_type:
            for side, dst_root in (("predicted", all_pred), ("original", all_orig)):
                src = crops_dir / ann_type / side
                if not src.exists():
                    continue
                for img_path in src.glob("*.png"):
                    dst = dst_root / f"{ann_type}__{img_path.name}"
                    if not dst.exists():
                        try:
                            dst.symlink_to(img_path.resolve())
                        except (OSError, NotImplementedError):
                            # Fallback for filesystems without symlinks
                            import shutil
                            shutil.copy2(img_path, dst)

        print("Computing FID per annotation type...")
        for ann_type in sorted(by_type.keys()):
            pred_dir = crops_dir / ann_type / "predicted"
            orig_dir = crops_dir / ann_type / "original"
            fid_score = compute_fid(pred_dir, orig_dir, device)
            fid_per_type[ann_type] = fid_score
            print(f"  {ann_type:<16}: FID = {fid_score:.3f}")

        print("Computing overall FID...")
        fid_overall = compute_fid(all_pred, all_orig, device)
        fid_per_type["OVERALL"] = fid_overall
        print(f"  OVERALL         : FID = {fid_overall:.3f}")
        print()

    # ------------------- summary CSV + markdown -------------------
    summary_csv = out_dir / "fid_lpips_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "n", "lpips_mean", "lpips_median", "lpips_std", "fid",
        ])
        for ann_type in sorted(by_type.keys()):
            scores = by_type[ann_type]
            writer.writerow([
                ann_type,
                len(scores),
                f"{statistics.mean(scores):.4f}",
                f"{statistics.median(scores):.4f}",
                f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "",
                f"{fid_per_type.get(ann_type, float('nan')):.3f}" if not skip_fid else "",
            ])
        all_scores = [s for v in by_type.values() for s in v]
        if all_scores:
            writer.writerow([
                "OVERALL",
                len(all_scores),
                f"{statistics.mean(all_scores):.4f}",
                f"{statistics.median(all_scores):.4f}",
                f"{statistics.stdev(all_scores):.4f}" if len(all_scores) > 1 else "",
                f"{fid_per_type.get('OVERALL', float('nan')):.3f}" if not skip_fid else "",
            ])

    print("=" * 78)
    print("FID + LPIPS results (region-cropped)")
    print("=" * 78)
    print()
    print(f"| {'Type':<18} | {'N':>5} | {'LPIPS mean':>10} | {'LPIPS med':>9} | {'LPIPS std':>9} | {'FID':>8} |")
    print("|" + "-" * 76 + "|")
    for ann_type in sorted(by_type.keys()):
        scores = by_type[ann_type]
        fid_str = f"{fid_per_type.get(ann_type, float('nan')):.3f}" if not skip_fid else "—"
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"| {ann_type:<18} | {len(scores):>5} | "
            f"{statistics.mean(scores):>10.4f} | "
            f"{statistics.median(scores):>9.4f} | "
            f"{std:>9.4f} | {fid_str:>8} |"
        )
    if all_scores:
        std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        fid_str = f"{fid_per_type.get('OVERALL', float('nan')):.3f}" if not skip_fid else "—"
        print(
            f"| {'OVERALL':<18} | {len(all_scores):>5} | "
            f"{statistics.mean(all_scores):>10.4f} | "
            f"{statistics.median(all_scores):>9.4f} | "
            f"{std:>9.4f} | {fid_str:>8} |"
        )
    print()
    print(f"Per-sample LPIPS:  {per_sample_csv}")
    print(f"Summary:           {summary_csv}")
    print(f"Crops:             {crops_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-json", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", default="./eval_fid_lpips")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Square crop size for FID/LPIPS (default 256)")
    p.add_argument("--margin-px", type=int, default=16,
                   help="Padding around the annotation bbox (default 16)")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: auto)")
    p.add_argument("--skip-fid", action="store_true",
                   help="Skip FID, compute LPIPS only (faster, no clean-fid dep)")
    args = p.parse_args()
    evaluate(
        args.test_json, args.data_root, args.results_dir, args.out_dir,
        args.crop_size, args.margin_px, args.device, args.skip_fid,
    )


if __name__ == "__main__":
    main()
