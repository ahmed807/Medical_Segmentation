"""
evaluate_boundary_f1.py
=======================

Computes per-sample IoU and Boundary F1 between predicted SAM3Tracker object
masks and ground-truth object masks, broken down by annotation type.

This script extends the IoU numbers already produced by inference_tracker.py
with Boundary F1 (also called BF score) — a metric that captures the quality
of the predicted boundary independently of region overlap. A mask can have
high IoU but ragged or shifted boundaries; Boundary F1 measures that
explicitly.

Reference: Csurka et al., "What is a good evaluation measure for semantic
segmentation?" BMVC 2013.

Boundary F1 algorithm
---------------------
1. Extract one-pixel boundary from each binary mask using morphological
   gradient (dilation − erosion).
2. For each predicted boundary pixel, count it as a true positive if there
   is a GT boundary pixel within `tolerance_px` pixels (Euclidean distance).
3. Precision = TP / |predicted boundary pixels|
4. Recall    = TP / |GT boundary pixels|
5. F1        = 2 · P · R / (P + R)

Distance is computed via SciPy's exact distance transform — no approximation.

Inputs
------
- test.json      : list of dicts with keys {image, annotation, annotation_type}
- dataset root   : directory containing the relative paths in test.json
- results dir    : the inference_results_full/ directory containing
                   <type>/<basename>_object_mask.png

Outputs
-------
- per-sample CSV : one row per test sample with iou, boundary_f1, precision,
                   recall, type, basename
- summary CSV    : one row per annotation type with mean/median/std for both
                   metrics, plus an "overall" row
- prints a markdown summary table to stdout (paste-ready for the thesis)

Usage
-----
pip install numpy scipy opencv-python pillow tqdm
python evaluate_boundary_f1.py \\
    --test-json     /path/to/test.json \\
    --data-root     /path/to/dataset_root \\
    --results-dir   /path/to/inference_results_full \\
    --out-dir       ./eval_outputs \\
    --tolerance-px  2
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def binarize(mask: np.ndarray) -> np.ndarray:
    """Convert any mask to {0, 1} uint8."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard binary IoU."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide boundary via morphological gradient."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    return ((dil - ero) > 0).astype(np.uint8)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol_px: int = 2) -> dict:
    """
    Boundary F1 with a `tol_px` distance tolerance.

    A predicted boundary pixel is correct if it lies within `tol_px` of a
    GT boundary pixel; analogously for the recall side.
    """
    pred_b = boundary_pixels(pred)
    gt_b = boundary_pixels(gt)

    n_pred = int(pred_b.sum())
    n_gt = int(gt_b.sum())

    # Edge cases: empty boundaries
    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_pred == 0 or n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Distance from every pixel to the nearest "1" of the OTHER boundary.
    # distance_transform_edt(0 == foreground convention) — so we invert.
    dt_gt = distance_transform_edt(1 - gt_b)
    dt_pred = distance_transform_edt(1 - pred_b)

    # Precision: predicted boundary points within tolerance of GT boundary
    tp_pred = int(((pred_b == 1) & (dt_gt <= tol_px)).sum())
    # Recall: GT boundary points within tolerance of predicted boundary
    tp_gt = int(((gt_b == 1) & (dt_pred <= tol_px)).sum())

    precision = tp_pred / n_pred
    recall = tp_gt / n_gt
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def basename_from_image_path(image_rel_path: str) -> str:
    """images/679_1_1.jpg -> 679_1_1"""
    return Path(image_rel_path).stem


def evaluate(
    test_json_path: str,
    data_root: str,
    results_dir: str,
    out_dir: str,
    tolerance_px: int = 2,
) -> None:
    test_json_path = Path(test_json_path)
    data_root = Path(data_root)
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(test_json_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} test entries from {test_json_path.name}")
    print(f"Tolerance: {tolerance_px}px")
    print()

    per_sample_csv = out_dir / "boundary_f1_per_sample.csv"
    summary_csv = out_dir / "boundary_f1_summary.csv"

    by_type = defaultdict(lambda: {"iou": [], "f1": [], "precision": [], "recall": []})
    skipped = 0

    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "basename", "annotation_type", "iou", "boundary_f1",
            "precision", "recall",
        ])

        for entry in tqdm(entries, desc="Evaluating", unit="img"):
            ann_type = entry["annotation_type"]
            basename = basename_from_image_path(entry["image"])

            gt_path = data_root / entry["annotation"]
            pred_path = results_dir / ann_type / f"{basename}_object_mask.png"

            if not gt_path.exists() or not pred_path.exists():
                skipped += 1
                continue

            gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred_raw = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            if gt_raw is None or pred_raw is None:
                skipped += 1
                continue

            # Align resolution: predicted mask is at original image resolution
            # (inference resizes back), but if for any reason they differ,
            # resize predicted to GT to make IoU meaningful.
            if pred_raw.shape != gt_raw.shape:
                pred_raw = cv2.resize(
                    pred_raw, (gt_raw.shape[1], gt_raw.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            gt = binarize(gt_raw)
            pred = binarize(pred_raw)

            iou_val = iou(pred, gt)
            bf = boundary_f1(pred, gt, tol_px=tolerance_px)

            writer.writerow([
                basename, ann_type,
                f"{iou_val:.4f}", f"{bf['f1']:.4f}",
                f"{bf['precision']:.4f}", f"{bf['recall']:.4f}",
            ])

            by_type[ann_type]["iou"].append(iou_val)
            by_type[ann_type]["f1"].append(bf["f1"])
            by_type[ann_type]["precision"].append(bf["precision"])
            by_type[ann_type]["recall"].append(bf["recall"])

    # ----------------------- summary -----------------------
    print()
    if skipped:
        print(f"Note: skipped {skipped} entries with missing files.")
        print()

    def stats(arr):
        if not arr:
            return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "n": len(arr),
            "mean": statistics.mean(arr),
            "median": statistics.median(arr),
            "std": statistics.stdev(arr) if len(arr) > 1 else 0.0,
        }

    # All types
    all_iou, all_f1, all_p, all_r = [], [], [], []
    for v in by_type.values():
        all_iou.extend(v["iou"])
        all_f1.extend(v["f1"])
        all_p.extend(v["precision"])
        all_r.extend(v["recall"])

    rows = []
    for ann_type in sorted(by_type.keys()):
        v = by_type[ann_type]
        rows.append({
            "type": ann_type,
            "n": len(v["iou"]),
            "iou_mean": statistics.mean(v["iou"]),
            "iou_median": statistics.median(v["iou"]),
            "f1_mean": statistics.mean(v["f1"]),
            "f1_median": statistics.median(v["f1"]),
            "precision_mean": statistics.mean(v["precision"]),
            "recall_mean": statistics.mean(v["recall"]),
        })

    rows.append({
        "type": "OVERALL (micro)",
        "n": len(all_iou),
        "iou_mean": statistics.mean(all_iou) if all_iou else 0.0,
        "iou_median": statistics.median(all_iou) if all_iou else 0.0,
        "f1_mean": statistics.mean(all_f1) if all_f1 else 0.0,
        "f1_median": statistics.median(all_f1) if all_f1 else 0.0,
        "precision_mean": statistics.mean(all_p) if all_p else 0.0,
        "recall_mean": statistics.mean(all_r) if all_r else 0.0,
    })

    if by_type:
        per_type_iou_means = [statistics.mean(v["iou"]) for v in by_type.values()]
        per_type_f1_means = [statistics.mean(v["f1"]) for v in by_type.values()]
        rows.append({
            "type": "OVERALL (macro)",
            "n": len(all_iou),
            "iou_mean": statistics.mean(per_type_iou_means),
            "iou_median": float("nan"),
            "f1_mean": statistics.mean(per_type_f1_means),
            "f1_median": float("nan"),
            "precision_mean": float("nan"),
            "recall_mean": float("nan"),
        })

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "n", "iou_mean", "iou_median", "f1_mean", "f1_median",
            "precision_mean", "recall_mean",
        ])
        for r in rows:
            writer.writerow([
                r["type"], r["n"],
                f"{r['iou_mean']:.4f}",
                f"{r['iou_median']:.4f}" if r["iou_median"] == r["iou_median"] else "",
                f"{r['f1_mean']:.4f}",
                f"{r['f1_median']:.4f}" if r["f1_median"] == r["f1_median"] else "",
                f"{r['precision_mean']:.4f}" if r["precision_mean"] == r["precision_mean"] else "",
                f"{r['recall_mean']:.4f}" if r["recall_mean"] == r["recall_mean"] else "",
            ])

    # markdown summary
    print("=" * 78)
    print(f"Boundary F1 results (tolerance = {tolerance_px}px)")
    print("=" * 78)
    print()
    header = f"| {'Type':<18} | {'N':>5} | {'IoU mean':>9} | {'IoU med':>8} | {'BF1 mean':>9} | {'BF1 med':>8} | {'Precision':>9} | {'Recall':>7} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    print(header)
    print(sep)
    for r in rows:
        med_iou = f"{r['iou_median']:.4f}" if r["iou_median"] == r["iou_median"] else "—"
        med_f1 = f"{r['f1_median']:.4f}" if r["f1_median"] == r["f1_median"] else "—"
        prec = f"{r['precision_mean']:.4f}" if r["precision_mean"] == r["precision_mean"] else "—"
        rec = f"{r['recall_mean']:.4f}" if r["recall_mean"] == r["recall_mean"] else "—"
        print(
            f"| {r['type']:<18} | {r['n']:>5} | {r['iou_mean']:>9.4f} | "
            f"{med_iou:>8} | {r['f1_mean']:>9.4f} | {med_f1:>8} | "
            f"{prec:>9} | {rec:>7} |"
        )
    print()
    print(f"Per-sample results: {per_sample_csv}")
    print(f"Summary:            {summary_csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-json", required=True, help="Path to test.json")
    p.add_argument("--data-root", required=True,
                   help="Root dir for relative paths in test.json")
    p.add_argument("--results-dir", required=True,
                   help="Path to inference_results_full/")
    p.add_argument("--out-dir", default="./eval_boundary_f1",
                   help="Where to write CSVs (default: ./eval_boundary_f1)")
    p.add_argument("--tolerance-px", type=int, default=2,
                   help="Boundary F1 distance tolerance in pixels (default 2)")
    args = p.parse_args()
    evaluate(args.test_json, args.data_root, args.results_dir,
             args.out_dir, args.tolerance_px)


if __name__ == "__main__":
    main()
