"""
analyze_pipeline_b_buckets.py
=============================

Per-detection error-propagation analysis for Pipeline B (§6.5.2).

For each of the 18,532 Pipeline B detections, computes the IoU between
its predicted annotation mask (SAM-refined) and the best-matching
ground-truth annotation mask on the same source image. Joins that
matched-IoU column with the per-detection LPIPS file already produced
by evaluate_fid_lpips_pipeline_b.py, then produces:

  1. detection_iou_per_sample.csv       (the IoU column itself)
  2. bucketed_stats.csv                  (count + LPIPS per IoU bucket per class)
  3. correlations.csv                    (Pearson + Spearman r per class)
  4. bucket_crops/<bucket>/<type>/{predicted,original}/  (symlinks)

The symlink directories are drop-in inputs for compute_fid_from_crops.py.
After this script finishes, run compute_fid_from_crops.py four times
(once per bucket) to get the per-bucket FID numbers; the driver
shell script does that.

WHY RECOMPUTE INSTEAD OF READING raw_records.json
-------------------------------------------------
raw_records.json from the GroundingSAM evaluator likely already has
matched IoU baked in, but the matching it performed is for the
GroundingSAM mAP evaluation under specific thresholds. Recomputing
from masks directly removes any ambiguity about which threshold,
which match function, and which IoU convention produced our numbers.
The pred and GT masks themselves are unambiguous.

USAGE
-----
python analyze_pipeline_b_buckets.py \\
    --input-json   /home/ahma/Grounded-SAM-2/pipeline_b_input_default.json \\
    --test-json    /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset/test.json \\
    --sam3-root    /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset \\
    --lpips-csv    /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask/pipeline_b_default/eval_fid_lpips/lpips_per_sample.csv \\
    --crops-dir    /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask/pipeline_b_default/eval_fid_lpips/crops \\
    --orig-root-replace "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset:/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset" \\
    --out-dir      /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask/pipeline_b_default/error_propagation
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
from tqdm import tqdm


# --- IoU bucket boundaries -------------------------------------------------
# Four buckets chosen to separate "false positive" (effectively zero
# overlap) from three quality tiers of true overlap. Boundaries selected
# so that no bucket is too small for stable per-bucket FID estimation.
BUCKETS = [
    ("zero_overlap",     0.00, 0.05),
    ("low_iou",          0.05, 0.30),
    ("mid_iou",          0.30, 0.60),
    ("high_iou",         0.60, 1.01),
]


def bucket_of(iou: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= iou < hi:
            return name
    return BUCKETS[-1][0]


def binarize(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def apply_prefix_replace(p: str, rules: list) -> str:
    if not p or not rules:
        return p
    for old, new in rules:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def entry_id_of(entry: dict) -> str:
    if "source_image" in entry and "detection_idx" in entry:
        return f"{Path(entry['source_image']).stem}_det{entry['detection_idx']}"
    return Path(entry["image"]).stem


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    rx = ranks(xs)
    ry = ranks(ys)
    return pearson(rx, ry)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-json", required=True,
                    help="pipeline_b_input_default.json")
    ap.add_argument("--test-json", required=True,
                    help="SAM3Tracker test.json (GT annotation source)")
    ap.add_argument("--sam3-root", required=True,
                    help="SAM3Tracker dataset root (joined with relative "
                         "annotation_mask paths from test.json)")
    ap.add_argument("--lpips-csv", required=True,
                    help="Pipeline B lpips_per_sample.csv from "
                         "evaluate_fid_lpips_pipeline_b.py")
    ap.add_argument("--crops-dir", required=True,
                    help="Pipeline B eval_fid_lpips/crops dir (used as "
                         "symlink source for per-bucket FID)")
    ap.add_argument("--orig-root-replace", action="append", default=[],
                    metavar="OLD:NEW",
                    help="Prefix-rewrite the pred annotation_mask paths "
                         "in the input JSON (same flag as the LPIPS "
                         "evaluator).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_root = out_dir / "bucket_crops"
    bucket_root.mkdir(exist_ok=True)

    rules = []
    for spec in args.orig_root_replace:
        if ":" in spec:
            o, n = spec.split(":", 1)
            rules.append((o, n))

    # -------- index GT annotations by source-image stem --------
    with open(args.test_json) as f:
        gt_entries = json.load(f)
    print(f"Loaded {len(gt_entries)} GT entries from test.json")

    gt_by_source = defaultdict(list)
    for g in gt_entries:
        # unique_base format: "<src>_<idx>" e.g. "1002_0_1"
        # source stem in pipeline_b is "<folder>_<page>" e.g. "1002_0"
        ub = Path(g["image"]).stem  # g["unique_base"]
        # Strip the trailing annotation index: rsplit once on "_"
        src_stem = ub.rsplit("_", 1)[0]
        gt_by_source[src_stem].append(g)

    print(f"Indexed GT by {len(gt_by_source)} source stems")
    print(f"  GT per source: min={min(len(v) for v in gt_by_source.values())}, "
          f"max={max(len(v) for v in gt_by_source.values())}, "
          f"mean={sum(len(v) for v in gt_by_source.values())/len(gt_by_source):.2f}")

    # -------- compute per-detection matched IoU --------
    with open(args.input_json) as f:
        detections = json.load(f)
    print(f"Loaded {len(detections)} Pipeline B detections")

    iou_csv = out_dir / "detection_iou_per_sample.csv"
    iou_by_entry = {}
    type_by_entry = {}
    skipped = {"no_gt_for_source": 0,
               "pred_mask_missing": 0,
               "pred_mask_unreadable": 0}

    # GT-mask cache so we don't re-read the same file 5 times
    gt_cache = {}

    with open(iou_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entry_id", "true_annotation_type", "score",
            "matched_iou", "n_gt_in_image", "matched_unique_base",
        ])

        for det in tqdm(detections, desc="matched IoU", unit="det"):
            eid = entry_id_of(det)
            true_t = det.get("true_annotation_type",
                              det.get("annotation_type", "unknown"))
            score = det.get("score", float("nan"))
            src_stem = Path(det["source_image"]).stem

            pred_path = apply_prefix_replace(det["annotation_mask"], rules)
            if not os.path.exists(pred_path):
                skipped["pred_mask_missing"] += 1
                continue
            pred_raw = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred_raw is None:
                skipped["pred_mask_unreadable"] += 1
                continue
            pred = binarize(pred_raw)

            gt_list = gt_by_source.get(src_stem, [])
            if not gt_list:
                skipped["no_gt_for_source"] += 1
                continue

            best_iou, best_ub = 0.0, ""
            for g in gt_list:
                gt_rel = g["annotation_mask"]
                gt_path = os.path.join(args.sam3_root, gt_rel)
                if gt_path not in gt_cache:
                    raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if raw is None:
                        gt_cache[gt_path] = None
                        continue
                    gt_cache[gt_path] = binarize(raw)
                gt_bin = gt_cache[gt_path]
                if gt_bin is None:
                    continue
                v = iou(pred, gt_bin)
                if v > best_iou:
                    best_iou = v
                    best_ub = Path(g["image"]).stem #g["unique_base"]

            iou_by_entry[eid] = best_iou
            type_by_entry[eid] = true_t
            w.writerow([eid, true_t, f"{score:.4f}",
                        f"{best_iou:.4f}", len(gt_list), best_ub])

    print()
    for k, v in skipped.items():
        if v:
            print(f"  skipped[{k}] = {v}")
    print(f"  matched {len(iou_by_entry)} detections")

    # -------- join with LPIPS --------
    print()
    print(f"Reading LPIPS from {args.lpips_csv} ...")
    lpips_by_entry = {}
    with open(args.lpips_csv) as f:
        for row in csv.DictReader(f):
            lpips_by_entry[row["entry_id"]] = float(row["lpips"])

    joined = []  # (entry_id, type, iou, lpips, bucket)
    for eid, v_iou in iou_by_entry.items():
        if eid not in lpips_by_entry:
            continue
        joined.append({
            "entry_id": eid,
            "type": type_by_entry[eid],
            "iou": v_iou,
            "lpips": lpips_by_entry[eid],
            "bucket": bucket_of(v_iou),
        })
    print(f"  joined {len(joined)} detections "
          f"(have IoU AND LPIPS)")

    # -------- bucket statistics --------
    bucket_csv = out_dir / "bucketed_stats.csv"
    by_bucket_type = defaultdict(lambda: {"iou": [], "lpips": []})
    for r in joined:
        by_bucket_type[(r["bucket"], r["type"])]["iou"].append(r["iou"])
        by_bucket_type[(r["bucket"], r["type"])]["lpips"].append(r["lpips"])
    by_bucket_overall = defaultdict(lambda: {"iou": [], "lpips": []})
    for r in joined:
        by_bucket_overall[r["bucket"]]["iou"].append(r["iou"])
        by_bucket_overall[r["bucket"]]["lpips"].append(r["lpips"])

    with open(bucket_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bucket", "type", "n",
            "iou_mean", "iou_median",
            "lpips_mean", "lpips_median", "lpips_std",
        ])
        for (b_name, b_lo, b_hi) in BUCKETS:
            for t in ("arrow", "freeform_bbox", "number_letter"):
                d = by_bucket_type.get((b_name, t),
                                       {"iou": [], "lpips": []})
                if not d["lpips"]:
                    continue
                w.writerow([
                    b_name, t, len(d["lpips"]),
                    f"{statistics.mean(d['iou']):.4f}",
                    f"{statistics.median(d['iou']):.4f}",
                    f"{statistics.mean(d['lpips']):.4f}",
                    f"{statistics.median(d['lpips']):.4f}",
                    f"{statistics.stdev(d['lpips']):.4f}"
                    if len(d["lpips"]) > 1 else "",
                ])
            # overall row for bucket
            d = by_bucket_overall.get(b_name, {"iou": [], "lpips": []})
            if d["lpips"]:
                w.writerow([
                    b_name, "OVERALL", len(d["lpips"]),
                    f"{statistics.mean(d['iou']):.4f}",
                    f"{statistics.median(d['iou']):.4f}",
                    f"{statistics.mean(d['lpips']):.4f}",
                    f"{statistics.median(d['lpips']):.4f}",
                    f"{statistics.stdev(d['lpips']):.4f}"
                    if len(d["lpips"]) > 1 else "",
                ])

    # -------- correlations --------
    corr_csv = out_dir / "correlations.csv"
    by_type = defaultdict(lambda: {"iou": [], "lpips": []})
    for r in joined:
        by_type[r["type"]]["iou"].append(r["iou"])
        by_type[r["type"]]["lpips"].append(r["lpips"])
    with open(corr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "n", "pearson_r", "spearman_r"])
        for t in sorted(by_type):
            xs = by_type[t]["iou"]
            ys = by_type[t]["lpips"]
            w.writerow([t, len(xs),
                        f"{pearson(xs, ys):.4f}",
                        f"{spearman(xs, ys):.4f}"])
        xs = [r["iou"] for r in joined]
        ys = [r["lpips"] for r in joined]
        w.writerow(["OVERALL", len(xs),
                    f"{pearson(xs, ys):.4f}",
                    f"{spearman(xs, ys):.4f}"])

    # -------- bucket crop symlinks (for compute_fid_from_crops.py) --------
    print()
    print("Creating per-bucket crop symlinks ...")
    crops_src = Path(args.crops_dir)
    if not crops_src.exists():
        print(f"  WARNING: crops dir {crops_src} not found; skipping FID prep")
    else:
        for r in joined:
            b = r["bucket"]
            t = r["type"]
            eid = r["entry_id"]
            for side in ("predicted", "original"):
                src = crops_src / t / side / f"{eid}.png"
                if not src.exists():
                    continue
                dst_dir = bucket_root / b / "crops" / t / side
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{eid}.png"
                if not dst.exists():
                    os.symlink(src, dst)

        # mirror lpips_per_sample.csv into each bucket so compute_fid can
        # pick up the LPIPS column too
        for (b_name, _, _) in BUCKETS:
            bdir = bucket_root / b_name
            if not bdir.exists():
                continue
            bdir.mkdir(parents=True, exist_ok=True)
            mini_lpips = bdir / "lpips_per_sample.csv"
            with open(mini_lpips, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["entry_id", "annotation_type", "lpips"])
                for r in joined:
                    if r["bucket"] == b_name:
                        w.writerow([r["entry_id"], r["type"],
                                    f"{r['lpips']:.4f}"])

    # -------- console summary --------
    print()
    print("=" * 78)
    print("Pipeline B  --  Bucketed error-propagation analysis")
    print("=" * 78)
    print()
    print(f"| {'Bucket':<14} | {'Type':<14} | {'N':>6} | "
          f"{'IoU mean':>8} | {'LPIPS mean':>10} | {'LPIPS med':>9} |")
    print("|" + "-" * 76 + "|")
    for (b_name, b_lo, b_hi) in BUCKETS:
        for t in ("arrow", "freeform_bbox", "number_letter"):
            d = by_bucket_type.get((b_name, t),
                                   {"iou": [], "lpips": []})
            if not d["lpips"]:
                continue
            print(f"| {b_name:<14} | {t:<14} | {len(d['lpips']):>6} | "
                  f"{statistics.mean(d['iou']):>8.4f} | "
                  f"{statistics.mean(d['lpips']):>10.4f} | "
                  f"{statistics.median(d['lpips']):>9.4f} |")
        d = by_bucket_overall.get(b_name, {"iou": [], "lpips": []})
        if d["lpips"]:
            print(f"| {b_name:<14} | {'OVERALL':<14} | "
                  f"{len(d['lpips']):>6} | "
                  f"{statistics.mean(d['iou']):>8.4f} | "
                  f"{statistics.mean(d['lpips']):>10.4f} | "
                  f"{statistics.median(d['lpips']):>9.4f} |")
        print("|" + "-" * 76 + "|")

    print()
    print(f"detection_iou       : {iou_csv}")
    print(f"bucket stats        : {bucket_csv}")
    print(f"correlations        : {corr_csv}")
    print(f"bucket crop symlinks: {bucket_root}")
    print()
    print("Next: for each bucket dir under bucket_crops/, run")
    print("    python compute_fid_from_crops.py "
          "--eval-dir bucket_crops/<bucket>")
    print("to fill in the per-bucket FID. The driver script does this.")


if __name__ == "__main__":
    main()
