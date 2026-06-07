"""
Per-class threshold sweep from a zero-shot GroundedSAM 2 run's raw_records.json.

The persistence script (zero_shot_eval_v2_with_persistence.py) writes raw_records.json
containing one record per (prediction OR missed GT) with fields:
    class, score, matched (bool), missed (bool), box_iou, mask_iou_sam, mask_iou_color
This script reads those records, filters predictions by score threshold per class,
and recomputes precision and recall at each threshold. No re-running of GroundedSAM 2
or SAM 2 is required.

Note on validity: the original eval does greedy box-IoU matching in score-descending
order. Raising the threshold removes low-score predictions only, so previously matched
high-score predictions remain matched. Matched count is therefore monotonically
non-increasing in threshold. Precision can go up (matched stays, n_pred drops faster)
or down (matched drops faster than n_pred); recall can only decrease.

Usage:
    python3 threshold_sweep_from_records.py \
        --records /path/to/zero_shot_v2_full_promptD/raw_records.json \
        --out threshold_sweep.csv

    # Default sweep is {0.25, 0.30, 0.35, 0.40, 0.45} for all classes; use --thresholds
    # to pass a custom comma-separated list (e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50").
"""
import argparse
import csv
import json
from collections import defaultdict


def load_records(path):
    with open(path) as f:
        return json.load(f)


def gt_count(records, cls):
    """Total ground-truth annotations of class `cls`: records that are
    either matched (a prediction that matched this GT) or missed (a GT that
    no prediction matched). Both are counted exactly once per GT."""
    n = 0
    for r in records:
        if r.get("class") != cls:
            continue
        if r.get("matched") is True or r.get("missed") is True:
            n += 1
    return n


def sweep_class(records, cls, thresholds):
    """Yield (threshold, n_pred, n_matched, precision, recall, f1) for class cls."""
    preds = [r for r in records if r.get("class") == cls and r.get("score") is not None]
    n_gt = gt_count(records, cls)
    for t in thresholds:
        filtered = [r for r in preds if r["score"] >= t]
        n_pred = len(filtered)
        n_matched = sum(1 for r in filtered if r["matched"])
        precision = (n_matched / n_pred) if n_pred else 0.0
        recall = (n_matched / n_gt) if n_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        yield t, n_pred, n_matched, precision, recall, f1, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to raw_records.json")
    ap.add_argument("--out", default=None, help="Optional output CSV path")
    ap.add_argument(
        "--thresholds",
        default="0.25,0.30,0.35,0.40,0.45",
        help="Comma-separated thresholds to sweep",
    )
    ap.add_argument(
        "--classes",
        default="arrow,freeform_bbox,number_letter",
        help="Comma-separated class names",
    )
    args = ap.parse_args()

    records = load_records(args.records)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    classes = args.classes.split(",")

    rows = []
    for cls in classes:
        for t, n_pred, n_matched, p, r, f, n_gt in sweep_class(records, cls, thresholds):
            rows.append(
                {
                    "class": cls,
                    "threshold": t,
                    "n_gt": n_gt,
                    "n_pred": n_pred,
                    "n_matched": n_matched,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f, 4),
                }
            )

    # Pretty-print to stdout
    print(f"{'class':<16}{'thresh':>7}{'n_gt':>7}{'n_pred':>8}{'n_match':>9}"
          f"{'prec':>8}{'recall':>8}{'f1':>8}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['class']:<16}{row['threshold']:>7.2f}{row['n_gt']:>7}"
            f"{row['n_pred']:>8}{row['n_matched']:>9}"
            f"{row['precision']:>8.4f}{row['recall']:>8.4f}{row['f1']:>8.4f}"
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
