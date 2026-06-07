"""
search_samples.py
=================

Quick filtered query over any per-sample evaluation CSV produced by
the evaluation scripts. Returns basenames sorted by a metric, so you
can find good success or failure cases for the chapter 6 composite
figures.

Auto-detects schema. Works with all the per-sample CSVs in this
project:

  boundary_f1_per_sample.csv      basename, annotation_type, iou, ...
  lpips_per_sample.csv  (Pipe A)  basename, annotation_type, lpips
  lpips_per_sample.csv  (Pipe B)  entry_id, annotation_type, lpips
  detection_iou_per_sample.csv    entry_id, true_annotation_type,
                                   score, matched_iou, ...

EXAMPLES
--------

# top 20 arrow successes by IoU (pipeline A segmentation)
python search_samples.py --csv eval_boundary_f1/boundary_f1_per_sample.csv \\
    --class arrow --metric iou --order desc --limit 20

# 20 worst arrow cases by IoU
python search_samples.py --csv eval_boundary_f1/boundary_f1_per_sample.csv \\
    --class arrow --metric iou --order asc --limit 20

# pipeline B freeform detections with the worst (highest) LPIPS
python search_samples.py --csv pipeline_b_default/eval_fid_lpips/lpips_per_sample.csv \\
    --class freeform_bbox --metric lpips --order desc --limit 20

# pipeline B detections with high matched IoU
python search_samples.py --csv error_propagation/detection_iou_per_sample.csv \\
    --class arrow --metric matched_iou --order desc --limit 10

# filter by metric range too (e.g. mid-quality samples)
python search_samples.py --csv eval_boundary_f1/boundary_f1_per_sample.csv \\
    --class freeform_bbox --metric iou --min 0.75 --max 0.85 \\
    --order desc --limit 10

# show ALL columns of the matching rows (useful for cross-checking)
python search_samples.py --csv ... --class arrow --metric iou --limit 5 --full

# print just basenames, one per line (pipe-friendly)
python search_samples.py --csv ... --class arrow --metric iou --limit 20 --ids-only
"""

import argparse
import csv
import sys
from pathlib import Path


# possible ID columns, in priority order
ID_KEYS = ["basename", "entry_id"]
TYPE_KEYS = ["annotation_type", "true_annotation_type"]


def detect_id_col(headers):
    for k in ID_KEYS:
        if k in headers:
            return k
    raise SystemExit(
        f"Could not find an ID column in {headers}. "
        f"Expected one of {ID_KEYS}.")


def detect_type_col(headers):
    for k in TYPE_KEYS:
        if k in headers:
            return k
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="per-sample CSV path")
    ap.add_argument("--class", dest="cls", default=None,
                    help="filter by annotation_type/true_annotation_type")
    ap.add_argument("--metric", required=True,
                    help="column to sort/filter by (iou, lpips, "
                         "boundary_f1, matched_iou, score, ...)")
    ap.add_argument("--order", default="desc", choices=["asc", "desc"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min", type=float, default=None,
                    help="minimum metric value (inclusive)")
    ap.add_argument("--max", type=float, default=None,
                    help="maximum metric value (inclusive)")
    ap.add_argument("--full", action="store_true",
                    help="print all columns of matching rows")
    ap.add_argument("--ids-only", action="store_true",
                    help="print only the ID column, one per line")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"file not found: {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    id_col = detect_id_col(headers)
    type_col = detect_type_col(headers)
    if args.metric not in headers:
        raise SystemExit(
            f"metric column {args.metric!r} not in CSV. "
            f"available: {[h for h in headers if h not in (id_col, type_col)]}")

    # filter by class
    if args.cls and type_col is None:
        print(f"warning: CSV has no type column, --class {args.cls} "
              f"ignored", file=sys.stderr)
        args.cls = None
    if args.cls:
        rows = [r for r in rows if r[type_col] == args.cls]

    # filter by metric range, drop unparseable
    out = []
    for r in rows:
        try:
            v = float(r[args.metric])
        except (ValueError, KeyError):
            continue
        if args.min is not None and v < args.min:
            continue
        if args.max is not None and v > args.max:
            continue
        out.append((v, r))

    # sort
    reverse = (args.order == "desc")
    out.sort(key=lambda t: t[0], reverse=reverse)
    out = out[:args.limit]

    if not out:
        print("no matches", file=sys.stderr)
        sys.exit(0)

    if args.ids_only:
        for _, r in out:
            print(r[id_col])
        return

    if args.full:
        # print all columns
        for _, r in out:
            print("  ".join(f"{k}={r[k]}" for k in headers))
        return

    # default: nice table
    cols = [id_col]
    if type_col:
        cols.append(type_col)
    cols.append(args.metric)
    widths = {c: max(len(c), max(len(r[c]) for _, r in out)) for c in cols}
    sep = "  "
    print(sep.join(c.ljust(widths[c]) for c in cols))
    print(sep.join("-" * widths[c] for c in cols))
    for _, r in out:
        print(sep.join(r[c].ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
