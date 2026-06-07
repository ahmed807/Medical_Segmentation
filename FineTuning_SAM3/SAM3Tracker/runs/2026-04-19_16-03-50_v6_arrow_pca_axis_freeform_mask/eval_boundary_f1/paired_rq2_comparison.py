"""
paired_rq2_comparison.py
========================

Restrict the production per-sample metrics to the same basenames the
generic-routing baseline completed on, and report paired means and
paired deltas for IoU and Boundary F1.

This addresses the unpaired-sample criticism of the current Table 6.3:
the headline "BF1 drops more than IoU under generic routing" currently
compares production on 8,288 entries to generic on 5,531. After
restricting production to the same 5,531, the comparison becomes a
paired difference on identical samples, and the asymmetry (which is a
structural argument about box prompts) is reported as exact rather
than as "robust to subset variation."

CSV schema expected (both files):
  basename, annotation_type, iou, boundary_f1, precision, recall

USAGE
-----
    python3 paired_rq2_comparison.py \\
        --production /home/ahma/.../eval_boundary_f1/boundary_f1_per_sample.csv \\
        --generic    /home/ahma/.../eval_boundary_f1/generic_boundary_f1_per_sample.csv
"""

import argparse
import csv
import math
from collections import defaultdict


def load_csv(path):
    """Return {basename: {iou: float, bf1: float, prec: float, rec: float,
    ann_type: str}}."""
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows[row["basename"]] = {
                    "iou":      float(row["iou"]),
                    "bf1":      float(row["boundary_f1"]),
                    "prec":     float(row["precision"]),
                    "rec":      float(row["recall"]),
                    "ann_type": row["annotation_type"],
                }
            except (ValueError, KeyError):
                continue
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def paired_se(prod_xs, gen_xs):
    """Standard error of the paired mean difference."""
    diffs = [p - g for p, g in zip(prod_xs, gen_xs)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = stdev(diffs)
    return sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production", required=True,
                    help="path to production per-sample CSV")
    ap.add_argument("--generic", required=True,
                    help="path to generic-baseline per-sample CSV")
    ap.add_argument("--out-csv", default="paired_rq2_per_sample.csv",
                    help="output: per-sample matched rows")
    args = ap.parse_args()

    print(f"Loading production: {args.production}")
    prod = load_csv(args.production)
    print(f"  rows: {len(prod)}")

    print(f"Loading generic:    {args.generic}")
    gen = load_csv(args.generic)
    print(f"  rows: {len(gen)}")

    matched = sorted(set(prod) & set(gen))
    print(f"\nMatched basenames (paired N): {len(matched)}")
    print(f"  prod-only (dropped): {len(set(prod) - set(gen))}")
    print(f"  gen-only  (orphans): {len(set(gen) - set(prod))}")

    # paired lists in the same order
    prod_iou = [prod[b]["iou"] for b in matched]
    gen_iou  = [gen[b]["iou"]  for b in matched]
    prod_bf1 = [prod[b]["bf1"] for b in matched]
    gen_bf1  = [gen[b]["bf1"]  for b in matched]

    # aggregate
    m_prod_iou = mean(prod_iou)
    m_gen_iou  = mean(gen_iou)
    m_prod_bf1 = mean(prod_bf1)
    m_gen_bf1  = mean(gen_bf1)
    d_iou = m_prod_iou - m_gen_iou
    d_bf1 = m_prod_bf1 - m_gen_bf1
    se_iou = paired_se(prod_iou, gen_iou)
    se_bf1 = paired_se(prod_bf1, gen_bf1)

    print("\n" + "=" * 70)
    print("PAIRED COMPARISON ON {} MATCHED ENTRIES".format(len(matched)))
    print("=" * 70)
    print(f"{'routing':24s} {'IoU (mean)':>12s} {'BF1 (mean)':>12s}")
    print("-" * 50)
    print(f"{'Production (type-specific)':24s} {m_prod_iou:>12.4f} {m_prod_bf1:>12.4f}")
    print(f"{'Generic (rect_bbox)':24s} {m_gen_iou:>12.4f} {m_gen_bf1:>12.4f}")
    print("-" * 50)
    print(f"{'Paired delta':24s} {d_iou:>+12.4f} {d_bf1:>+12.4f}")
    print(f"{'Paired SE':24s} {se_iou:>12.4f} {se_bf1:>12.4f}")
    print(f"{'Delta / SE':24s} {d_iou/se_iou:>12.2f} {d_bf1/se_bf1:>12.2f}")

    # Per-class breakdown using production's annotation_type as the true class
    print("\n" + "=" * 70)
    print("PER-CLASS BREAKDOWN (class taken from production annotation_type)")
    print("=" * 70)
    classes = sorted(set(prod[b]["ann_type"] for b in matched))
    print(f"{'class':16s} {'N':>6s} {'prod IoU':>10s} {'gen IoU':>10s} {'ΔIoU':>10s} "
          f"{'prod BF1':>10s} {'gen BF1':>10s} {'ΔBF1':>10s}")
    print("-" * 90)
    for c in classes:
        cs = [b for b in matched if prod[b]["ann_type"] == c]
        if not cs:
            continue
        p_iou = mean([prod[b]["iou"] for b in cs])
        g_iou = mean([gen[b]["iou"]  for b in cs])
        p_bf1 = mean([prod[b]["bf1"] for b in cs])
        g_bf1 = mean([gen[b]["bf1"]  for b in cs])
        print(f"{c:16s} {len(cs):>6d} {p_iou:>10.4f} {g_iou:>10.4f} "
              f"{p_iou - g_iou:>+10.4f} {p_bf1:>10.4f} {g_bf1:>10.4f} "
              f"{p_bf1 - g_bf1:>+10.4f}")

    # sanity check
    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    print(f"Generic full-CSV  IoU mean: {mean([r['iou'] for r in gen.values()]):.4f}  "
          f"(published aggregate: 0.6860)")
    print(f"Generic full-CSV  BF1 mean: {mean([r['bf1'] for r in gen.values()]):.4f}  "
          f"(published aggregate: 0.6946)")
    print("These should match the published row in Table 6.3 to within rounding.")
    print("If they don't, the per-sample CSV uses a different aggregation than")
    print("the published numbers and the paired deltas should be reconciled")
    print("before quoting them in the chapter.")

    # write per-sample CSV (useful as an artefact)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basename", "annotation_type",
                    "prod_iou", "gen_iou", "delta_iou",
                    "prod_bf1", "gen_bf1", "delta_bf1"])
        for b in matched:
            w.writerow([b, prod[b]["ann_type"],
                        f"{prod[b]['iou']:.4f}", f"{gen[b]['iou']:.4f}",
                        f"{prod[b]['iou'] - gen[b]['iou']:+.4f}",
                        f"{prod[b]['bf1']:.4f}", f"{gen[b]['bf1']:.4f}",
                        f"{prod[b]['bf1'] - gen[b]['bf1']:+.4f}"])
    print(f"\nPer-sample paired CSV: {args.out_csv}")

    # LaTeX-ready replacement rows for Table 6.3
    print("\n" + "=" * 70)
    print("LATEX TABLE 6.3 REPLACEMENT ROWS (paired, same {} basenames)".format(len(matched)))
    print("=" * 70)
    n_str = f"{len(matched):,}".replace(",", "{,}")
    print(f"Type-specific (production) & {n_str} & {m_prod_iou:.4f} & {m_prod_bf1:.4f} \\\\")
    print(f"Generic (forced rect\\_bbox) & {n_str} & {m_gen_iou:.4f} & {m_gen_bf1:.4f} \\\\")


if __name__ == "__main__":
    main()
