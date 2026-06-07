"""
compose_chapter_6_figures.py  (v2 -- manual sample selection)
============================================================

Produces the four chapter 6 composite figures from data on disk.

  Fig 6.5  pipeline_a_inpainting_examples.png        (§6.4.1)
  Fig 6.6  pipeline_a_vs_b_examples.png              (§6.5.1)
  Fig 6.7  qualitative_success_cases.png             (§6.6.1)
  Fig 6.8  qualitative_failure_cases.png             (§6.6.2)

v2 supports manual sample IDs via --samples-json. If no JSON is
provided, falls back to auto-selection by metric percentile.

Why manual: auto-selection sometimes picks visually unhelpful samples,
or Pipeline B detections whose source images are outside the SAM3Tracker
test split (which produces stale-file mismatches in the A-vs-B
comparison). Specifying exact IDs removes both problems.

SAMPLES JSON FORMAT
-------------------
{
  "fig5_inpainting_examples": {
    "arrow":         "41_1_1",
    "freeform_bbox": "40_2_9",
    "number_letter": "94_0_1"
  },
  "fig6_a_vs_b_examples": [
    {"row_label": "arrow: detection picks wrong object",
     "pb_entry_id": "9_2_det2", "pa_basename": null},
    {"row_label": "n/l: false positive absorbed",
     "pb_entry_id": "8_3_det1", "pa_basename": null}
  ],
  "fig7_success_cases": {
    "arrow":         "41_1_1",
    "freeform_bbox": "40_2_9",
    "number_letter": "94_0_1"
  },
  "fig8_failure_cases": {
    "arrow":         null,
    "freeform_bbox": null,
    "number_letter": null
  }
}

Slot value = Pipeline A basename (unique_base, e.g. "41_1_1") to
pick manually. null means auto-select by percentile.

For fig6, each row has:
  row_label   -- left-side text label
  pb_entry_id -- Pipeline B entry id, e.g. "8_3_det1"
  pa_basename -- (optional) Pipeline A basename to pair with. null
                 auto-derives a paired Pipeline A entry on the
                 same source image with an on-disk _clean.png.

USAGE
-----
    python compose_chapter_6_figures.py \\
        --dataset-root  ./sam_finetuning_dataset \\
        --test-json     ./sam_finetuning_dataset/test.json \\
        --pa-inference  ./runs/.../inference_results_full_v2 \\
        --pa-lpips-csv  ./runs/.../eval_fid_lpips/lpips_per_sample.csv \\
        --pa-bf1-csv    ./runs/.../eval_boundary_f1/boundary_f1_per_sample.csv \\
        --pb-inference  ./runs/.../pipeline_b_default \\
        --pb-lpips-csv  ./runs/.../pipeline_b_default/eval_fid_lpips/lpips_per_sample.csv \\
        --pb-input-json /home/ahma/Grounded-SAM-2/pipeline_b_input_default.json \\
        --orig-root-replace "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset:/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset" \\
        --samples-json  ./chapter_6_figure_samples.json \\
        --out-dir       ./figures
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def read_image(path):
    if path is None or not os.path.exists(str(path)):
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show(ax, img, title=None, fontsize=9):
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#a44")
    else:
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")
        s.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=fontsize)


def row_label(ax, text):
    ax.text(-0.08, 0.5, text, transform=ax.transAxes,
            ha="right", va="center", fontsize=9, fontweight="bold")


def annotation_bbox(mask, margin=16):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 127
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    H, W = binary.shape
    return (max(int(xs.min()) - margin, 0),
            max(int(ys.min()) - margin, 0),
            min(int(xs.max()) + 1 + margin, W),
            min(int(ys.max()) + 1 + margin, H))


def crop_to_bbox(img, bbox):
    if img is None or bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


def apply_prefix_replace(p, rules):
    if not p or not rules:
        return p
    p = str(p)
    for old, new in rules:
        if p.startswith(old):
            return new + p[len(old):]
    return p


# ---------------------------------------------------------------------------
# CSV reading + percentile picker
# ---------------------------------------------------------------------------

def read_per_sample_csv(path, value_key, id_key="basename",
                        type_key="annotation_type"):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[value_key])
            except (KeyError, ValueError):
                continue
            out.append({
                "id": row[id_key],
                "type": row.get(type_key) or row.get("true_annotation_type",
                                                     "unknown"),
                "value": v,
            })
    return out


def pick_percentile(rows, type_filter, pct):
    vs = [r for r in rows if r["type"] == type_filter]
    if not vs:
        return None
    vs.sort(key=lambda r: r["value"])
    idx = int(round((len(vs) - 1) * pct / 100.0))
    idx = max(0, min(len(vs) - 1, idx))
    return vs[idx]


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

class PathResolver:
    def __init__(self, test_json, dataset_root, pa_inference,
                 pb_input_json=None, pb_inference=None, replace_rules=None):
        self.dataset_root = Path(dataset_root)
        self.pa_inference = Path(pa_inference)
        self.pb_inference = Path(pb_inference) if pb_inference else None
        self.replace_rules = replace_rules or []

        with open(test_json) as f:
            self.test_entries = json.load(f)
        self.pa_by_basename = {}
        self.pa_by_source = defaultdict(list)
        for e in self.test_entries:
            base = Path(e["image"]).stem
            self.pa_by_basename[base] = e
            src = base.rsplit("_", 1)[0]
            self.pa_by_source[src].append(e)

        self.pb_by_entry = {}
        if pb_input_json and os.path.exists(pb_input_json):
            with open(pb_input_json) as f:
                pb = json.load(f)
            for d in pb:
                if "source_image" in d and "detection_idx" in d:
                    eid = (f"{Path(d['source_image']).stem}"
                           f"_det{d['detection_idx']}")
                else:
                    eid = Path(d["image"]).stem
                self.pb_by_entry[eid] = d

    def pa_paths(self, basename):
        e = self.pa_by_basename.get(basename)
        if e is None:
            return None
        ann_type = e.get("annotation_type", "arrow")
        return {
            "annotated_input": self.dataset_root / e["image"],
            "gt_object_mask":  self.dataset_root / e["annotation"],
            "gt_annotation_mask": self.dataset_root / e["annotation_mask"],
            "gt_clean": apply_prefix_replace(
                str(self.dataset_root / e["original_clean_image"]),
                self.replace_rules),
            "pa_predicted_clean": (self.pa_inference / ann_type
                                   / f"{basename}_clean.png"),
            "pa_predicted_object_mask": (self.pa_inference / ann_type
                                         / f"{basename}_object_mask.png"),
            "annotation_type": ann_type,
        }

    def pb_paths(self, entry_id):
        d = self.pb_by_entry.get(entry_id)
        if d is None:
            return None
        ann_type = d["annotation_type"]
        return {
            "annotated_input": apply_prefix_replace(
                d.get("image", ""), self.replace_rules),
            "gt_clean": apply_prefix_replace(
                d.get("original_clean_image", ""), self.replace_rules),
            "annotation_mask": apply_prefix_replace(
                d.get("annotation_mask", ""), self.replace_rules),
            "pb_predicted_clean": (self.pb_inference / ann_type
                                   / f"{entry_id}_clean.png"),
            "source_stem": Path(d["source_image"]).stem,
            "annotation_type": ann_type,
            "true_annotation_type": d.get("true_annotation_type", ann_type),
        }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig5_inpainting(args, resolver, samples):
    print("\n=== Fig 6.5  pipeline_a_inpainting_examples ===")
    rows = read_per_sample_csv(args.pa_lpips_csv, value_key="lpips")
    classes = ["arrow", "freeform_bbox", "number_letter"]
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 5.6))
    for c, t in enumerate(["annotated input", "predicted clean",
                            "original clean"]):
        axes[0, c].set_title(t, fontsize=10, pad=4)

    sel = samples.get("fig5_inpainting_examples", {})
    for r, cls in enumerate(classes):
        manual = sel.get(cls)
        if manual:
            basename, metric_str = manual, "manual pick"
        else:
            pick = pick_percentile(rows, cls, 50)
            if pick is None:
                print(f"  {cls}: no samples")
                for c in range(3): show(axes[r, c], None)
                continue
            basename, metric_str = pick["id"], f"LPIPS={pick['value']:.3f}"
        paths = resolver.pa_paths(basename)
        if paths is None:
            print(f"  {cls}: basename {basename} not in test.json")
            for c in range(3): show(axes[r, c], None)
            continue
        print(f"  {cls:14s} basename={basename}  {metric_str}")
        ann_mask = read_image(paths["gt_annotation_mask"])
        bbox = annotation_bbox(ann_mask, margin=16)
        show(axes[r, 0], crop_to_bbox(read_image(paths["annotated_input"]), bbox))
        show(axes[r, 1], crop_to_bbox(read_image(paths["pa_predicted_clean"]), bbox))
        show(axes[r, 2], crop_to_bbox(read_image(paths["gt_clean"]), bbox))
        row_label(axes[r, 0], cls)

    plt.tight_layout()
    out_path = args.out_dir / "pipeline_a_inpainting_examples.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig6_a_vs_b(args, resolver, samples):
    print("\n=== Fig 6.6  pipeline_a_vs_b_examples ===")
    manual_rows = samples.get("fig6_a_vs_b_examples")
    fallback_rows = [
        {"row_label": "freeform", "pb_class": "freeform_bbox", "pct": 90},
        {"row_label": "n/l",      "pb_class": "number_letter", "pct": 50},
    ]
    use_rows = manual_rows if manual_rows else fallback_rows
    n_rows = len(use_rows)

    fig, axes = plt.subplots(n_rows, 4, figsize=(7.5, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for c, t in enumerate(["annotated input", "Pipeline A clean",
                            "Pipeline B clean", "original clean"]):
        axes[0, c].set_title(t, fontsize=10, pad=4)

    rows_pb = None
    if not manual_rows:
        rows_pb = read_per_sample_csv(args.pb_lpips_csv, value_key="lpips",
                                      id_key="entry_id")

    for r, spec in enumerate(use_rows):
        pb_entry_id = spec.get("pb_entry_id")
        pa_basename_hint = spec.get("pa_basename")
        row_text = spec.get("row_label", "")

        if pb_entry_id:
            pb_paths_pick = resolver.pb_paths(pb_entry_id)
            if pb_paths_pick is None:
                print(f"  row {r}: pb_entry_id {pb_entry_id} not in input")
                for c in range(4): show(axes[r, c], None)
                continue
            metric_str = "manual pb"
        else:
            cls = spec["pb_class"]
            candidates = [c for c in rows_pb if c["type"] == cls]
            if not candidates:
                print(f"  row {r}: no PB samples for {cls}")
                for c in range(4): show(axes[r, c], None)
                continue
            candidates.sort(key=lambda c: c["value"])
            target = int(round((len(candidates) - 1) * spec["pct"] / 100.0))
            order = sorted(range(len(candidates)),
                           key=lambda i: abs(i - target))
            picked = None
            for idx in order[:60]:
                cand = candidates[idx]
                pb_paths_t = resolver.pb_paths(cand["id"])
                if pb_paths_t is None:
                    continue
                if os.path.exists(pb_paths_t["pb_predicted_clean"]):
                    picked = (cand, pb_paths_t)
                    break
            if picked is None:
                print(f"  row {r}: no PB candidate with on-disk clean")
                for c in range(4): show(axes[r, c], None)
                continue
            cand, pb_paths_pick = picked
            pb_entry_id = cand["id"]
            metric_str = f"LPIPS={cand['value']:.3f}"

        if pa_basename_hint:
            pa_pair = resolver.pa_paths(pa_basename_hint)
            if pa_pair is None:
                print(f"  row {r}: pa_basename {pa_basename_hint} not in "
                      f"test.json")
                for c in range(4): show(axes[r, c], None)
                continue
            pa_basename = pa_basename_hint
        else:
            src = pb_paths_pick["source_stem"]
            pa_entries = resolver.pa_by_source.get(src, [])
            pa_pair = None
            pa_basename = None
            for pa_e in pa_entries:
                bn = Path(pa_e["image"]).stem
                t_paths = resolver.pa_paths(bn)
                if t_paths and os.path.exists(t_paths["pa_predicted_clean"]):
                    pa_pair = t_paths
                    pa_basename = bn
                    break
            if pa_pair is None:
                print(f"  row {r}: no on-disk PA pair for source {src}")
                pa_pair = {"pa_predicted_clean": None}
                pa_basename = "(no pair)"

        print(f"  row {r}: pb={pb_entry_id}  pa={pa_basename}  {metric_str}")
        # debug-paths: show exactly which files are being read for this row.
        # If the rendered image doesn't match expectations, this is where to look.
        print(f"           PB source_stem    = {pb_paths_pick.get('source_stem')}")
        print(f"           PB annotated_input = {pb_paths_pick['annotated_input']}")
        print(f"           PB annotation_mask = {pb_paths_pick['annotation_mask']}")
        print(f"           PB predicted_clean = {pb_paths_pick['pb_predicted_clean']}"
              f"  {'(EXISTS)' if os.path.exists(pb_paths_pick['pb_predicted_clean']) else '(MISSING)'}")
        print(f"           PA predicted_clean = {pa_pair.get('pa_predicted_clean')}"
              f"  {'(EXISTS)' if pa_pair.get('pa_predicted_clean') and os.path.exists(pa_pair['pa_predicted_clean']) else '(MISSING)'}")
        print(f"           gt_clean (orig)    = {pb_paths_pick.get('gt_clean')}"
              f"  {'(EXISTS)' if os.path.exists(pb_paths_pick['gt_clean']) else '(MISSING)'}")
        ann_mask = read_image(pb_paths_pick["annotation_mask"])
        bbox = annotation_bbox(ann_mask, margin=16)
        show(axes[r, 0], crop_to_bbox(read_image(pb_paths_pick["annotated_input"]), bbox))
        show(axes[r, 1], crop_to_bbox(read_image(pa_pair["pa_predicted_clean"]), bbox))
        show(axes[r, 2], crop_to_bbox(read_image(pb_paths_pick["pb_predicted_clean"]), bbox))
        show(axes[r, 3], crop_to_bbox(read_image(pb_paths_pick["gt_clean"]), bbox))
        row_label(axes[r, 0], row_text)

    plt.tight_layout()
    out_path = args.out_dir / "pipeline_a_vs_b_examples.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def _qualitative_figure(args, resolver, samples, samples_key, percentile,
                         fname, fig_label):
    print(f"\n=== {fig_label} ({fname}) ===")
    rows = read_per_sample_csv(args.pa_bf1_csv, value_key="iou")
    classes = ["arrow", "freeform_bbox", "number_letter"]
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 5.6))
    for c, t in enumerate(["annotated input", "predicted object mask",
                            "ground-truth object mask"]):
        axes[0, c].set_title(t, fontsize=10, pad=4)

    sel = samples.get(samples_key, {})
    for r, cls in enumerate(classes):
        manual = sel.get(cls)
        iou_val = None
        if manual:
            basename = manual
            metric_str = "manual pick"
            for rr in rows:
                if rr["id"] == manual:
                    iou_val = rr["value"]
                    break
        else:
            pick = pick_percentile(rows, cls, percentile)
            if pick is None:
                print(f"  {cls}: no samples")
                for c in range(3): show(axes[r, c], None)
                continue
            basename = pick["id"]
            iou_val = pick["value"]
            metric_str = f"IoU={iou_val:.3f}"
        paths = resolver.pa_paths(basename)
        if paths is None:
            print(f"  {cls}: basename {basename} not in test.json")
            for c in range(3): show(axes[r, c], None)
            continue
        print(f"  {cls:14s} basename={basename}  {metric_str}")
        show(axes[r, 0], read_image(paths["annotated_input"]))
        show(axes[r, 1], read_image(paths["pa_predicted_object_mask"]))
        show(axes[r, 2], read_image(paths["gt_object_mask"]))
        if iou_val is not None:
            row_label(axes[r, 0], f"{cls}\nIoU {iou_val:.2f}")
        else:
            row_label(axes[r, 0], cls)

    plt.tight_layout()
    out_path = args.out_dir / fname
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def fig7_success(args, resolver, samples):
    _qualitative_figure(args, resolver, samples, "fig7_success_cases",
                         90, "qualitative_success_cases.png",
                         "Fig 6.7  qualitative_success")


def fig8_failure(args, resolver, samples):
    _qualitative_figure(args, resolver, samples, "fig8_failure_cases",
                         10, "qualitative_failure_cases.png",
                         "Fig 6.8  qualitative_failure")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--pa-inference", required=True)
    ap.add_argument("--pa-lpips-csv", required=True)
    ap.add_argument("--pa-bf1-csv", required=True)
    ap.add_argument("--pb-inference", default=None)
    ap.add_argument("--pb-lpips-csv", default=None)
    ap.add_argument("--pb-input-json", default=None)
    ap.add_argument("--orig-root-replace", action="append", default=[],
                    metavar="OLD:NEW")
    ap.add_argument("--samples-json", default=None,
                    help="Optional JSON file with manual sample IDs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rules = []
    for spec in args.orig_root_replace:
        if ":" in spec:
            o, n = spec.split(":", 1)
            rules.append((o, n))

    resolver = PathResolver(
        test_json=args.test_json,
        dataset_root=args.dataset_root,
        pa_inference=args.pa_inference,
        pb_input_json=args.pb_input_json,
        pb_inference=args.pb_inference,
        replace_rules=rules,
    )
    print(f"PA test entries: {len(resolver.pa_by_basename)}")
    print(f"PB input entries: {len(resolver.pb_by_entry)}")

    samples = {}
    if args.samples_json and os.path.exists(args.samples_json):
        with open(args.samples_json) as f:
            samples = json.load(f)
        print(f"Loaded manual samples from {args.samples_json}")

    only = set(args.only.split(",")) if args.only else None

    if only is None or "a" in only:
        fig5_inpainting(args, resolver, samples)
    if (only is None or "ab" in only) and args.pb_lpips_csv and \
            args.pb_input_json and args.pb_inference:
        fig6_a_vs_b(args, resolver, samples)
    if only is None or "s" in only:
        fig7_success(args, resolver, samples)
    if only is None or "f" in only:
        fig8_failure(args, resolver, samples)

    print("\nDone. PNGs in:", args.out_dir)


if __name__ == "__main__":
    main()