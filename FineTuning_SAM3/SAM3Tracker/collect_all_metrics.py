"""
collect_all_metrics.py
=======================

Gathers every metric that the evaluation drivers and the inference scripts
have produced across the four runs, and writes ONE consolidated file you
can upload in a single step.

Runs covered
------------
  1. iou      Pipeline A v6 (production, loss with IoU-MSE term)
  2. no_iou   v6 ablation (loss without IoU-MSE term)
  3. pvs      SAM3TrackerModel zero-shot (off-the-shelf, geometric prompts)
  4. pcs      Sam3Model zero-shot (off-the-shelf, text prompts)

For each run it looks for, and tolerates the absence of:
  - <run>/eval_boundary_f1/boundary_f1_summary.csv   (IoU + Boundary F1)
  - <run>/eval_fid_lpips/fid_lpips_summary.csv        (LPIPS + FID)
  - <run>/<results_subdir>/evaluation_metrics.json    (inference-time IoU)
    (also tries <run>/evaluation_metrics.json)

Outputs (written next to this script unless --out-dir is given)
  - ALL_METRICS.md   human-readable, paste-ready, every table inline
  - ALL_METRICS.csv  long format: run, source, type, metric, value
  - prints ALL_METRICS.md to stdout as well

Usage
-----
    python collect_all_metrics.py
    python collect_all_metrics.py --root /path/to/SAM3Tracker
    python collect_all_metrics.py --out-dir /tmp

Edit the RUNS table below if any directory name differs from the default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Run registry. Paths are relative to --root (default: this script's dir).
# results_subdir is where the inference-time evaluation_metrics.json lives.
# ---------------------------------------------------------------------------

RUNS = [
    {
        "key": "iou",
        "label": "Pipeline A v6 (loss WITH IoU-MSE term) — production",
        "run_dir": "runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask",
        "results_subdir": "inference_results_full_v2",
    },
    {
        "key": "no_iou",
        "label": "v6 ablation (loss WITHOUT IoU-MSE term)",
        "run_dir": "runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss",
        "results_subdir": "inference_results_full",
    },
    {
        "key": "pvs",
        "label": "SAM3TrackerModel zero-shot (off-the-shelf, geometric prompts)",
        "run_dir": "runs/PVS_ZERO_SHOT",
        "results_subdir": ".",
    },
    {
        "key": "pcs",
        "label": "Sam3Model zero-shot (off-the-shelf, text prompts)",
        "run_dir": "runs/PCS_ZERO_SHOT",
        "results_subdir": ".",
    },
]


# ---------------------------------------------------------------------------
# Readers — each returns (rows, note). rows is a list of dicts. note is a
# short string explaining why rows is empty, or "" on success.
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> tuple[list[dict], str]:
    if not path.exists():
        return [], f"not found: {path}"
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return [], f"empty file: {path}"
        return rows, ""
    except Exception as e:  # noqa: BLE001
        return [], f"unreadable ({e}): {path}"


def read_eval_metrics_json(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, f"not found: {path}"
    try:
        with open(path) as f:
            return json.load(f), ""
    except Exception as e:  # noqa: BLE001
        return {}, f"unreadable ({e}): {path}"


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def bf1_block(rows: list[dict]) -> str:
    headers = ["type", "n", "iou_mean", "iou_median",
               "f1_mean", "f1_median", "precision_mean", "recall_mean"]
    body = []
    for r in rows:
        body.append([r.get(h, "") for h in headers])
    return md_table(headers, body)


def fidlpips_block(rows: list[dict]) -> str:
    headers = ["type", "n", "lpips_mean", "lpips_median", "lpips_std", "fid"]
    body = []
    for r in rows:
        body.append([r.get(h, "") for h in headers])
    return md_table(headers, body)


def json_block(obj: dict) -> str:
    """Render the inference-time evaluation_metrics.json compactly."""
    if not obj:
        return "_(no data)_\n"
    # Common shapes: {"per_type": {...}, "overall": {...}} or flat per-type.
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```\n"


# ---------------------------------------------------------------------------
# Long-format CSV emitter
# ---------------------------------------------------------------------------

def csv_long_rows(run_key: str, source: str,
                  rows: list[dict], metric_cols: list[str]) -> list[list[str]]:
    out = []
    for r in rows:
        rtype = r.get("type", "")
        n = r.get("n", "")
        for m in metric_cols:
            val = r.get(m, "")
            if val == "":
                continue
            out.append([run_key, source, rtype, n, m, val])
    return out


def flatten_json(prefix: str, obj, acc: list[tuple[str, str]]):
    """Flatten a nested dict/number JSON into (dotted_key, value) pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_json(f"{prefix}.{k}" if prefix else str(k), v, acc)
    elif isinstance(obj, (int, float, str)):
        acc.append((prefix, str(obj)))
    # lists are ignored on purpose; these summaries are scalar trees


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="SAM3Tracker root (default: this script's directory)")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write ALL_METRICS.* (default: --root)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    md_parts: list[str] = []
    csv_long: list[list[str]] = []
    notes: list[str] = []

    md_parts.append("# Consolidated Metrics — iou / no_iou / pvs / pcs\n")
    md_parts.append(f"Root: `{root}`\n")
    md_parts.append(
        "Each run section has up to three sources: Boundary-F1 evaluation "
        "(IoU + BF1), FID/LPIPS evaluation (inpainting, only present for the "
        "full-pipeline runs), and the inference-time `evaluation_metrics.json` "
        "(raw IoU written during inference).\n"
    )

    for run in RUNS:
        rd = root / run["run_dir"]
        md_parts.append("\n" + "=" * 70)
        md_parts.append(f"\n## {run['key']} — {run['label']}\n")
        md_parts.append(f"Run dir: `{rd}`\n")

        if not rd.exists():
            md_parts.append(f"\n**RUN DIRECTORY MISSING:** `{rd}`\n")
            notes.append(f"[{run['key']}] run dir missing: {rd}")
            continue

        # --- Boundary F1 ---
        bf1_path = rd / "eval_boundary_f1" / "boundary_f1_summary.csv"
        bf1_rows, bf1_note = read_csv_rows(bf1_path)
        md_parts.append("\n### Boundary F1 (IoU + BF1)\n")
        if bf1_rows:
            md_parts.append(bf1_block(bf1_rows))
            csv_long += csv_long_rows(
                run["key"], "boundary_f1", bf1_rows,
                ["iou_mean", "iou_median", "f1_mean", "f1_median",
                 "precision_mean", "recall_mean"],
            )
        else:
            md_parts.append(f"_missing_ — {bf1_note}\n")
            notes.append(f"[{run['key']}] boundary_f1: {bf1_note}")

        # --- FID / LPIPS ---
        fl_path = rd / "eval_fid_lpips" / "fid_lpips_summary.csv"
        fl_rows, fl_note = read_csv_rows(fl_path)
        md_parts.append("\n### FID + LPIPS (inpainting)\n")
        if fl_rows:
            md_parts.append(fidlpips_block(fl_rows))
            csv_long += csv_long_rows(
                run["key"], "fid_lpips", fl_rows,
                ["lpips_mean", "lpips_median", "lpips_std", "fid"],
            )
        else:
            md_parts.append(
                f"_not applicable / missing_ — {fl_note}\n"
                "(seg-only zero-shot runs have no inpainted image, so FID and "
                "LPIPS do not apply.)\n"
            )
            notes.append(f"[{run['key']}] fid_lpips: {fl_note}")

        # --- inference-time evaluation_metrics.json ---
        sub = run["results_subdir"]
        candidates = [rd / sub / "evaluation_metrics.json",
                      rd / "evaluation_metrics.json"]
        em_obj, em_note = {}, "not found"
        em_used = None
        for c in candidates:
            em_obj, em_note = read_eval_metrics_json(c)
            if em_obj:
                em_used = c
                break
        md_parts.append("\n### Inference-time evaluation_metrics.json\n")
        if em_obj:
            md_parts.append(f"Source: `{em_used}`\n\n")
            md_parts.append(json_block(em_obj))
            flat: list[tuple[str, str]] = []
            flatten_json("", em_obj, flat)
            for k, v in flat:
                csv_long.append([run["key"], "eval_metrics_json", k, "", k, v])
        else:
            md_parts.append(f"_missing_ — {em_note}\n")
            notes.append(f"[{run['key']}] evaluation_metrics.json: {em_note}")

    # --- collection notes ---
    md_parts.append("\n" + "=" * 70)
    md_parts.append("\n## Collection notes\n")
    if notes:
        for n in notes:
            md_parts.append(f"- {n}")
        md_parts.append("")
    else:
        md_parts.append("All sources found for all runs.\n")

    md_text = "\n".join(md_parts) + "\n"

    md_out = out_dir / "ALL_METRICS.md"
    csv_out = out_dir / "ALL_METRICS.csv"

    md_out.write_text(md_text)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "source", "type_or_key", "n", "metric", "value"])
        w.writerows(csv_long)

    print(md_text)
    print("=" * 70)
    print(f"Written: {md_out}")
    print(f"Written: {csv_out}")
    print()
    print("Upload ALL_METRICS.md (paste-ready) or ALL_METRICS.csv (compact).")


if __name__ == "__main__":
    main()
