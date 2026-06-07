"""
zero_shot_eval_v2.py
--------------------
Production-realistic zero-shot evaluation of Grounded SAM 2 on the
gdino_finetuning_dataset (image-grouped schema with mask paths).

  - CATEGORY prompt: "arrow . dashed line outline . letter or number ."
  - Multi-detection per image (all GDINO boxes above threshold).
  - Per-detection class via GDINO's returned label phrase.
  - TWO mask strategies side-by-side:
        sam2:  GDINO box -> SAM 2 multimask -> best score
        color: HSV thresholding inside (padded) box, dominant-color auto-pick
  - Greedy bbox-IoU matching of preds to GTs PER CLASS.
  - Per-class metrics: AP@0.5, recall, precision, mean mask IoU on matched.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 zero_shot_eval_v2.py --limit 500
    CUDA_VISIBLE_DEVICES=1 python3 zero_shot_eval_v2.py --limit 0
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROMPT = "arrow . dashed line outline . letter or number ."

COLOR_RANGES = {
    "red":    [(np.array([0,   100, 80]),  np.array([10,  255, 255])),
               (np.array([170, 100, 80]),  np.array([180, 255, 255]))],
    "blue":   [(np.array([100, 100, 60]),  np.array([130, 255, 255]))],
    "green":  [(np.array([40,  80,  60]),  np.array([85,  255, 255]))],
    "yellow": [(np.array([20,  100, 100]), np.array([35,  255, 255]))],
    "purple": [(np.array([130, 60,  60]),  np.array([160, 255, 255]))],
    "white":  [(np.array([0,   0,   200]), np.array([180, 40,  255]))],
    "black":  [(np.array([0,   0,   0]),   np.array([180, 255, 60]))],
}


def label_to_class(label):
    s = label.lower()
    if "arrow" in s:
        return "arrow"
    if "dashed" in s or "outline" in s or "line" in s:
        return "freeform_bbox"
    if "letter" in s or "number" in s:
        return "number_letter"
    return None


def mask_from_sam2(predictor, image_np, box):
    H, W = image_np.shape[:2]
    masks, scores, _ = predictor.predict(
        box=np.asarray([box]),
        multimask_output=True,
    )
    if masks.ndim == 4:
        masks = masks[0]
        scores = scores[0] if scores.ndim > 1 else scores
    best = int(np.argmax(scores))
    full = np.zeros((H, W), dtype=np.uint8)
    full[masks[best] > 0] = 1
    return full


def mask_from_color(image_np, box, pad_frac=0.15, min_pixels=10):
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x1p = max(0, x1 - px); y1p = max(0, y1 - py)
    x2p = min(W, x2 + px); y2p = min(H, y2 + py)
    if x2p <= x1p or y2p <= y1p:
        return np.zeros((H, W), dtype=np.uint8), None

    crop_bgr = cv2.cvtColor(image_np[y1p:y2p, x1p:x2p], cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    box_area = (x2p - x1p) * (y2p - y1p)

    best_mask, best_count, best_color = None, 0, None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for color_name, ranges in COLOR_RANGES.items():
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n = int((m > 0).sum())
        if n > 0.7 * box_area or n < min_pixels:
            continue
        if n > best_count:
            best_count, best_mask, best_color = n, m, color_name

    full = np.zeros((H, W), dtype=np.uint8)
    if best_mask is not None:
        full[y1p:y2p, x1p:x2p] = (best_mask > 0).astype(np.uint8)
    return full, best_color


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def mask_iou(m1, m2):
    m1, m2 = m1.astype(bool), m2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def greedy_match(preds, gts, iou_thresh=0.5):
    by_class_preds = defaultdict(list)
    by_class_gts   = defaultdict(list)
    for p in preds:
        by_class_preds[p["class"]].append(p)
    for g in gts:
        by_class_gts[g["class"]].append(g)

    records = []
    classes = set(by_class_preds) | set(by_class_gts)
    for c in classes:
        ps = sorted(by_class_preds.get(c, []), key=lambda p: -p["score"])
        gs = list(by_class_gts.get(c, []))
        gt_used = [False] * len(gs)

        for p in ps:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gs):
                if gt_used[j]:
                    continue
                iou = box_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            matched = best_iou >= iou_thresh and best_j >= 0
            mIoU_sam   = mask_iou(p["mask_sam"],   gs[best_j]["mask"]) if matched else 0.0
            mIoU_color = mask_iou(p["mask_color"], gs[best_j]["mask"]) if matched else 0.0
            if matched:
                gt_used[best_j] = True
            records.append({
                "class": c, "score": p["score"], "matched": matched, "missed": False,
                "box_iou": best_iou,
                "mask_iou_sam": mIoU_sam,
                "mask_iou_color": mIoU_color,
                "color_name": p.get("color_name"),
            })
        for j, used in enumerate(gt_used):
            if not used:
                records.append({
                    "class": c, "score": None, "matched": False, "missed": True,
                    "box_iou": 0.0, "mask_iou_sam": 0.0, "mask_iou_color": 0.0,
                    "color_name": None,
                })
    return records


def compute_ap(records_for_class):
    preds = sorted([r for r in records_for_class if r["score"] is not None],
                   key=lambda r: -r["score"])
    n_gt = sum(1 for r in records_for_class if r["matched"] or r["missed"])
    if not preds or n_gt == 0:
        return 0.0
    tp = fp = 0
    p_arr, r_arr = [], []
    for p in preds:
        if p["matched"]:
            tp += 1
        else:
            fp += 1
        p_arr.append(tp / (tp + fp))
        r_arr.append(tp / n_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        valid = [pr for pr, rc in zip(p_arr, r_arr) if rc >= t]
        ap += (max(valid) if valid else 0.0) / 11
    return ap


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)

    print(f"[init] device: {device}")
    print(f"[init] loading GDINO: {args.gdino_id}")
    processor = AutoProcessor.from_pretrained(args.gdino_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_id).to(device).eval()

    print(f"[init] loading SAM 2: {args.sam2_ckpt}")
    sam2_model = build_sam2(args.sam2_cfg, str(args.sam2_ckpt), device=device)
    sam2 = SAM2ImagePredictor(sam2_model)

    with open(args.test_json) as f:
        all_entries = json.load(f)
    if args.limit and args.limit > 0:
        all_entries = all_entries[:args.limit]
    print(f"[data] {len(all_entries)} images loaded from {args.test_json}")

    all_records = []
    n_overlays = 0

    for entry in tqdm(all_entries, desc="Eval"):
        img_path = os.path.join(args.dataset_root, entry["image"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        H, W = image_np.shape[:2]

        gts = []
        for a in entry["annotations"]:
            mask_path = os.path.join(args.dataset_root, a["mask"])
            if not os.path.exists(mask_path):
                continue
            m = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            gts.append({"box": a["box"], "class": a["class"], "mask": m})

        # ── 1. GDINO detection ──────────────────────────────────────────────
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gdino(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(H, W)],
        )[0]
        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = [str(l) for l in results["labels"]]

        # ── 2. Per-detection: SAM 2 mask + color mask ───────────────────────
        preds = []
        if len(boxes) > 0:
            sam2.set_image(image_np)
            for box, score, label in zip(boxes, scores, labels):
                cls = label_to_class(label)
                if cls is None:
                    continue
                mask_sam = mask_from_sam2(sam2, image_np, box.tolist())
                mask_color, color_name = mask_from_color(image_np, box.tolist())
                preds.append({
                    "box": box.tolist(),
                    "score": float(score),
                    "class": cls,
                    "mask_sam": mask_sam,
                    "mask_color": mask_color,
                    "color_name": color_name,
                })

        records = greedy_match(preds, gts, iou_thresh=0.5)
        for r in records:
            r["image"] = entry["image"]
        all_records.extend(records)

        # ── 3. Overlay sanity check ─────────────────────────────────────────
        if n_overlays < args.n_overlays:
            n_overlays += 1
            stem = Path(entry["image"]).stem
            for strat, key in [("sam", "mask_sam"), ("color", "mask_color")]:
                ov = image_np.copy()
                for g in gts:
                    ov[g["mask"].astype(bool)] = [0, 255, 0]
                for p in preds:
                    if p[key] is not None:
                        m = p[key].astype(bool)
                        ov[m] = (ov[m] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
                for p in preds:
                    x1, y1, x2, y2 = [int(v) for v in p["box"]]
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(ov, p["class"], (x1, max(15, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imwrite(str(out / "overlays" / f"{stem}_{strat}.jpg"),
                            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    # ── 4. Metrics ──────────────────────────────────────────────────────────
    by_class = defaultdict(list)
    for r in all_records:
        by_class[r["class"]].append(r)

    print("\n" + "=" * 110)
    print(f"ZERO-SHOT V2  |  prompt: {PROMPT}")
    print(f"             |  thresholds: box={args.box_threshold}  text={args.text_threshold}")
    print(f"             |  N images: {len(all_entries)}")
    print("=" * 110)
    header = (f"{'class':<16}{'N_gt':>6}{'N_pred':>8}{'matched':>9}"
              f"{'recall':>9}{'precision':>11}"
              f"{'AP@0.5':>9}{'mIoU(sam,hit)':>15}{'mIoU(color,hit)':>17}")
    print(header)
    print("-" * 110)

    summary = {}
    for cls, recs in sorted(by_class.items()):
        n_gt   = sum(1 for r in recs if r["matched"] or r["missed"])
        n_pred = sum(1 for r in recs if r["score"] is not None)
        n_match = sum(1 for r in recs if r["matched"])
        recall    = n_match / n_gt   if n_gt   else 0.0
        precision = n_match / n_pred if n_pred else 0.0
        ap        = compute_ap(recs)
        sam_hits   = [r["mask_iou_sam"]   for r in recs if r["matched"]]
        color_hits = [r["mask_iou_color"] for r in recs if r["matched"]]
        miou_sam   = float(np.mean(sam_hits))   if sam_hits   else 0.0
        miou_color = float(np.mean(color_hits)) if color_hits else 0.0
        print(f"{cls:<16}{n_gt:>6}{n_pred:>8}{n_match:>9}"
              f"{recall:>9.3f}{precision:>11.3f}"
              f"{ap:>9.3f}{miou_sam:>15.3f}{miou_color:>17.3f}")
        summary[cls] = {
            "n_gt": n_gt, "n_pred": n_pred, "n_matched": n_match,
            "recall": round(recall, 4), "precision": round(precision, 4),
            "ap@0.5": round(ap, 4),
            "miou_sam_on_matched":   round(miou_sam, 4),
            "miou_color_on_matched": round(miou_color, 4),
        }
    print("=" * 110)

    print("\nBest mask strategy per class (by mIoU on matched):")
    for cls, m in summary.items():
        winner = "sam2" if m["miou_sam_on_matched"] >= m["miou_color_on_matched"] else "color"
        delta = abs(m["miou_sam_on_matched"] - m["miou_color_on_matched"])
        print(f"  {cls:<16} -> {winner}  (Δ = {delta:.3f})")

    with open(out / "summary.json", "w") as f:
        json.dump({"prompt": PROMPT,
                   "thresholds": {"box": args.box_threshold, "text": args.text_threshold},
                   "n_images": len(all_entries),
                   "per_class": summary}, f, indent=2)

    slim = []
    for r in all_records:
        slim.append({k: v for k, v in r.items() if not isinstance(v, np.ndarray)})
    with open(out / "raw_records.json", "w") as f:
        json.dump(slim, f, indent=2)

    print(f"\n[done] results: {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-json",     default="gdino_finetuning_dataset/test.json")
    p.add_argument("--dataset-root",  default="gdino_finetuning_dataset")
    p.add_argument("--output-dir",    default="zero_shot_v2_results")
    p.add_argument("--gdino-id",      default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam2-cfg",      default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",     default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--box-threshold",  type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.20)
    p.add_argument("--limit",          type=int,   default=500,
                   help="Number of images to evaluate. 0 = full set.")
    p.add_argument("--n-overlays",     type=int,   default=20)
    args = p.parse_args()
    run_eval(args)
