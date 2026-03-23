"""
inference.py
------------
Full inference pipeline:

  Step 1 - SAM3
    Input : annotated image + bounding box of the annotation (arrow / symbol)
    Output: segmentation mask of the TARGET object the annotation points at

  Step 2 - Flux Fill
    Input : original annotated image + annotation mask (arrow pixels)
    Output: clean image with annotation removed

  Final outputs per image:
    {id}_clean.png         <- annotation removed by Flux
    {id}_object_mask.png   <- object segmentation mask from SAM3
    {id}_comparison.png    <- 4-panel side-by-side visualisation

Usage - single image:
    python inference.py \\
        --model      sam3_best.pth \\
        --image      path/to/annotated.jpg \\
        --annot_mask path/to/annotation_mask.png \\
        --out_dir    results/

Usage - full test split with per-type evaluation:
    python inference.py \\
        --model    sam3_best.pth \\
        --batch \\
        --dataset  sam_finetuning_dataset \\
        --out_dir  results/

Evaluation metrics (requires: pip install torchmetrics lpips):
    SSIM  (higher = better) - structural similarity vs original clean image
    LPIPS (lower  = better) - perceptual similarity vs original clean image
    IoU   (higher = better) - SAM mask vs GT segmap (test split only)
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL          import Image
from transformers import Sam3Processor, Sam3Model
from diffusers    import FluxFillPipeline
from tqdm         import tqdm
from collections  import defaultdict

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    import lpips as lpips_lib
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Evaluation metrics unavailable.  pip install torchmetrics lpips")


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
SAM_MODEL_ID  = "facebook/sam3"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
DEVICE        = "cuda:0"
RANDOM_SEED   = 42

FLUX_PROMPT         = "a clean photo, no annotations, no arrows, no labels"
FLUX_GUIDANCE_SCALE = 30
FLUX_STEPS          = 50
# ───────────────────────────────────────────────────────────────────────────────


# ── HELPERS ────────────────────────────────────────────────────────────────────

def get_prompt_box(annot_np: np.ndarray, fallback=None, padding: int = 10) -> list:
    """Union bbox of all annotation contours + padding."""
    contours, _ = cv2.findContours(annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = annot_np.shape
        return fallback or [0, 0, w, h]
    all_pts      = np.vstack(contours)
    x, y, w, h   = cv2.boundingRect(all_pts)
    img_h, img_w = annot_np.shape
    return [
        max(0,     x - padding),
        max(0,     y - padding),
        min(img_w, x + w + padding),
        min(img_h, y + h + padding),
    ]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 tensor [1, 3, H, W] in [0, 1]."""
    return torch.tensor(
        np.array(img.convert("RGB")) / 255.0, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

def load_sam(model_path: str, device: str = DEVICE):
    """Load fine-tuned SAM3."""
    print(f"Loading SAM3 weights from {model_path}...")
    processor = Sam3Processor.from_pretrained(SAM_MODEL_ID)
    model     = Sam3Model.from_pretrained(SAM_MODEL_ID)
    ckpt      = torch.load(model_path, map_location=device)
    # Support both plain state dict and full checkpoint dict
    state     = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print("  SAM3 ready.")
    return model, processor


def load_flux(device: str = DEVICE):
    """Load Flux Fill pipeline."""
    print(f"Loading Flux Fill from {FLUX_MODEL_ID}...")
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)
    print("  Flux Fill ready.")
    return pipe


# ── CORE INFERENCE ─────────────────────────────────────────────────────────────

def sam_predict(model, processor,
                image: Image.Image,
                annot_np: np.ndarray,
                device: str = DEVICE) -> Image.Image:
    """
    SAM3 inference: annotation bbox -> object segmentation mask.

    Returns:
        PIL 'L' image (255 = object, 0 = background), same size as input.
    """
    orig_w, orig_h = image.size
    prompt_box     = get_prompt_box(annot_np)

    inputs = processor(
        image, input_boxes=[[prompt_box]], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            outputs = model(**inputs, multimask_output=False)

    low_res  = outputs.pred_masks[:, 0, :, :]
    upscaled = F.interpolate(
        low_res.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bilinear", align_corners=False
    ).squeeze()

    binary = (torch.sigmoid(upscaled) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def flux_remove_annotation(pipe,
                            image: Image.Image,
                            annot_mask: Image.Image,
                            device: str = DEVICE) -> Image.Image:
    """
    Flux Fill: erase annotation pixels from image.

    Args:
        image       : PIL RGB annotated image
        annot_mask  : PIL 'L' mask — white = pixels to erase (the arrow/symbol)

    Returns:
        PIL RGB clean image with annotation removed.
    """
    orig_w, orig_h = image.size

    return pipe(
        prompt              = FLUX_PROMPT,
        image               = image,
        mask_image          = annot_mask,
        height              = orig_h,
        width               = orig_w,
        guidance_scale      = FLUX_GUIDANCE_SCALE,
        num_inference_steps = FLUX_STEPS,
        max_sequence_length = 512,
        generator           = torch.Generator(device).manual_seed(RANDOM_SEED),
    ).images[0]


# ── EVALUATION METRICS ─────────────────────────────────────────────────────────

_lpips_fn = None   # Lazy-load once

def compute_image_metrics(pred: Image.Image, gt_path: str) -> dict:
    """SSIM and LPIPS between Flux output and ground truth clean image."""
    global _lpips_fn
    if not METRICS_AVAILABLE:
        return {}

    gt = Image.open(gt_path).convert("RGB").resize(pred.size, Image.BILINEAR)

    pred_t = pil_to_tensor(pred)
    gt_t   = pil_to_tensor(gt)

    ssim_val = ssim_fn(pred_t, gt_t, data_range=1.0).item()

    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net="alex").to("cpu")
    lpips_val = _lpips_fn(
        pred_t * 2 - 1, gt_t * 2 - 1
    ).item()   # LPIPS expects [-1, 1]

    return {"ssim": round(ssim_val, 4), "lpips": round(lpips_val, 4)}


def compute_iou(pred_mask: Image.Image, gt_mask_path: str) -> float:
    """IoU between SAM3 predicted mask and GT segmap."""
    pred = np.array(pred_mask.convert("L")) > 127
    gt   = np.array(Image.open(gt_mask_path).convert("L").resize(
        pred_mask.size, Image.NEAREST)) > 127
    inter  = (pred & gt).sum()
    union  = (pred | gt).sum()
    return round(float(inter) / float(union + 1e-6), 4)


# ── SINGLE IMAGE PIPELINE ──────────────────────────────────────────────────────

def run_single(image_path: str,
               annot_mask_path: str,
               sam_model, sam_processor,
               flux_pipe,
               out_dir: str,
               entry_id: str = "result",
               gt_clean_path: str = None,
               gt_segmap_path: str = None) -> dict:
    """
    Full pipeline for one image pair.
    Returns dict with output paths and metric scores.
    """
    os.makedirs(out_dir, exist_ok=True)

    image     = Image.open(image_path).convert("RGB")
    annot_np  = np.array(Image.open(annot_mask_path).convert("L"))
    annot_pil = Image.fromarray(annot_np, mode="L")

    # Step 1: SAM3 -> object mask
    object_mask = sam_predict(sam_model, sam_processor, image, annot_np)

    # Step 2: Flux -> clean image
    clean_image = flux_remove_annotation(flux_pipe, image, annot_pil)

    # Save outputs
    clean_path = os.path.join(out_dir, f"{entry_id}_clean.png")
    mask_path  = os.path.join(out_dir, f"{entry_id}_object_mask.png")
    comp_path  = os.path.join(out_dir, f"{entry_id}_comparison.png")

    clean_image.save(clean_path)
    object_mask.save(mask_path)

    # 4-panel comparison
    prompt_box = get_prompt_box(annot_np)
    fig, axes  = plt.subplots(1, 4, figsize=(20, 5))
    data = [
        (image,      None,   "Annotated input"),
        (annot_pil,  "gray", "Annotation mask"),
        (clean_image,None,   "Clean (Flux)"),
        (object_mask,"gray", "Object mask (SAM3)"),
    ]
    for ax, (img, cmap, title) in zip(axes, data):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    x1, y1, x2, y2 = prompt_box
    axes[0].add_patch(patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="lime", facecolor="none"
    ))
    axes[0].set_title("Annotated input\n(green = SAM3 prompt box)", fontsize=11)

    plt.tight_layout()
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()

    result = {
        "clean_image": clean_path,
        "object_mask": mask_path,
        "comparison":  comp_path,
    }

    # Optional evaluation metrics
    if gt_clean_path and os.path.exists(gt_clean_path):
        result["image_metrics"] = compute_image_metrics(clean_image, gt_clean_path)
    if gt_segmap_path and os.path.exists(gt_segmap_path):
        result["iou"] = compute_iou(object_mask, gt_segmap_path)

    return result


# ── BATCH PIPELINE ─────────────────────────────────────────────────────────────

def run_batch(dataset_dir: str,
              sam_model, sam_processor,
              flux_pipe,
              out_dir: str):
    """Run pipeline over the full test split, print per-type metrics."""
    test_json = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"'{test_json}' not found.")

    with open(test_json, "r") as f:
        test_data = json.load(f)

    print(f"\nRunning pipeline on {len(test_data):,} test samples...")

    metrics_by_type = defaultdict(list)

    for entry in tqdm(test_data, desc="Inference", dynamic_ncols=True):
        image_path      = os.path.join(dataset_dir, entry["image"])
        annot_mask_path = os.path.join(dataset_dir, entry["annotation_mask"])
        gt_clean_path   = (os.path.join(dataset_dir, entry["original_clean_image"])
                           if entry.get("original_clean_image") else None)
        gt_segmap_path  = os.path.join(dataset_dir, entry["annotation"])

        entry_id  = os.path.splitext(os.path.basename(image_path))[0]
        ann_type  = entry.get("annotation_type", "unknown")
        type_dir  = os.path.join(out_dir, ann_type)

        result = run_single(
            image_path      = image_path,
            annot_mask_path = annot_mask_path,
            sam_model       = sam_model,
            sam_processor   = sam_processor,
            flux_pipe       = flux_pipe,
            out_dir         = type_dir,
            entry_id        = entry_id,
            gt_clean_path   = gt_clean_path,
            gt_segmap_path  = gt_segmap_path,
        )

        row = {"annotation_type": ann_type}
        if "image_metrics" in result:
            row.update(result["image_metrics"])
        if "iou" in result:
            row["iou"] = result["iou"]
        metrics_by_type[ann_type].append(row)

    # ── Evaluation summary ─────────────────────────────────────────────────────
    all_rows = [r for rows in metrics_by_type.values() for r in rows]
    if not all_rows or ("ssim" not in all_rows[0] and "iou" not in all_rows[0]):
        print("\nNo evaluation metrics computed (missing GT files or torchmetrics).")
    else:
        print("\nEvaluation Results")
        header = f"  {'Type':<20} {'N':>5}  {'SSIM':>7}  {'LPIPS':>7}  {'IoU':>7}"
        print(header)
        print(f"  {'-'*52}")

        all_ssim, all_lpips, all_iou = [], [], []

        for ann_type, rows in sorted(metrics_by_type.items()):
            ssims  = [r["ssim"]  for r in rows if "ssim"  in r]
            lpipss = [r["lpips"] for r in rows if "lpips" in r]
            ious   = [r["iou"]   for r in rows if "iou"   in r]

            ssim_str  = f"{np.mean(ssims):.4f}"  if ssims  else "  -    "
            lpips_str = f"{np.mean(lpipss):.4f}" if lpipss else "  -    "
            iou_str   = f"{np.mean(ious):.4f}"   if ious   else "  -    "

            print(f"  {ann_type:<20} {len(rows):>5}  {ssim_str:>7}  "
                  f"{lpips_str:>7}  {iou_str:>7}")

            all_ssim.extend(ssims)
            all_lpips.extend(lpipss)
            all_iou.extend(ious)

        print(f"  {'-'*52}")
        ssim_str  = f"{np.mean(all_ssim):.4f}"  if all_ssim  else "  -    "
        lpips_str = f"{np.mean(all_lpips):.4f}" if all_lpips else "  -    "
        iou_str   = f"{np.mean(all_iou):.4f}"   if all_iou   else "  -    "
        print(f"  {'OVERALL':<20} {len(all_rows):>5}  {ssim_str:>7}  "
              f"{lpips_str:>7}  {iou_str:>7}")

        # Save metrics to JSON for thesis tables
        metrics_path = os.path.join(out_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(dict(metrics_by_type), f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    print(f"\nResults saved to '{out_dir}/'")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SAM3 + Flux annotation removal pipeline")
    p.add_argument("--model",     required=True,
                   help="Path to fine-tuned SAM3 weights (.pth)")
    p.add_argument("--out_dir",   default="results",
                   help="Output directory")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch",  action="store_true",
                      help="Run on full test split (requires --dataset)")
    mode.add_argument("--image",
                      help="Single annotated image path")

    p.add_argument("--annot_mask",
                   help="Annotation mask path (single-image mode)")
    p.add_argument("--dataset",   default="sam_finetuning_dataset",
                   help="Dataset root dir (batch mode)")
    p.add_argument("--gt_clean",
                   help="Ground truth clean image for SSIM/LPIPS (single mode)")
    p.add_argument("--gt_segmap",
                   help="Ground truth segmap for IoU (single mode)")
    return p.parse_args()


def main():
    args = parse_args()

    sam_model, sam_processor = load_sam(args.model)
    flux_pipe                = load_flux()

    if args.batch:
        run_batch(
            dataset_dir   = args.dataset,
            sam_model     = sam_model,
            sam_processor = sam_processor,
            flux_pipe     = flux_pipe,
            out_dir       = args.out_dir,
        )
    else:
        if not args.annot_mask:
            raise ValueError("--annot_mask is required in single-image mode.")

        result = run_single(
            image_path      = args.image,
            annot_mask_path = args.annot_mask,
            sam_model       = sam_model,
            sam_processor   = sam_processor,
            flux_pipe       = flux_pipe,
            out_dir         = args.out_dir,
            entry_id        = "result",
            gt_clean_path   = args.gt_clean,
            gt_segmap_path  = args.gt_segmap,
        )

        print(f"\nDone!")
        print(f"  Clean image  -> {result['clean_image']}")
        print(f"  Object mask  -> {result['object_mask']}")
        print(f"  Comparison   -> {result['comparison']}")
        if "image_metrics" in result:
            print(f"  SSIM         : {result['image_metrics']['ssim']}")
            print(f"  LPIPS        : {result['image_metrics']['lpips']}")
        if "iou" in result:
            print(f"  IoU          : {result['iou']}")


if __name__ == "__main__":
    main()
