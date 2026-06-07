"""
inference.py  (Sam3Model — PCS detector version)
--------------------------------------------------
Full inference pipeline using Sam3Model (PCS detector).
All prompts are exemplar boxes derived from annotation mask geometry.

Model: Sam3Model — the DETR detector for Promptable Concept Segmentation.
       Outputs multiple query masks; we select the highest-scoring one.

Prompting (all annotation types → exemplar boxes):
  arrow         -> box around arrowhead tip area  -> positive exemplar box
  number_letter -> box around annotation pixels    -> positive exemplar box
  rect_bbox     -> annotation bounding box         -> positive exemplar box
  freeform_bbox -> annotation bounding box         -> positive exemplar box

Pipeline:
  Step 1: annotation mask + annotation_type -> exemplar box
  Step 2: Sam3Model(image, exemplar box)    -> multi-query masks
  Step 3: Select best mask by confidence score
  Step 4: Flux Fill(image, annotation mask) -> clean image (optional)

No text at inference time — purely spatial exemplar prompts.

Usage - SAM3 only:
    python inference.py \\
        --model   sam3_best.pth \\
        --batch \\
        --dataset sam_finetuning_dataset \\
        --out_dir results/ \\
        --sam_only

Usage - full pipeline (SAM3 + Flux):
    python inference.py \\
        --model   sam3_best.pth \\
        --batch \\
        --dataset sam_finetuning_dataset \\
        --out_dir results/

Usage - single image:
    python inference.py \\
        --model          sam3_best.pth \\
        --image          path/to/annotated.jpg \\
        --annot_mask     path/to/annotation_mask.png \\
        --annotation_type arrow \\
        --out_dir        results/
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
from tqdm         import tqdm
from collections  import defaultdict
from huggingface_hub import login

# ── HF LOGIN ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    raise EnvironmentError("Run: export HF_TOKEN='hf_...' first")
login(token=HF_TOKEN)
print("Logged in to Hugging Face.")

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

ARROW_TIP_BOX_FRAC = 0.05
ARROW_TIP_BOX_MIN  = 20

SCORE_THRESHOLD   = 0.1    # Min score to consider a query detection
MASK_THRESHOLD    = 0.5    # Binarization threshold for predicted masks

FLUX_PROMPT         = "a clean photo, no annotations, no arrows, no labels"
FLUX_GUIDANCE_SCALE = 30
FLUX_STEPS          = 50
# ───────────────────────────────────────────────────────────────────────────────


# ── PROMPT HELPERS ─────────────────────────────────────────────────────────────

def find_arrow_tip(annot_np: np.ndarray):
    """Finds the arrowhead tip — contour point furthest from centroid."""
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts   = np.vstack(contours).reshape(-1, 2).astype(float)
    centroid  = all_pts.mean(axis=0)
    distances = np.linalg.norm(all_pts - centroid, axis=1)
    tip       = all_pts[distances.argmax()]
    return [int(tip[0]), int(tip[1])]


def tip_to_box(tip, img_shape,
               fraction: float = ARROW_TIP_BOX_FRAC,
               min_size: int = ARROW_TIP_BOX_MIN) -> list:
    """Converts arrow tip to exemplar box around tip area (xyxy)."""
    x, y = tip
    img_h, img_w = img_shape[:2]
    half = max(min_size, int(min(img_h, img_w) * fraction))
    return [
        max(0,     x - half),
        max(0,     y - half),
        min(img_w, x + half),
        min(img_h, y + half),
    ]


def get_prompt_box(annot_np: np.ndarray,
                   padding: int = 10,
                   fallback: list = None) -> list:
    """Union bbox of annotation contours + padding (xyxy)."""
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def annotation_to_exemplar_box(annot_np: np.ndarray,
                                annotation_type: str) -> list:
    """
    Converts any annotation to a positive exemplar box (xyxy pixels).
    For Sam3Model, all annotation types become positive exemplar boxes.
    """
    if annotation_type == "arrow":
        tip = find_arrow_tip(annot_np)
        if tip:
            return tip_to_box(tip, annot_np.shape)
        return get_prompt_box(annot_np, padding=10)

    elif annotation_type == "number_letter":
        return get_prompt_box(annot_np, padding=20)

    else:
        return get_prompt_box(annot_np, padding=5)


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

def load_sam(model_path: str, device: str = DEVICE):
    # Check if the user requested the base model
    is_base = (model_path == SAM_MODEL_ID or model_path.lower() == "base")

    if is_base:
        print(f"Loading ORIGINAL base Sam3Model (PCS) from {SAM_MODEL_ID}...")
    else:
        print(f"Loading FINE-TUNED Sam3Model (PCS) from {model_path}...")

    # Always load the base architecture and processor first
    processor = Sam3Processor.from_pretrained(SAM_MODEL_ID)
    model     = Sam3Model.from_pretrained(SAM_MODEL_ID)

    # Only load local .pth weights if we aren't using the base model
    if not is_base:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find fine-tuned weights at {model_path}")
        ckpt  = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)

    model.to(device).eval()
    print("  Sam3Model ready.")
    return model, processor


def load_flux(device: str = DEVICE):
    from diffusers import FluxFillPipeline
    print(f"Loading Flux Fill from {FLUX_MODEL_ID}...")
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    print("  Flux Fill ready.")
    return pipe


# ── CORE SAM3 INFERENCE ────────────────────────────────────────────────────────

def sam_predict(model, processor,
                image: Image.Image,
                annot_np: np.ndarray,
                annotation_type: str,
                device: str = DEVICE) -> Image.Image:
    """
    Sam3Model inference using exemplar box prompt.
    No text — purely geometric exemplar, works for any domain.

    Selects the highest-scoring detection from multi-query output.
    Returns PIL 'L' image (255=object, 0=background).
    """
    orig_w, orig_h = image.size

    exemplar_box = annotation_to_exemplar_box(annot_np, annotation_type)

    inputs = processor(
        images             = image,
        input_boxes        = [[exemplar_box]],
        input_boxes_labels = [[1]],
        return_tensors     = "pt"
    ).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            outputs = model(**inputs)

    # ── Select best mask from multi-query output ──────────────────────
    # Use post_process_instance_segmentation for proper handling
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SCORE_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=[[orig_h, orig_w]]
    )[0]

    if len(results["masks"]) > 0:
        # Take the mask with highest score
        best_idx = results["scores"].argmax()
        mask_np  = results["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
    else:
        # Fallback: take raw best query by logit score
        pred_masks = outputs.pred_masks[0]  # (num_queries, H, W)
        if hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
            best_idx = outputs.pred_logits[0].argmax()
        else:
            # No logits → take first query
            best_idx = 0

        best_mask = pred_masks[best_idx]
        upscaled = F.interpolate(
            best_mask.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        ).squeeze()
        mask_np = (torch.sigmoid(upscaled) > 0.5).cpu().numpy().astype(np.uint8) * 255

    return Image.fromarray(mask_np, mode="L")


# ── FLUX INPAINTING ────────────────────────────────────────────────────────────

def flux_remove_annotation(pipe, image: Image.Image,
                            annot_mask: Image.Image,
                            device: str = DEVICE) -> Image.Image:
    """Flux Fill: erase annotation pixels, restore clean image."""
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

_lpips_fn = None

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.tensor(
        np.array(img.convert("RGB")) / 255.0, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)


def compute_image_metrics(pred: Image.Image, gt_path: str) -> dict:
    if not METRICS_AVAILABLE:
        return {}
    global _lpips_fn
    gt     = Image.open(gt_path).convert("RGB").resize(pred.size, Image.BILINEAR)
    pred_t = pil_to_tensor(pred)
    gt_t   = pil_to_tensor(gt)
    ssim_val = ssim_fn(pred_t, gt_t, data_range=1.0).item()
    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net="alex").to("cpu")
    lpips_val = _lpips_fn(pred_t * 2 - 1, gt_t * 2 - 1).item()
    return {"ssim": round(ssim_val, 4), "lpips": round(lpips_val, 4)}


def compute_iou(pred_mask: Image.Image, gt_mask_path: str) -> float:
    pred  = np.array(pred_mask.convert("L")) > 127
    gt    = np.array(Image.open(gt_mask_path).convert("L").resize(
        pred_mask.size, Image.NEAREST)) > 127
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return round(float(inter) / float(union + 1e-6), 4)


# ── SINGLE IMAGE PIPELINE ──────────────────────────────────────────────────────

def run_single(image_path: str,
               annot_mask_path: str,
               annotation_type: str,
               sam_model, sam_processor,
               flux_pipe,
               out_dir: str,
               entry_id: str = "result",
               gt_clean_path: str = None,
               gt_segmap_path: str = None,
               sam_only: bool = False) -> dict:

    os.makedirs(out_dir, exist_ok=True)

    image     = Image.open(image_path).convert("RGB")
    annot_np  = np.array(Image.open(annot_mask_path).convert("L"))
    annot_pil = Image.fromarray(annot_np, mode="L")

    # Step 1: Sam3Model -> object segmentation mask
    object_mask = sam_predict(
        sam_model, sam_processor, image, annot_np, annotation_type)

    # Step 2: Flux -> clean image (skip when sam_only)
    if sam_only or flux_pipe is None:
        clean_image = image
        flux_ran    = False
    else:
        clean_image = flux_remove_annotation(flux_pipe, image, annot_pil)
        flux_ran    = True

    # Save outputs
    mask_path  = os.path.join(out_dir, f"{entry_id}_object_mask.png")
    comp_path  = os.path.join(out_dir, f"{entry_id}_comparison.png")
    object_mask.save(mask_path)

    clean_path = None
    if flux_ran:
        clean_path = os.path.join(out_dir, f"{entry_id}_clean.png")
        clean_image.save(clean_path)

    # Comparison panel
    exemplar_box = annotation_to_exemplar_box(annot_np, annotation_type)
    n_panels = 3 if sam_only else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    panels = [
        (image,       None,   f"Annotated input\n({annotation_type})"),
        (annot_pil,   "gray", "Annotation mask"),
        (object_mask, "gray", "Object mask (SAM3)"),
    ]
    if not sam_only:
        panels.insert(2, (clean_image, None, "Clean (Flux)"))

    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Draw exemplar box on first panel
    ax0 = axes[0]
    x1, y1, x2, y2 = exemplar_box
    ax0.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor="lime", facecolor="none"))
    ax0.set_title(f"Annotated input ({annotation_type})\n"
                  f"(green = exemplar box prompt)", fontsize=9)

    plt.tight_layout()
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()

    result = {"object_mask": mask_path, "comparison": comp_path}
    if clean_path:
        result["clean_image"] = clean_path

    if flux_ran and gt_clean_path and os.path.exists(gt_clean_path):
        result["image_metrics"] = compute_image_metrics(clean_image, gt_clean_path)
    if gt_segmap_path and os.path.exists(gt_segmap_path):
        result["iou"] = compute_iou(object_mask, gt_segmap_path)

    return result


# ── BATCH PIPELINE ─────────────────────────────────────────────────────────────

def run_batch(dataset_dir: str,
              sam_model, sam_processor,
              flux_pipe, out_dir: str,
              sam_only: bool = False):

    test_json = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"'{test_json}' not found.")

    with open(test_json) as f:
        test_data = json.load(f)

    mode_str = "SAM3 only" if sam_only else "SAM3 + Flux"
    print(f"\nRunning {mode_str} on {len(test_data):,} test samples...")

    metrics_by_type = defaultdict(list)

    for entry in tqdm(test_data, desc="Inference", dynamic_ncols=True):
        image_path      = os.path.join(dataset_dir, entry["image"])
        annot_mask_path = os.path.join(dataset_dir, entry["annotation_mask"])
        annotation_type = entry.get("annotation_type", "arrow")
        gt_clean_path   = (os.path.join(dataset_dir, entry["original_clean_image"])
                           if entry.get("original_clean_image") else None)
        gt_segmap_path  = os.path.join(dataset_dir, entry["annotation"])

        entry_id = os.path.splitext(os.path.basename(image_path))[0]
        type_dir = os.path.join(out_dir, annotation_type)

        result = run_single(
            image_path      = image_path,
            annot_mask_path = annot_mask_path,
            annotation_type = annotation_type,
            sam_model       = sam_model,
            sam_processor   = sam_processor,
            flux_pipe       = flux_pipe,
            out_dir         = type_dir,
            entry_id        = entry_id,
            gt_clean_path   = gt_clean_path,
            gt_segmap_path  = gt_segmap_path,
            sam_only        = sam_only,
        )

        row = {"annotation_type": annotation_type}
        if "image_metrics" in result:
            row.update(result["image_metrics"])
        if "iou" in result:
            row["iou"] = result["iou"]
        metrics_by_type[annotation_type].append(row)

    # ── Per-type evaluation summary ────────────────────────────────────────────
    all_rows = [r for rows in metrics_by_type.values() for r in rows]
    has_iou  = any("iou"  in r for r in all_rows)
    has_ssim = any("ssim" in r for r in all_rows)

    if has_iou or has_ssim:
        print("\nEvaluation Results")
        print(f"  {'Type':<20} {'N':>5}  {'SSIM':>7}  {'LPIPS':>7}  {'IoU':>7}")
        print(f"  {'-'*52}")

        all_ssim, all_lpips, all_iou = [], [], []
        for ann_type, rows in sorted(metrics_by_type.items()):
            ssims  = [r["ssim"]  for r in rows if "ssim"  in r]
            lpipss = [r["lpips"] for r in rows if "lpips" in r]
            ious   = [r["iou"]   for r in rows if "iou"   in r]
            ss = f"{np.mean(ssims):.4f}"  if ssims  else "  -    "
            lp = f"{np.mean(lpipss):.4f}" if lpipss else "  -    "
            io = f"{np.mean(ious):.4f}"   if ious   else "  -    "
            print(f"  {ann_type:<20} {len(rows):>5}  {ss:>7}  {lp:>7}  {io:>7}")
            all_ssim.extend(ssims); all_lpips.extend(lpipss); all_iou.extend(ious)

        print(f"  {'-'*52}")
        ss = f"{np.mean(all_ssim):.4f}"  if all_ssim  else "  -    "
        lp = f"{np.mean(all_lpips):.4f}" if all_lpips else "  -    "
        io = f"{np.mean(all_iou):.4f}"   if all_iou   else "  -    "
        print(f"  {'OVERALL':<20} {len(all_rows):>5}  {ss:>7}  {lp:>7}  {io:>7}")

        metrics_path = os.path.join(out_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(dict(metrics_by_type), f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    else:
        print("\nNo evaluation metrics computed.")

    print(f"\nResults saved to '{out_dir}/'")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sam3Model (PCS) + Flux annotation removal pipeline")
    p.add_argument("--model",    required=True,
                   help="Path to fine-tuned Sam3Model weights (.pth)")
    p.add_argument("--out_dir",  default="results")
    p.add_argument("--sam_only", action="store_true",
                   help="Run SAM3 only — skip Flux Fill")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch",  action="store_true",
                      help="Run on full test split")
    mode.add_argument("--image",  help="Single annotated image path")

    p.add_argument("--annot_mask",
                   help="Annotation mask path (single-image mode)")
    p.add_argument("--annotation_type",
                   choices=["arrow", "number_letter", "rect_bbox", "freeform_bbox"],
                   default="arrow",
                   help="Annotation type (single-image mode)")
    p.add_argument("--dataset",  default="sam_finetuning_dataset")
    p.add_argument("--gt_clean", help="GT clean image for metrics (single mode)")
    p.add_argument("--gt_segmap",help="GT segmap for IoU (single mode)")
    return p.parse_args()


def main():
    args = parse_args()

    sam_model, sam_processor = load_sam(args.model)
    flux_pipe = None if args.sam_only else load_flux()

    if args.sam_only:
        print("SAM3-only mode — Flux Fill not loaded.")

    if args.batch:
        run_batch(
            dataset_dir   = args.dataset,
            sam_model     = sam_model,
            sam_processor = sam_processor,
            flux_pipe     = flux_pipe,
            out_dir       = args.out_dir,
            sam_only      = args.sam_only,
        )
    else:
        if not args.annot_mask:
            raise ValueError("--annot_mask is required in single-image mode.")

        result = run_single(
            image_path      = args.image,
            annot_mask_path = args.annot_mask,
            annotation_type = args.annotation_type,
            sam_model       = sam_model,
            sam_processor   = sam_processor,
            flux_pipe       = flux_pipe,
            out_dir         = args.out_dir,
            entry_id        = "result",
            gt_clean_path   = args.gt_clean,
            gt_segmap_path  = args.gt_segmap,
            sam_only        = args.sam_only,
        )

        print(f"\nDone!")
        print(f"  Object mask -> {result['object_mask']}")
        print(f"  Comparison  -> {result['comparison']}")
        if "clean_image" in result:
            print(f"  Clean image -> {result['clean_image']}")
        if "image_metrics" in result:
            print(f"  SSIM  : {result['image_metrics']['ssim']}")
            print(f"  LPIPS : {result['image_metrics']['lpips']}")
        if "iou" in result:
            print(f"  IoU   : {result['iou']}")


if __name__ == "__main__":
    main()
