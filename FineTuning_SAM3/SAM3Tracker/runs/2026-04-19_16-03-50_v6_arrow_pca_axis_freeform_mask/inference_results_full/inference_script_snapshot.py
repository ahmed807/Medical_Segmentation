Sapman@2024@275
"""
inference_tracker.py
--------------------
Full inference pipeline using Sam3TrackerModel (PVS) — no text prompts needed.
All prompts are derived purely from the annotation mask geometry.

Model: Sam3TrackerModel — the SAM2-style single-instance segmenter.
       Correct for "given a spatial prompt, segment the one object there."

Pipeline:
  Step 1: annotation mask + annotation_type -> spatial prompt
  Step 2: Sam3Tracker(image, spatial prompt) -> object segmentation mask
  Step 3: Flux Fill(image, annotation mask)  -> clean image (annotation removed)

Works for any domain (natural images, medical) because it never relies
on text labels — only on annotation geometry.

================================================================================
CHANGELOG
================================================================================

[v6 — PCA arrow tip + adaptive Flux + hard composite]

1. ARROW TIP DETECTION — PCA principal-axis algorithm (v6)
   - Previous versions used convex hull "furthest pair" to find arrow endpoints.
     This failed on diagonal arrows (picked two arrowhead corners instead of
     tip vs tail) and sometimes returned coordinates OUTSIDE the annotation.
   - v6 uses PCA on all annotation pixels to find the true arrow axis.
     Min/max projections along this axis define the two endpoints.
     Each endpoint is SNAPPED to the nearest actual annotation pixel,
     guaranteeing the prompt is always on the arrow.
   - Tip vs tail: measure perpendicular spread at each end (outer 25% band).
     The arrowhead flares wider than the shaft → tip = larger spread.

2. FREEFORM BBOX — filled contour as input_masks prompt
   - Instead of a rectangular box approximation, the annotation outline is
     filled (cv2.drawContours FILLED) and passed as input_masks to SAM3.
   - Preserves the exact freeform shape, prevents SAM from segmenting the
     dashed annotation lines along with the target object.
   - rect_bbox still uses a standard box prompt (no information loss).

3. PROMPT ROUTING TABLE
     arrow         -> PCA tip pixel              -> positive point prompt
     number_letter -> sampled pixels ON object    -> multiple positive point prompts
     rect_bbox     -> annotation bounding box     -> box prompt
     freeform_bbox -> filled annotation contour   -> mask prompt (+ box context)

4. FLUX INPAINTING — per-type guidance + adaptive dilation + hard composite
   a) Per-type guidance scale:
      - arrow / number_letter: guidance=10 (moderate, good blending)
      - freeform / rect bbox:  guidance=2  (very conservative, just erase stroke)
   b) Adaptive dilation for bbox types:
      - Measures actual line thickness via distance transform.
      - Dilation = ceil(half_thickness) + 3px margin.
      - Ensures the FULL annotation stroke is covered regardless of line width.
      - Prevents leftover line fragments (thin lines) and blurry interpolation
        (when mask only partially covers the stroke).
   c) Fixed dilation for arrows/letters: 4px (small marks, safe to be generous).
   d) Morphological closing DISABLED — closing dashed outlines creates filled
      regions that Flux interprets as "regenerate everything inside."
   e) Hard composite after Flux:
      - Only take Flux's output where the mask is active.
      - Keep original image pixels everywhere else (byte-identical).
      - Gaussian feather (7x7, σ=2) at mask edges for smooth blending.
      - Prevents color shifts, person distortion, and quality degradation
        that Flux's diffusion process causes outside the masked region.
      - Flux output resized back to original dimensions before compositing
        (Flux internally resizes to multiples of 16).

5. COMPARISON PANEL — includes GT object mask
   - When GT segmap is available, a 5th panel "GT object mask" is added
     to the comparison image for visual side-by-side evaluation.

6. SCRIPT SNAPSHOT — saves a copy of this file to the output directory
   on every run for full reproducibility.

================================================================================

Usage - SAM3 only (fast, no Flux needed):
    python inference_tracker.py \\
        --model   sam3_best.pth \\
        --batch \\
        --dataset sam_finetuning_dataset \\
        --out_dir results/ \\
        --sam_only

Usage - full pipeline (SAM3 + Flux):
    python inference_tracker.py \\
        --model   sam3_best.pth \\
        --batch \\
        --dataset sam_finetuning_dataset \\
        --out_dir results/

Usage - single image:
    python inference_tracker.py \\
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
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
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
N_LETTER_POINTS   = 5

FLUX_PROMPT         = "seamless continuation of the surrounding image, no markings, no lines"
FLUX_STEPS          = 50

# Per-type Flux settings.
# The key insight: freeform/rect bbox outlines ENCLOSE objects, so Flux
# must be extremely conservative to avoid regenerating enclosed content.
# Arrows/letters are small localized marks where Flux can be more aggressive.
FLUX_GUIDANCE_ARROW  = 10    # arrows/letters: moderate guidance, good blending
FLUX_GUIDANCE_BBOX   = 2     # freeform/rect bbox: very low guidance — Flux
                              # barely modifies pixels, just erases the stroke

# Mask preprocessing for Flux Fill.
# For bbox types: ONLY dilate the raw line pixels. Never close gaps
# (closing creates filled regions that Flux regenerates entirely).
# For arrows/letters: slightly more dilation for better blending.
FLUX_MASK_DILATE_ARROW = 4    # arrows/letters get more blending margin
FLUX_MASK_DILATE_BBOX_MARGIN = 3   # bbox: dilate by (measured_thickness + margin)
                                    # ensures the FULL stroke is covered + blending room
# ───────────────────────────────────────────────────────────────────────────────


# ── PROMPT HELPERS ─────────────────────────────────────────────────────────────

def find_arrow_tip(annot_np: np.ndarray):
    """
    Finds the arrowhead tip using PCA to find the arrow's main axis,
    then measuring perpendicular spread at each end of that axis.

    Why PCA instead of convex hull:
      - Convex hull's "furthest pair" can pick two arrowhead corners
        (left vs right barb of the triangle) instead of tip vs tail,
        especially on diagonal arrows.
      - Hull vertices can fall just OUTSIDE the actual annotation pixels,
        producing prompts that miss the arrow entirely.
      - PCA finds the true direction the arrow points by analyzing all
        annotation pixels, not just the hull boundary.

    Algorithm:
      1. Compute PCA on all annotation pixels — main eigenvector = arrow axis.
      2. Project pixels onto the axis. Min/max projections = the two ends.
      3. Snap each end to the nearest ACTUAL annotation pixel (never
         return a coordinate that's outside the mask).
      4. Measure perpendicular spread at each end (within outer 25% band).
      5. Tip = end with LARGER perpendicular spread (arrowhead flare).

    Returns [x, y] coordinate that is GUARANTEED to be on the annotation,
    or None if detection fails.
    """
    ys, xs = np.where(annot_np > 127)
    if len(xs) < 2:
        if len(xs) == 1:
            return [int(xs[0]), int(ys[0])]
        return None

    pts = np.column_stack([xs, ys]).astype(float)  # shape (N, 2)

    # Step 1: PCA to find the arrow's main axis
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov      = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue's eigenvector = principal axis
    axis_dir = eigvecs[:, np.argmax(eigvals)]
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])

    # Step 2: Project all pixels onto the axis
    projections = centered @ axis_dir
    proj_min = projections.min()
    proj_max = projections.max()
    axis_len = proj_max - proj_min
    if axis_len < 1e-6:
        return [int(centroid[0]), int(centroid[1])]

    # Step 3: Snap each extreme to the nearest actual annotation pixel
    end_a_target = centroid + proj_min * axis_dir
    end_b_target = centroid + proj_max * axis_dir
    end_a = pts[np.linalg.norm(pts - end_a_target, axis=1).argmin()]
    end_b = pts[np.linalg.norm(pts - end_b_target, axis=1).argmin()]

    # Step 4: Measure perpendicular spread at each end within outer 25% band
    band_len = 0.25 * axis_len

    def perpendicular_spread(end_along_proj):
        in_band = np.abs(projections - end_along_proj) < band_len
        if not in_band.any():
            return 0.0
        perp_coords = centered[in_band] @ perp_dir
        return float(perp_coords.max() - perp_coords.min())

    spread_a = perpendicular_spread(proj_min)
    spread_b = perpendicular_spread(proj_max)

    # Step 5: Tip = end with larger perpendicular spread (arrowhead flare)
    tip = end_a if spread_a >= spread_b else end_b
    return [int(round(tip[0])), int(round(tip[1]))]


def get_points_from_mask(annot_np: np.ndarray, n_points: int = 5):
    """
    Samples N evenly-spaced points from annotation mask pixels.
    For number/letter annotations sitting ON the object surface.
    Returns list of [x, y] or None.
    """
    ys, xs = np.where(annot_np > 127)
    if len(xs) == 0:
        return None
    indices = np.linspace(0, len(xs) - 1,
                          min(n_points, len(xs)), dtype=int)
    return [[int(xs[i]), int(ys[i])] for i in indices]


def fill_annotation_contour(annot_np: np.ndarray) -> np.ndarray:
    """
    Fills the INTERIOR of a freeform annotation contour to create a
    solid mask suitable for input_masks prompt.

    The annotation mask is typically just the outline (dashed lines).
    The object is inside. This function fills the enclosed region.

    Returns: filled binary mask (H, W), same size as input, uint8 0/255.
    """
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return annot_np

    filled = np.zeros_like(annot_np)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def get_prompt_box(annot_np: np.ndarray,
                   padding: int = 10,
                   fallback: list = None) -> list:
    """Union bbox of all annotation contours + padding."""
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


def annotation_to_sam_prompt(annot_np: np.ndarray,
                              annotation_type: str) -> dict:
    """
    Converts annotation mask to Sam3Tracker spatial prompt kwargs.
    No text — purely geometric, works for any domain.

    Returns dict ready to unpack into processor(images=..., **kwargs).
    """
    if annotation_type == "arrow":
        # Arrow tip = positive point click near/on the target object
        tip = find_arrow_tip(annot_np)
        if tip:
            return {
                "input_points": [[[tip]]],
                "input_labels": [[[1]]],
            }
        # Fallback if tip detection fails
        return {"input_boxes": [[get_prompt_box(annot_np, padding=10)]]}

    elif annotation_type == "number_letter":
        # Letter/number pixels sit directly on the object surface
        points = get_points_from_mask(annot_np, n_points=N_LETTER_POINTS)
        if points:
            return {
                "input_points": [[points]],
                "input_labels": [[[1] * len(points)]],
            }
        # Fallback: expand box to give SAM3 wider context
        return {"input_boxes": [[get_prompt_box(annot_np, padding=150)]]}

    elif annotation_type == "rect_bbox":
        # Rectangular bbox — box surrounds the object directly
        return {"input_boxes": [[get_prompt_box(annot_np, padding=5)]]}

    else:
        # freeform_bbox — return box for processor + filled mask for model.
        # The filled contour preserves the exact freeform shape.
        # "_freeform_mask" is a special key handled by sam_predict.
        filled = fill_annotation_contour(annot_np)
        return {
            "input_boxes":    [[get_prompt_box(annot_np, padding=5)]],
            "_freeform_mask": filled,
        }


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

def load_sam(model_path: str, device: str = DEVICE):
    print(f"Loading Sam3TrackerModel from {model_path}...")
    processor = Sam3TrackerProcessor.from_pretrained(SAM_MODEL_ID)
    model     = Sam3TrackerModel.from_pretrained(SAM_MODEL_ID)
    ckpt      = torch.load(model_path, map_location=device)
    state     = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print("  Sam3Tracker ready.")
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
    Sam3Tracker inference using annotation-type-specific spatial prompt.
    No text needed — prompt derived purely from annotation geometry.

    For freeform_bbox: uses input_masks (filled contour) + box context.
    For all others: uses points or boxes directly.

    Returns PIL 'L' image (255=object, 0=background).
    """
    orig_w, orig_h = image.size

    # Get the correct prompt type for this annotation
    prompt_kwargs = annotation_to_sam_prompt(annot_np, annotation_type)

    # Extract freeform mask if present (not passed to processor)
    freeform_mask = prompt_kwargs.pop("_freeform_mask", None)

    try:
        inputs = processor(
            images         = image,
            return_tensors = "pt",
            **prompt_kwargs
        ).to(device)
    except Exception as e:
        # If prompts fail, fall back to box
        print(f"  Prompt failed ({e}), falling back to box prompt")
        box    = get_prompt_box(annot_np, padding=20)
        inputs = processor(
            images         = image,
            input_boxes    = [[box]],
            return_tensors = "pt"
        ).to(device)
        freeform_mask = None

    # Add filled freeform mask as input_masks for the model
    if freeform_mask is not None:
        filled_resized = cv2.resize(freeform_mask, (256, 256),
                                     interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(filled_resized > 0, dtype=torch.float32)
        # input_masks shape: (batch=1, num_objects=1, H, W)
        inputs["input_masks"] = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            outputs = model(**inputs, multimask_output=False)

    # pred_masks: (B, point_batch, num_masks, H, W)
    low_res = outputs.pred_masks.squeeze(0).squeeze(0).squeeze(0)  # -> (H, W)

    upscaled = F.interpolate(
        low_res.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear", align_corners=False
    ).squeeze()

    binary = (torch.sigmoid(upscaled) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


# ── FLUX INPAINTING ────────────────────────────────────────────────────────────

def measure_line_thickness(annot_np: np.ndarray) -> float:
    """
    Measures the typical line thickness of the annotation stroke.
    Uses the distance transform: median distance from mask pixels to the
    nearest background pixel ≈ half-thickness of the stroke.
    Returns half-thickness in pixels, or 1.0 if mask is empty.
    """
    mask = (annot_np > 127).astype(np.uint8)
    if mask.sum() == 0:
        return 1.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return float(np.median(dist[mask > 0]))


def prepare_flux_mask(annot_np: np.ndarray,
                       dilate_px: int = 3) -> Image.Image:
    """
    Prepares the annotation mask for Flux Fill inpainting.

    Strategy: dilate ONLY the raw annotation line pixels. Never
    morphologically close gaps — closing dashed outlines creates
    filled regions that Flux interprets as "regenerate everything inside."

    Args:
        dilate_px: pixels of outward expansion around the annotation stroke.

    Returns: PIL 'L' image, ready to pass to Flux as mask_image.
    """
    mask = (annot_np > 127).astype(np.uint8) * 255

    if dilate_px > 0:
        d_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, d_kernel)

    return Image.fromarray(mask, mode="L")


def flux_remove_annotation(pipe, image: Image.Image,
                            annot_mask: Image.Image,
                            guidance_scale: float = 10,
                            device: str = DEVICE) -> Image.Image:
    """
    Flux Fill: erase annotation pixels, restore clean image.

    After Flux generates its output, we HARD COMPOSITE: only take Flux's
    pixels where the mask is active, and keep the original image everywhere
    else. This prevents:
      - Color shifts on objects outside the mask
      - Distortion of people/objects near the annotation
      - Overall quality degradation from the diffusion process

    A small Gaussian feather on the mask edge ensures smooth blending
    between Flux-generated and original pixels.
    """
    orig_w, orig_h = image.size
    flux_output = pipe(
        prompt              = FLUX_PROMPT,
        image               = image,
        mask_image          = annot_mask,
        height              = orig_h,
        width               = orig_w,
        guidance_scale      = guidance_scale,
        num_inference_steps = FLUX_STEPS,
        max_sequence_length = 512,
        generator           = torch.Generator(device).manual_seed(RANDOM_SEED),
    ).images[0]

    # Hard composite: original pixels outside mask, Flux pixels inside mask.
    # Flux may resize to multiples of 16, so resize output back to original.
    if flux_output.size != image.size:
        flux_output = flux_output.resize(image.size, Image.LANCZOS)

    # Feather the mask edge (Gaussian blur) so the transition is smooth.
    mask_np = np.array(annot_mask.resize(image.size, Image.NEAREST)).astype(np.float32) / 255.0
    # Slight Gaussian blur for soft blending at edges (3px radius)
    mask_np = cv2.GaussianBlur(mask_np, (7, 7), sigmaX=2.0)
    mask_3ch = np.stack([mask_np] * 3, axis=-1)

    orig_np = np.array(image).astype(np.float32)
    flux_np = np.array(flux_output).astype(np.float32)

    # Blend: mask=1 → Flux, mask=0 → original
    composited = orig_np * (1 - mask_3ch) + flux_np * mask_3ch
    return Image.fromarray(composited.astype(np.uint8))


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

    # Step 1: Sam3Tracker -> object segmentation mask
    object_mask = sam_predict(
        sam_model, sam_processor, image, annot_np, annotation_type)

    # Step 2: Inpaint annotation away (skip when sam_only)
    if sam_only or flux_pipe is None:
        clean_image = image
        flux_ran    = False
    else:
        # Per-type Flux settings:
        #   arrow/number_letter: moderate guidance + fixed dilation (small marks)
        #   freeform/rect bbox:  very low guidance + ADAPTIVE dilation
        #       Dilation = measured_line_thickness + margin, so the mask
        #       always covers the full stroke width + blending room.
        #       This prevents leftover line fragments (bag case) and
        #       blurry interpolation (baseball case).
        if annotation_type in ("freeform_bbox", "rect_bbox"):
            half_thick = measure_line_thickness(annot_np)
            adaptive_dilate = int(np.ceil(half_thick)) + FLUX_MASK_DILATE_BBOX_MARGIN
            flux_mask   = prepare_flux_mask(annot_np, dilate_px=adaptive_dilate)
            clean_image = flux_remove_annotation(
                flux_pipe, image, flux_mask, guidance_scale=FLUX_GUIDANCE_BBOX)
        else:
            flux_mask   = prepare_flux_mask(annot_np, dilate_px=FLUX_MASK_DILATE_ARROW)
            clean_image = flux_remove_annotation(
                flux_pipe, image, flux_mask, guidance_scale=FLUX_GUIDANCE_ARROW)
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
    prompt_kwargs = annotation_to_sam_prompt(annot_np, annotation_type)
    freeform_vis_mask = prompt_kwargs.pop("_freeform_mask", None)

    # Load GT object mask if available (for visual comparison)
    gt_object_mask = None
    if gt_segmap_path and os.path.exists(gt_segmap_path):
        gt_object_mask = Image.open(gt_segmap_path).convert("L")

    # Build panel list dynamically
    panels = [
        (image,       None,   f"Annotated input\n({annotation_type})"),
        (annot_pil,   "gray", "Annotation mask"),
    ]
    if not sam_only:
        panels.append((clean_image, None, "Clean (Flux)"))
    panels.append((object_mask, "gray", "Object mask (SAM3)"))
    if gt_object_mask is not None:
        panels.append((gt_object_mask, "gray", "GT object mask"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Draw prompt visualisation on first panel
    ax0 = axes[0]
    if freeform_vis_mask is not None:
        # Show filled freeform mask as semi-transparent overlay
        mask_rgba = np.zeros((*freeform_vis_mask.shape, 4), dtype=np.uint8)
        mask_rgba[freeform_vis_mask > 0] = [0, 255, 0, 80]  # green, 30% opacity
        ax0.imshow(mask_rgba)
        ax0.set_title(f"Annotated input ({annotation_type})\n"
                      f"(green = SAM3 mask prompt)", fontsize=9)
    elif "input_boxes" in prompt_kwargs:
        x1, y1, x2, y2 = prompt_kwargs["input_boxes"][0][0]
        ax0.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="lime", facecolor="none"))
        ax0.set_title(f"Annotated input ({annotation_type})\n"
                      f"(green = SAM3 box prompt)", fontsize=9)
    elif "input_points" in prompt_kwargs:
        pts = prompt_kwargs["input_points"][0][0]
        if isinstance(pts[0], list):
            for px, py in pts:
                ax0.plot(px, py, "go", markersize=6)
        else:
            px, py = pts
            ax0.plot(px, py, "go", markersize=8)
        ax0.set_title(f"Annotated input ({annotation_type})\n"
                      f"(green dots = SAM3 point prompts)", fontsize=9)

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
        description="SAM3 Tracker + Flux annotation removal pipeline")
    p.add_argument("--model",    required=True,
                   help="Path to fine-tuned Sam3Tracker weights (.pth)")
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


def save_script_snapshot(out_dir: str):
    """
    Saves a copy of this inference script to the output directory.
    Ensures full reproducibility — you can always see exactly which
    version of the code produced a given set of results.
    """
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    if os.path.exists(script_path):
        dest = os.path.join(out_dir, "inference_script_snapshot.py")
        shutil.copy2(script_path, dest)
        print(f"  Script snapshot -> {dest}")


def main():
    args = parse_args()

    # Save a snapshot of this script for reproducibility
    save_script_snapshot(args.out_dir)

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
