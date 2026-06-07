"""
train.py  (Sam3Model — PCS detector version)
----------------------------------------------
Fine-tunes the Sam3Model DETR-based detector for annotation-to-object segmentation.

Model: Sam3Model — the DETR detector that performs Promptable Concept Segmentation.
       Accepts text prompts and/or exemplar boxes, outputs up to 200 query masks.

Prompting strategy (all annotation types → exemplar boxes):
  arrow         -> box around arrowhead tip area  -> positive exemplar box
  number_letter -> box around annotation pixels    -> positive exemplar box
  rect_bbox     -> annotation bounding box         -> positive exemplar box
  freeform_bbox -> annotation bounding box         -> positive exemplar box

Text handling:
  - During training: text label (e.g., "person") is passed alongside the
    exemplar box with configurable batch-level dropout (TEXT_DROP_PROB).
  - During inference: no text (matches the inference constraint).

Multi-query output:
  - Sam3Model outputs pred_masks of shape (B, num_queries, H, W).
  - During training we match each GT to the best-scoring query (highest IoU)
    and compute loss only on the matched query.

Frozen  : vision_encoder, text_encoder, text_projection
Trained : geometry_encoder, detr_encoder, detr_decoder,
          mask_decoder, dot_product_scoring

Loss    : BCE-with-logits + Dice on matched query mask
          + BCE on matched query logit (detection score)
Hardware: NVIDIA A40 46GB

Usage:
    python train.py
    python train.py --resume sam3_checkpoint_ep25.pth
"""

import os
import json
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL              import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim      import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers     import Sam3Processor, Sam3Model
from tqdm             import tqdm
from huggingface_hub  import login

# ── HF LOGIN ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    raise EnvironmentError(
        "HF_TOKEN not set. Run: export HF_TOKEN='hf_...'"
    )
login(token=HF_TOKEN)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False
    print("TensorBoard not available.  pip install tensorboard")


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DATASET_DIR   = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset"
MODEL_ID      = "facebook/sam3"
DEVICE        = "cuda:0"

BATCH_SIZE    = 4
GRAD_ACCUM    = 4
NUM_EPOCHS    = 100
LR            = 1e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 8
PIN_MEMORY    = True
PREFETCH      = 4

# Arrow tip box: fraction of shorter image dim, min pixel size
ARROW_TIP_BOX_FRAC = 0.05
ARROW_TIP_BOX_MIN  = 20

# Text dropout: probability of dropping text for the entire batch.
# Trains the model to work without text (matching inference).
TEXT_DROP_PROB = 0.5

# Loss weighting
MASK_LOSS_WEIGHT  = 1.0
SCORE_LOSS_WEIGHT = 0.5

CHECKPOINT_EVERY = 25
BEST_MODEL_PATH  = "sam3_best.pth"
LOG_DIR          = "runs/sam3_pcs_finetune"

# Freeze heavy pre-trained encoders.
# Train: geometry_encoder, DETR encoder/decoder, mask_decoder, scoring.
FROZEN_PREFIXES = [
    "vision_encoder",
    "text_encoder",
    "text_projection",
]
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
    """Converts arrow tip to an exemplar box around the tip area (xyxy)."""
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
    """Union bbox of all annotation contours + padding (xyxy pixel format)."""
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if fallback:
            return fallback
        h, w = annot_np.shape
        return [0, 0, w, h]
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
                                annotation_type: str,
                                fallback_box: list) -> list:
    """
    Converts any annotation type to an exemplar box (xyxy pixel format).
    For Sam3Model, all annotation types become positive exemplar boxes.
    """
    if annotation_type == "arrow":
        tip = find_arrow_tip(annot_np)
        if tip:
            return tip_to_box(tip, annot_np.shape)
        return get_prompt_box(annot_np, padding=10, fallback=fallback_box)

    elif annotation_type == "number_letter":
        # Annotation sits on the object — box captures object surface
        return get_prompt_box(annot_np, padding=20, fallback=fallback_box)

    else:
        # rect_bbox / freeform_bbox — box surrounds the object
        return get_prompt_box(annot_np, padding=5, fallback=fallback_box)


# ── DATASET ────────────────────────────────────────────────────────────────────

class AnnotationToSegDataset(Dataset):
    """
    Sam3Model (PCS) fine-tuning dataset.

    Every annotation type is converted to a positive exemplar box:
    the model learns "the object of interest is in/near this region."

    Text prompt (object label) is stored per entry; whether to actually
    pass it during training is controlled by TEXT_DROP_PROB at batch level.

    GT mask: segmentation of the TARGET object (annotation_segmap).
    """

    def __init__(self, json_file: str, root_dir: str, processor,
                 augment: bool = False):
        self.root_dir  = root_dir
        self.processor = processor
        self.augment   = augment
        with open(json_file) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path      = os.path.join(self.root_dir, item["image"])
        segmap_path     = os.path.join(self.root_dir, item["annotation"])
        annot_mask_path = os.path.join(self.root_dir, item["annotation_mask"])

        image   = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        gt_mask = np.array(Image.open(segmap_path).convert("L"))
        annot_np = np.array(Image.open(annot_mask_path).convert("L"))

        annotation_type = item.get("annotation_type", "arrow")
        prompt_text     = item.get("prompt_text", "")

        # Optional horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            image    = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_mask  = np.fliplr(gt_mask).copy()
            annot_np = np.fliplr(annot_np).copy()

        # ── Derive exemplar box from annotation ──────────────────────────
        exemplar_box = annotation_to_exemplar_box(
            annot_np, annotation_type, item["prompt_box"]
        )

        # ── Process image (no text here — text handled at batch level) ───
        inputs = self.processor(
            images             = image,
            input_boxes        = [[exemplar_box]],
            input_boxes_labels = [[1]],
            return_tensors     = "pt"
        )

        # Squeeze batch dimension from processor outputs
        result = {k: v.squeeze(0) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # GT mask resized to model's internal mask resolution
        resized_gt = cv2.resize(gt_mask, (1024, 1024),
                                interpolation=cv2.INTER_NEAREST)
        result["ground_truth_mask"] = torch.tensor(resized_gt > 0).float()
        result["original_size"]     = torch.tensor([orig_h, orig_w])
        result["annotation_type"]   = annotation_type
        result["prompt_text"]       = prompt_text if prompt_text else ""
        result["image_path"]        = image_path

        return result


def sam_collate_fn(batch):
    """
    Collate for Sam3Model. All samples have the same prompt format
    (exemplar boxes), so no grouping needed.
    """
    result = {
        "pixel_values":      torch.stack([b["pixel_values"]      for b in batch]),
        "input_boxes":       torch.stack([b["input_boxes"]       for b in batch]),
        "input_boxes_labels":torch.stack([b["input_boxes_labels"]for b in batch]),
        "ground_truth_mask": torch.stack([b["ground_truth_mask"] for b in batch]),
        "original_size":     torch.stack([b["original_size"]     for b in batch]),
        "annotation_type":   [b["annotation_type"] for b in batch],
        "prompt_text":       [b["prompt_text"]     for b in batch],
        "image_path":        [b["image_path"]      for b in batch],
    }
    return result


# ── LOSS ───────────────────────────────────────────────────────────────────────

def dice_loss(inputs, targets, smooth: float = 1.0):
    probs = torch.sigmoid(inputs).view(-1)
    tgts  = targets.view(-1)
    inter = (probs * tgts).sum()
    return 1 - (2. * inter + smooth) / (probs.sum() + tgts.sum() + smooth)


def mask_loss(pred, target):
    """BCE-with-logits + Dice on a single query mask."""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dc  = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dc


def compute_iou_batch(pred_masks: torch.Tensor,
                       gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between each query mask and a single GT mask.
    pred_masks: (num_queries, H, W)  — logits
    gt_mask:    (H, W)               — binary
    Returns: (num_queries,) IoU scores.
    """
    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
    gt_expanded = gt_mask.unsqueeze(0).expand_as(pred_binary)

    inter = (pred_binary * gt_expanded).sum(dim=(1, 2))
    union = ((pred_binary + gt_expanded) > 0).float().sum(dim=(1, 2))
    return inter / (union + 1e-6)


# ── FORWARD PASS ───────────────────────────────────────────────────────────────

def forward_pass(model, processor, batch, device, use_text: bool = True):
    """
    Forward pass for Sam3Model.

    1. Run model with exemplar box + optional text
    2. pred_masks: (B, num_queries, H, W) — multiple candidate detections
    3. For each sample, match best query to GT (highest IoU)
    4. Compute loss on matched query mask + detection score
    """
    pv         = batch["pixel_values"].to(device, non_blocking=True)
    boxes      = batch["input_boxes"].to(device, non_blocking=True)
    box_labels = batch["input_boxes_labels"].to(device, non_blocking=True)
    gt         = batch["ground_truth_mask"].to(device, non_blocking=True)

    kwargs = dict(
        pixel_values      = pv,
        input_boxes       = boxes,
        input_boxes_labels = box_labels,
    )

    # ── Optional text prompt ──────────────────────────────────────────
    if use_text and any(t != "" for t in batch["prompt_text"]):
        texts = [t if t else None for t in batch["prompt_text"]]
        text_inputs = processor.tokenizer(
            [t for t in texts if t is not None],
            padding=True, return_tensors="pt"
        )
        # Only pass text if we have valid text for all samples in batch
        if len([t for t in texts if t is not None]) == len(texts):
            kwargs["input_ids"]      = text_inputs["input_ids"].to(device)
            kwargs["attention_mask"] = text_inputs["attention_mask"].to(device)

    with torch.amp.autocast("cuda"):
        outputs = model(**kwargs)

        # pred_masks: (B, num_queries, H, W)
        pred_masks = outputs.pred_masks
        B, N, mH, mW = pred_masks.shape

        # Upscale predictions to GT resolution
        gt_H, gt_W = gt.shape[1], gt.shape[2]
        pred_up = F.interpolate(
            pred_masks, size=(gt_H, gt_W),
            mode="bilinear", align_corners=False
        )  # (B, N, gt_H, gt_W)

        # ── Match best query per sample ───────────────────────────────
        total_mask_loss  = torch.tensor(0.0, device=device)
        total_score_loss = torch.tensor(0.0, device=device)
        matched_ious = []

        for i in range(B):
            # IoU between each query and GT for this sample
            ious = compute_iou_batch(pred_up[i], gt[i])  # (N,)
            best_idx = ious.argmax()
            matched_ious.append(ious[best_idx].item())

            # Mask loss on best-matching query
            total_mask_loss = total_mask_loss + mask_loss(
                pred_up[i, best_idx], gt[i]
            )

            # Score loss: matched query should have high confidence
            if hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
                logits = outputs.pred_logits[i]  # (N,)
                target_scores = torch.zeros_like(logits)
                target_scores[best_idx] = 1.0
                total_score_loss = total_score_loss + \
                    F.binary_cross_entropy_with_logits(logits, target_scores)

        total_mask_loss  = total_mask_loss / B
        total_score_loss = total_score_loss / B
        loss = (MASK_LOSS_WEIGHT  * total_mask_loss +
                SCORE_LOSS_WEIGHT * total_score_loss)

    return loss, float(np.mean(matched_ious))


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading Sam3Model (PCS detector) from {MODEL_ID}...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model     = Sam3Model.from_pretrained(MODEL_ID)

    top_modules = sorted(set(n.split('.')[0] for n, _ in model.named_parameters()))
    print(f"  Top-level modules: {top_modules}")

    for name, param in model.named_parameters():
        param.requires_grad = not any(
            name.startswith(p) for p in FROZEN_PREFIXES)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Frozen    : {total_p-trainable:,}  "
          f"({100*(total_p-trainable)/total_p:.1f}%)")
    print(f"  Trainable : {trainable:,}  ({100*trainable/total_p:.1f}%)")
    print(f"  Training  : {sorted(set(n.split('.')[0] for n,p in model.named_parameters() if p.requires_grad))}")

    model.to(DEVICE)
    print(f"  Device    : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    # ── Data ───────────────────────────────────────────────────────────────────
    for split in ["train.json", "val.json"]:
        if not os.path.exists(os.path.join(DATASET_DIR, split)):
            raise FileNotFoundError(f"'{split}' not found. Run prepare_dataset.py first.")

    print("\nLoading datasets...")
    train_dataset = AnnotationToSegDataset(
        os.path.join(DATASET_DIR, "train.json"),
        DATASET_DIR, processor, augment=True)
    val_dataset   = AnnotationToSegDataset(
        os.path.join(DATASET_DIR, "val.json"),
        DATASET_DIR, processor, augment=False)

    print(f"  Train : {len(train_dataset):,}")
    print(f"  Val   : {len(val_dataset):,}")
    print(f"  Effective batch : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  Text dropout    : {TEXT_DROP_PROB:.0%}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH, persistent_workers=True,
        collate_fn=sam_collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH, persistent_workers=True,
        collate_fn=sam_collate_fn
    )

    # ── Optimizer + Scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch   = ckpt["epoch"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            train_losses  = ckpt.get("train_losses",  [])
            val_losses    = ckpt.get("val_losses",    [])
            print(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.4f}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded weights only.")

    writer = SummaryWriter(LOG_DIR) if TENSORBOARD else None

    # ── Training loop ──────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    print(f"\nStarting: epochs {start_epoch+1}-{NUM_EPOCHS}  "
          f"({steps_per_epoch:,} steps/epoch)\n")

    import shutil as _shutil

    def safe_save(obj, path):
        """Atomic save: write to temp then rename."""
        d = os.path.dirname(os.path.abspath(path))
        free_gb = _shutil.disk_usage(d).free / 1024**3
        if free_gb < 5.0:
            raise RuntimeError(
                f"Disk too full to save: {free_gb:.1f} GB free at {d}")
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            os.close(fd)
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_train = []
        epoch_ious  = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch+1:03d}/{NUM_EPOCHS} [train]",
                    leave=False, dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            # Batch-level text dropout
            use_text = random.random() > TEXT_DROP_PROB

            loss, mean_iou = forward_pass(
                model, processor, batch, DEVICE, use_text=use_text)
            loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_display = loss.item() * GRAD_ACCUM
            epoch_train.append(loss_display)
            epoch_ious.append(mean_iou)
            pbar.set_postfix(loss=f"{loss_display:.4f}", iou=f"{mean_iou:.3f}")

        scheduler.step()
        avg_train = float(np.mean(epoch_train))
        avg_iou   = float(np.mean(epoch_ious))
        train_losses.append(avg_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        epoch_val = []
        val_ious  = []
        with torch.no_grad():
            for batch in tqdm(val_loader,
                               desc=f"Ep {epoch+1:03d}/{NUM_EPOCHS} [val]  ",
                               leave=False, dynamic_ncols=True):
                # Validate without text (matches inference)
                loss, mean_iou = forward_pass(
                    model, processor, batch, DEVICE, use_text=False)
                epoch_val.append(loss.item())
                val_ious.append(mean_iou)

        avg_val = float(np.mean(epoch_val))
        avg_val_iou = float(np.mean(val_ious))
        val_losses.append(avg_val)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Ep {epoch+1:03d}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  iou_t={avg_iou:.3f}  "
              f"iou_v={avg_val_iou:.3f}  lr={lr_now:.2e}")

        if writer:
            writer.add_scalars("loss", {"train": avg_train, "val": avg_val}, epoch+1)
            writer.add_scalars("iou", {"train": avg_iou, "val": avg_val_iou}, epoch+1)
            writer.add_scalar("lr", lr_now, epoch+1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            safe_save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Saved best -> {BEST_MODEL_PATH}  (val={best_val_loss:.4f})")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = f"sam3_checkpoint_ep{epoch+1}.pth"
            safe_save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict(),
                "best_val_loss":   best_val_loss,
                "train_losses":    train_losses,
                "val_losses":      val_losses,
            }, ckpt_path)
            print(f"  Checkpoint -> {ckpt_path}")

    if writer:
        writer.close()

    plt.figure(figsize=(12, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Train", linewidth=1.5)
    plt.plot(epochs_range, val_losses,   label="Val",   linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Sam3Model (PCS) Fine-tuning — {NUM_EPOCHS} Epochs | "
              f"Best val: {best_val_loss:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()

    print(f"\nDone.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {BEST_MODEL_PATH}")
    print(f"  Loss curve    : loss_curve.png")


if __name__ == "__main__":
    main()
