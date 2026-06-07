"""
train.py
--------
Fine-tunes SAM3 Tracker (PVS) for annotation-to-object segmentation.

Model: Sam3TrackerModel — the SAM2-style single-instance segmenter.
       This is the correct SAM3 component for "point at one object, segment it."
       Sam3Model (the DETR detector for PCS) finds ALL instances of a concept —
       that's not what we need.

Prompt routing by annotation type (matches inference exactly):
  arrow         -> arrowhead tip pixel        -> positive point prompt
  number_letter -> pixels ON the object       -> multiple positive point prompts
  rect_bbox     -> annotation bounding box    -> box prompt
  freeform_bbox -> annotation bounding box    -> box prompt

Key design:
  - Training prompts are derived from the annotation mask the same way
    inference will derive them — zero train/test mismatch.
  - NO text prompts — Sam3Tracker uses purely spatial prompts.
  - GroupedBatchSampler ensures all items in a batch share the same
    annotation type, so collate always produces consistent tensor shapes.

Frozen  : vision_encoder (~600M params — heavy pre-trained backbone)
Trained : sam_prompt_encoder, sam_mask_decoder

Loss    : 20×Focal + Dice + IoU-MSE  (SAM paper, autocast-safe)
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
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim      import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers     import Sam3TrackerProcessor, Sam3TrackerModel
from tqdm             import tqdm
from collections      import defaultdict
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

BATCH_SIZE    = 48
GRAD_ACCUM    = 1
NUM_EPOCHS    = 30
LR            = 5e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 12
PIN_MEMORY    = True
PREFETCH      = 4

# Number of points sampled from number/letter annotation masks
N_LETTER_POINTS = 5

CHECKPOINT_EVERY = 5
BEST_MODEL_DIR   = "sam3_tracker_iou_best"          # save_pretrained directory
BEST_CKPT_PATH   = "sam3_best_iou.pth"      # state_dict fallback (backward compat)
LOG_DIR          = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/sam3_iou_finetune"

# Freeze the heavy vision encoder only.
# Train: sam_prompt_encoder (learns annotation->object spatial mapping)
#        sam_mask_decoder   (learns to produce accurate object masks)
FROZEN_PREFIXES = [
    "vision_encoder",
]
# ───────────────────────────────────────────────────────────────────────────────


# ── PROMPT HELPERS ─────────────────────────────────────────────────────────────

def find_arrow_tip(annot_np: np.ndarray):
    """
    Finds the arrowhead tip — the contour point furthest from the centroid.
    The tip is the closest point to the object being annotated.
    Returns [x, y] or None if detection fails.
    """
    contours, _ = cv2.findContours(
        annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts   = np.vstack(contours).reshape(-1, 2).astype(float)
    centroid  = all_pts.mean(axis=0)
    distances = np.linalg.norm(all_pts - centroid, axis=1)
    tip       = all_pts[distances.argmax()]
    return [int(tip[0]), int(tip[1])]


def get_points_from_mask(annot_np: np.ndarray,
                          n_points: int = 5):
    """
    Samples N evenly-spaced points from annotation mask pixels.
    Used for number/letter annotations that sit ON TOP of the object.
    Returns list of [x, y] or None if mask is empty.
    """
    ys, xs = np.where(annot_np > 127)
    if len(xs) == 0:
        return None
    indices = np.linspace(0, len(xs) - 1,
                          min(n_points, len(xs)), dtype=int)
    return [[int(xs[i]), int(ys[i])] for i in indices]


def get_prompt_box(annot_np: np.ndarray,
                   padding: int = 10,
                   fallback: list = None) -> list:
    """Union bbox of all annotation contours + padding."""
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


# ── GROUPED BATCH SAMPLER ──────────────────────────────────────────────────────

class GroupedBatchSampler(Sampler):
    """
    Groups dataset items by annotation_type so every batch is homogeneous.
    This ensures consistent tensor shapes in the collate function:
      - arrow batches          -> input_points shape [1, 1, 2]
      - number_letter batches  -> input_points shape [1, N, 2]
      - rect/freeform_bbox     -> input_boxes  shape [1, 4]
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle    = shuffle

        groups = defaultdict(list)
        for i, item in enumerate(dataset.data):
            ann_type = item.get("annotation_type", "arrow")
            groups[ann_type].append(i)

        self.batches = []
        for ann_type, indices in groups.items():
            if shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                if batch:
                    self.batches.append(batch)

        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# ── DATASET ────────────────────────────────────────────────────────────────────

class AnnotationToSegDataset(Dataset):
    """
    SAM3 Tracker fine-tuning dataset.

    Prompt strategy per annotation type:
      arrow         -> find arrowhead tip -> 1 positive point prompt
      number_letter -> sample pixels ON object -> N positive point prompts
      rect_bbox     -> annotation box surrounds object -> box prompt
      freeform_bbox -> annotation box surrounds object -> box prompt

    No text prompts — Sam3Tracker uses purely spatial prompts,
    matching inference exactly.

    GT mask: segmentation of the TARGET object (annotation_segmap)
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

        # Optional horizontal flip (train split only)
        if self.augment and torch.rand(1).item() > 0.5:
            image    = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_mask  = np.fliplr(gt_mask).copy()
            annot_np = np.fliplr(annot_np).copy()

        # ── Route prompt by annotation type ───────────────────────────────────
        inputs = self._build_inputs(
            image, annot_np, annotation_type, item["prompt_box"]
        )

        # Squeeze batch dim; keep only tensor outputs from processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # Resize GT to 256x256 — matches Sam3Tracker decoder output resolution.
        # Avoids the cost of upscaling predictions to 1024x1024 during training.
        resized_gt = cv2.resize(gt_mask, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
        inputs["ground_truth_mask"] = torch.tensor(resized_gt > 0).float()
        inputs["original_size"]     = torch.tensor([orig_h, orig_w])
        inputs["annotation_type"]   = annotation_type
        inputs["image_path"]        = image_path
        inputs["annot_mask_path"]   = annot_mask_path

        return inputs

    def _build_inputs(self, image, annot_np, annotation_type, fallback_box):
        """
        Build Sam3TrackerProcessor inputs based on annotation type.
        No text — purely spatial prompts matching inference.
        """

        if annotation_type == "arrow":
            # Arrow tip = positive point click near/on the target object
            tip = find_arrow_tip(annot_np)
            if tip:
                return self.processor(
                    images         = image,
                    input_points   = [[[tip]]],
                    input_labels   = [[[1]]],
                    return_tensors = "pt"
                )
            # Fallback to box if tip detection fails
            box = get_prompt_box(annot_np, padding=10, fallback=fallback_box)
            return self.processor(
                images         = image,
                input_boxes    = [[box]],
                return_tensors = "pt"
            )

        elif annotation_type == "number_letter":
            # Letters/numbers sit directly on the object — multi positive points
            points = get_points_from_mask(annot_np, n_points=N_LETTER_POINTS)
            if points:
                try:
                    return self.processor(
                        images         = image,
                        input_points   = [[points]],
                        input_labels   = [[[1] * len(points)]],
                        return_tensors = "pt"
                    )
                except Exception:
                    pass
            # Fallback: expand box to give SAM context
            box = get_prompt_box(annot_np, padding=150, fallback=fallback_box)
            return self.processor(
                images         = image,
                input_boxes    = [[box]],
                return_tensors = "pt"
            )

        else:
            # rect_bbox / freeform_bbox — box surrounds the object directly
            box = get_prompt_box(annot_np, padding=5, fallback=fallback_box)
            return self.processor(
                images         = image,
                input_boxes    = [[box]],
                return_tensors = "pt"
            )


def sam_collate_fn(batch):
    """
    Collate for homogeneous batches (same annotation_type per batch,
    guaranteed by GroupedBatchSampler).
    Handles both box-prompted and point-prompted items.
    """
    result = {
        "pixel_values":      torch.stack([b["pixel_values"]      for b in batch]),
        "ground_truth_mask": torch.stack([b["ground_truth_mask"]  for b in batch]),
        "original_size":     torch.stack([b["original_size"]      for b in batch]),
        "annotation_type":   [b["annotation_type"]   for b in batch],
        "image_path":        [b["image_path"]        for b in batch],
        "annot_mask_path":   [b["annot_mask_path"]   for b in batch],
    }

    # Box prompts (rect_bbox, freeform_bbox, arrow/number_letter fallbacks)
    if all("input_boxes" in b for b in batch):
        result["input_boxes"] = torch.stack([b["input_boxes"] for b in batch])

    # Point prompts (arrow tip, number_letter pixels)
    if all("input_points" in b for b in batch):
        result["input_points"] = torch.stack([b["input_points"] for b in batch])
        result["input_labels"] = torch.stack([b["input_labels"] for b in batch])

    return result


# ── LOSS ───────────────────────────────────────────────────────────────────────

def focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss (Lin et al., 2017).
    Down-weights easy pixels so the model focuses on hard boundary regions.
    All ops in float32 for autocast safety.
    """
    inputs  = inputs.float()
    targets = targets.float()
    prob    = torch.sigmoid(inputs)
    ce      = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean()


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss. Autocast-safe (explicit float cast)."""
    probs = torch.sigmoid(inputs.float()).view(-1)
    tgts  = targets.float().view(-1)
    inter = (probs * tgts).sum()
    return 1 - (2. * inter + smooth) / (probs.sum() + tgts.sum() + smooth)


def compute_iou_targets(pred_logits: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
    """
    Computes per-sample IoU between thresholded predictions and GT.
    Returns shape (B, 1) to match model's iou_scores output.
    """
    pred_bin = (torch.sigmoid(pred_logits.float()) > 0.5).float()
    tgts     = targets.float()
    inter    = (pred_bin * tgts).sum(dim=(-2, -1))
    union    = pred_bin.sum(dim=(-2, -1)) + tgts.sum(dim=(-2, -1)) - inter
    iou      = inter / (union + 1e-6)
    return iou.unsqueeze(1)  # (B, 1)


def combined_loss(pred_masks: torch.Tensor, targets: torch.Tensor,
                  iou_pred: torch.Tensor) -> torch.Tensor:
    """
    SAM-paper-style combined loss:
        20 × focal_loss + dice_loss + IoU-MSE

    - Focal: focuses on hard boundary pixels (α=0.25, γ=2)
    - Dice:  region overlap, handles class imbalance
    - IoU-MSE: trains the model's confidence head (iou_scores)
    """
    l_focal = focal_loss(pred_masks, targets)
    l_dice  = dice_loss(pred_masks, targets)

    iou_gt  = compute_iou_targets(pred_masks.detach(), targets)
    l_iou   = F.mse_loss(iou_pred.float(), iou_gt)

    return 20.0 * l_focal + l_dice + l_iou


# ── FORWARD PASS ───────────────────────────────────────────────────────────────

def forward_pass(model, batch, device):
    """
    Shared forward + loss for Sam3TrackerModel.
    Passes whichever prompt type is present in the batch.

    GT masks are 256×256 (matching decoder output).
    pred_masks shape: (B, point_batch, num_masks, H, W)
    iou_scores shape: (B, point_batch, num_masks)
    """
    pv = batch["pixel_values"].to(device, non_blocking=True)
    gt = batch["ground_truth_mask"].to(device, non_blocking=True)

    kwargs = dict(
        pixel_values     = pv,
        multimask_output = False,
    )

    if "input_boxes" in batch:
        kwargs["input_boxes"] = batch["input_boxes"].to(device, non_blocking=True)

    if "input_points" in batch:
        kwargs["input_points"] = batch["input_points"].to(device, non_blocking=True)
        kwargs["input_labels"] = batch["input_labels"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda"):
        outputs = model(**kwargs)

        # pred_masks: (B, point_batch, num_masks, H, W)
        # With single object + multimask_output=False: (B, 1, 1, H, W)
        low_res = outputs.pred_masks.squeeze(1).squeeze(1)  # -> (B, H, W)

        # iou_scores: (B, point_batch, num_masks) -> (B, 1)
        iou_pred = outputs.iou_scores.squeeze(1)  # -> (B, 1)

        # Resize predictions to match GT at 256×256
        # (decoder output is typically 256×256 already, but ensure match)
        if low_res.shape[-2:] != gt.shape[-2:]:
            low_res = F.interpolate(
                low_res.unsqueeze(1),
                size=(gt.shape[1], gt.shape[2]),
                mode="bilinear", align_corners=False
            ).squeeze(1)

        loss = combined_loss(low_res, gt, iou_pred)

    return loss, low_res, gt


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading Sam3TrackerModel from {MODEL_ID}...")
    processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    model     = Sam3TrackerModel.from_pretrained(MODEL_ID)

    # Print module structure for verification
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

    train_sampler = GroupedBatchSampler(train_dataset, BATCH_SIZE, shuffle=True)
    val_sampler   = GroupedBatchSampler(val_dataset,   BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=PREFETCH,
        collate_fn=sam_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=PREFETCH,
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
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch+1:03d}/{NUM_EPOCHS} [train]",
                    leave=False, dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            loss, _, _ = forward_pass(model, batch, DEVICE)
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
            pbar.set_postfix(loss=f"{loss_display:.4f}")

        scheduler.step()
        avg_train = float(np.mean(epoch_train))
        train_losses.append(avg_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        epoch_val  = []
        epoch_ious = []
        with torch.no_grad():
            for batch in tqdm(val_loader,
                               desc=f"Ep {epoch+1:03d}/{NUM_EPOCHS} [val]  ",
                               leave=False, dynamic_ncols=True):
                loss, pred, gt_v = forward_pass(model, batch, DEVICE)
                epoch_val.append(loss.item())
                # Track IoU for monitoring segmentation quality
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                inter = (pred_bin * gt_v).sum(dim=(-2, -1))
                union = pred_bin.sum(dim=(-2, -1)) + gt_v.sum(dim=(-2, -1)) - inter
                batch_iou = (inter / (union + 1e-6)).mean().item()
                epoch_ious.append(batch_iou)

        avg_val = float(np.mean(epoch_val))
        avg_iou = float(np.mean(epoch_ious))
        val_losses.append(avg_val)
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"Ep {epoch+1:03d}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  IoU={avg_iou:.4f}  lr={lr_now:.2e}")

        if writer:
            writer.add_scalars("loss", {"train": avg_train, "val": avg_val}, epoch+1)
            writer.add_scalar("val_iou", avg_iou, epoch+1)
            writer.add_scalar("lr", lr_now, epoch+1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # save_pretrained: portable HF format (load with from_pretrained)
            model.save_pretrained(BEST_MODEL_DIR)
            processor.save_pretrained(BEST_MODEL_DIR)
            # Also save .pth for backward compatibility with existing inference.py
            safe_save(model.state_dict(), BEST_CKPT_PATH)
            print(f"  Saved best -> {BEST_MODEL_DIR}/  (val={best_val_loss:.4f})")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = f"sam3_iou_checkpoint_ep{epoch+1}.pth"
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
    plt.plot(epochs_range, train_losses, label="Train (Focal+Dice+IoU)", linewidth=1.5)
    plt.plot(epochs_range, val_losses,   label="Val (Focal+Dice+IoU)",   linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SAM3 Tracker Fine-tuning (Focal+Dice+IoU) — {NUM_EPOCHS} Epochs | Best val: {best_val_loss:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()

    print(f"\nDone.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {BEST_MODEL_DIR}/  (HF format)")
    print(f"  Best weights  : {BEST_CKPT_PATH}  (.pth)")
    print(f"  Loss curve    : loss_curve.png")


if __name__ == "__main__":
    main()
