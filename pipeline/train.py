"""
train.py
--------
Fine-tunes SAM3 for the annotation-to-segmentation task.

Task:
  Input : Annotated image (arrow / bbox / letter / dashed-line annotation)
          + bounding box of that annotation as a SAM3 prompt
  Output: Segmentation mask of the TARGET object the annotation points at

Freezing strategy — Option B (balanced speed + quality):
  Frozen  : vision_encoder, text_encoder, geometry_encoder,
            detr_encoder, detr_decoder, text_projection
  Trained : mask_decoder, dot_product_scoring

  Rationale:
    - mask_decoder    : generates the pixel-level segmentation mask — must train
    - dot_product_scoring : small module that aligns text prompt ("person")
                            with detected object regions — helps when multiple
                            objects are present, low backward-pass cost

Loss    : BCE-with-logits + Dice  (autocast-safe)
Hardware: NVIDIA A40 46GB

Usage:
    python train.py
    python train.py --resume sam3_checkpoint_ep25.pth
"""

import os
import json
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
from huggingface_hub import login

# ── HUGGING FACE LOGIN ─────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    raise EnvironmentError(
        "HF_TOKEN environment variable not set.\n"
        "Run: export HF_TOKEN='hf_your_token_here'  before running train.py"
    )
login(token=HF_TOKEN)
print("Logged in to Hugging Face.")


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False
    print("TensorBoard not available.  pip install tensorboard")


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DATASET_DIR   = "sam_finetuning_dataset"
MODEL_ID      = "facebook/sam3"
DEVICE        = "cuda:0"

BATCH_SIZE    = 4
GRAD_ACCUM    = 4       # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
NUM_EPOCHS    = 100
LR            = 1e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 8
PIN_MEMORY    = True
PREFETCH      = 4

CHECKPOINT_EVERY = 25
BEST_MODEL_PATH  = "sam3_best.pth"
LOG_DIR          = "runs/sam3_finetune"

# ── Option B: freeze heavy encoders, train mask_decoder + dot_product_scoring ──
FROZEN_PREFIXES = [
    "vision_encoder",    # 807M params — heaviest, already well pre-trained
    "text_encoder",      # CLIP-style, already well pre-trained
    "geometry_encoder",  # spatial layout encoder
    "detr_encoder",      # object detection encoder
    "detr_decoder",      # object detection decoder
    "text_projection",   # small projection layer tied to frozen text encoder
]
# Everything NOT in FROZEN_PREFIXES is trained:
#   mask_decoder       — generates pixel-level segmentation mask
#   dot_product_scoring — aligns text prompt with detected object regions
# ───────────────────────────────────────────────────────────────────────────────


# ── DATASET ────────────────────────────────────────────────────────────────────

class AnnotationToSegDataset(Dataset):
    """
    SAM3 fine-tuning dataset for annotation -> object segmentation.

    Prompt  : bounding box derived at runtime from annotation_mask
              (the arrow / symbol / dashed line drawn on the image)
    GT mask : segmentation map of the TARGET object (annotation_segmap)
    Text    : object label e.g. "person", "bear"
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

        # Derive SAM3 prompt bbox from annotation mask at runtime
        annot_np    = np.array(Image.open(annot_mask_path).convert("L"))
        prompt_box  = self._get_prompt_box(annot_np, fallback=item["prompt_box"])
        prompt_text = item.get("prompt_text", "")

        # Optional horizontal flip augmentation (train split only)
        if self.augment and torch.rand(1).item() > 0.5:
            image   = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_mask = np.fliplr(gt_mask).copy()
            x1, y1, x2, y2 = prompt_box
            prompt_box = [orig_w - x2, y1, orig_w - x1, y2]

        inputs = self.processor(
            image,
            input_boxes=[[prompt_box]],
            text=prompt_text if prompt_text else None,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Resize GT to 1024x1024 to match SAM3's output resolution
        resized_gt = cv2.resize(gt_mask, (1024, 1024),
                                interpolation=cv2.INTER_NEAREST)
        inputs["ground_truth_mask"] = torch.tensor(resized_gt > 0).float()
        inputs["original_size"]     = torch.tensor([orig_h, orig_w])
        inputs["image_path"]        = image_path
        inputs["annot_mask_path"]   = annot_mask_path

        return inputs

    @staticmethod
    def _get_prompt_box(annot_np: np.ndarray, fallback: list,
                        padding: int = 10) -> list:
        """Union bbox of all annotation contours + padding."""
        contours, _ = cv2.findContours(
            annot_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback
        all_pts      = np.vstack(contours)
        x, y, w, h   = cv2.boundingRect(all_pts)
        img_h, img_w = annot_np.shape
        return [
            max(0,     x - padding),
            max(0,     y - padding),
            min(img_w, x + w + padding),
            min(img_h, y + h + padding),
        ]


def sam_collate_fn(batch):
    """Custom collate that handles string metadata fields alongside tensors."""
    return {
        "pixel_values":      torch.stack([b["pixel_values"]      for b in batch]),
        "input_boxes":       torch.stack([b["input_boxes"]        for b in batch]),
        "input_ids":         torch.stack([b["input_ids"]          for b in batch]),
        "ground_truth_mask": torch.stack([b["ground_truth_mask"]  for b in batch]),
        "original_size":     torch.stack([b["original_size"]      for b in batch]),
        "image_path":        [b["image_path"]       for b in batch],
        "annot_mask_path":   [b["annot_mask_path"]  for b in batch],
    }


# ── LOSS FUNCTIONS ─────────────────────────────────────────────────────────────

def dice_loss(inputs, targets, smooth: float = 1.0):
    """Dice loss on logits via sigmoid. Autocast-safe."""
    probs = torch.sigmoid(inputs).view(-1)
    tgts  = targets.view(-1)
    inter = (probs * tgts).sum()
    return 1 - (2. * inter + smooth) / (probs.sum() + tgts.sum() + smooth)


def combined_loss(pred, target):
    """
    BCE-with-logits + Dice.
    Uses binary_cross_entropy_with_logits (NOT binary_cross_entropy)
    to stay autocast-safe under fp16/bf16.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dc  = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dc


# ── FORWARD PASS ───────────────────────────────────────────────────────────────

def forward_pass(model, batch, device):
    """Shared forward + loss computation for train and val loops."""
    pv  = batch["pixel_values"].to(device, non_blocking=True)
    ib  = batch["input_boxes"].to(device,  non_blocking=True)
    ids = batch["input_ids"].to(device,    non_blocking=True)
    gt  = batch["ground_truth_mask"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda"):
        outputs  = model(
            pixel_values     = pv,
            input_boxes      = ib,
            input_ids        = ids,
            multimask_output = False,
        )
        low_res  = outputs.pred_masks[:, 0, :, :]
        upscaled = F.interpolate(
            low_res.unsqueeze(1),
            size=(gt.shape[1], gt.shape[2]),
            mode="bilinear", align_corners=False
        ).squeeze(1)
        loss = combined_loss(upscaled, gt)

    return loss, upscaled, gt


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume training")
    args = parser.parse_args()

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model     = Sam3Model.from_pretrained(MODEL_ID)

    # Apply Option B freezing
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in FROZEN_PREFIXES):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    frozen    = total_p - trainable

    print(f"  Frozen    : {frozen:,}  ({100*frozen/total_p:.1f}%)")
    print(f"  Trainable : {trainable:,}  ({100*trainable/total_p:.1f}%)")

    trainable_components = sorted(set(
        name.split(".")[0]
        for name, p in model.named_parameters() if p.requires_grad
    ))
    print(f"  Training  : {trainable_components}")

    model.to(DEVICE)
    print(f"  Device    : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    # ── Data ───────────────────────────────────────────────────────────────────
    for split in ["train.json", "val.json"]:
        if not os.path.exists(os.path.join(DATASET_DIR, split)):
            raise FileNotFoundError(
                f"'{split}' not found. Run prepare_dataset.py first.")

    print("\nLoading datasets...")
    train_dataset = AnnotationToSegDataset(
        os.path.join(DATASET_DIR, "train.json"),
        DATASET_DIR, processor, augment=True
    )
    val_dataset = AnnotationToSegDataset(
        os.path.join(DATASET_DIR, "val.json"),
        DATASET_DIR, processor, augment=False
    )

    print(f"  Train : {len(train_dataset):,}")
    print(f"  Val   : {len(val_dataset):,}")
    print(f"  Effective batch : {BATCH_SIZE} x {GRAD_ACCUM} = "
          f"{BATCH_SIZE * GRAD_ACCUM}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=PREFETCH,
        collate_fn=sam_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=PREFETCH,
        collate_fn=sam_collate_fn
    )

    # ── Optimizer — only trainable params ──────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        eta_min=LR * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )
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
            print(f"  Resumed at epoch {start_epoch}, "
                  f"best_val={best_val_loss:.4f}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded weights only (no optimizer state).")

    # ── TensorBoard ────────────────────────────────────────────────────────────
    writer = SummaryWriter(LOG_DIR) if TENSORBOARD else None

    # ── Training loop ──────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    print(f"\nStarting: epochs {start_epoch+1}-{NUM_EPOCHS}  "
          f"({steps_per_epoch:,} steps/epoch)\n")

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
                    max_norm=1.0
                )
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
        epoch_val = []

        with torch.no_grad():
            for batch in tqdm(val_loader,
                               desc=f"Ep {epoch+1:03d}/{NUM_EPOCHS} [val]  ",
                               leave=False, dynamic_ncols=True):
                loss, _, _ = forward_pass(model, batch, DEVICE)
                epoch_val.append(loss.item())

        avg_val = float(np.mean(epoch_val))
        val_losses.append(avg_val)
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"Ep {epoch+1:03d}  "
              f"train={avg_train:.4f}  "
              f"val={avg_val:.4f}  "
              f"lr={lr_now:.2e}")

        if writer:
            writer.add_scalars("loss", {"train": avg_train, "val": avg_val}, epoch+1)
            writer.add_scalar("lr", lr_now, epoch+1)

        # ── Save best ──────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Saved best -> {BEST_MODEL_PATH}  "
                  f"(val={best_val_loss:.4f})")

        # ── Periodic checkpoint with full state for resume ─────────────────────
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = f"sam3_checkpoint_ep{epoch+1}.pth"
            torch.save({
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

    # ── Loss curve ─────────────────────────────────────────────────────────────
    if writer:
        writer.close()

    plt.figure(figsize=(12, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Train (BCE+Dice)", linewidth=1.5)
    plt.plot(epochs_range, val_losses,   label="Val (BCE+Dice)",   linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SAM3 Fine-tuning — Option B — {NUM_EPOCHS} Epochs  |  "
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
    if TENSORBOARD:
        print(f"  TensorBoard   : tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
