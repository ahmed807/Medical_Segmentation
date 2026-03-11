
# %%

import json
import cv2
import numpy as np
import os
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
# from google.colab import files
from torch.cuda.amp import autocast, GradScaler  # <--- Needed for ROCm/W7900 Mixed Precision
import shutil                                     # <--- Needed for dataset file operations


# %%
# #Unzip your dataset

# if os.path.exists("sam_finetuning_dataset.zip"):
#     with zipfile.ZipFile("sam_finetuning_dataset.zip", 'r') as zip_ref:
#         zip_ref.extractall(".")
#     print("✅ Dataset unzipped successfully!")
# else:
#     print("⚠️ Error: Please upload 'sam_finetuning_dataset.zip' to the Files tab first.")

# %%


class CustomSAMDataset(Dataset):
    def __init__(self, json_file, root_dir, processor):
        self.root_dir = root_dir  # <--- Store the root folder path
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Load Image with CORRECT PATH
        # We join the 'root_dir' with the relative path from the JSON
        image_path = os.path.join(self.root_dir, item["image"])
        image = Image.open(image_path).convert("RGB")

        # 2. Load Ground Truth Mask with CORRECT PATH
        mask_path = os.path.join(self.root_dir, item["annotation"])
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        # 3. Get the Prompt Box
        prompt_box = item["prompt_box"]

        # 4. Format inputs for the SAM Processor
        inputs = self.processor(
            image,
            input_boxes=[[prompt_box]],
            return_tensors="pt"
        )

        # 5. Remove the batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 6. Resize and Normalize Ground Truth Mask
        inputs["ground_truth_mask"] = cv2.resize(mask, (256, 256))
        inputs["ground_truth_mask"] = (inputs["ground_truth_mask"] > 127).astype(np.float32)
        inputs["ground_truth_mask"] = torch.tensor(inputs["ground_truth_mask"])

        return inputs

# %%
# --- CONFIGURATION ---
BATCH_SIZE = 2            # Your W7900 has 48GB VRAM, so batch 4 is very safe and stable
NUM_EPOCHS = 200          
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4           # Fixes the speed bottleneck
PIN_MEMORY = True         # Faster CPU -> GPU transfer
# ---------------------

# %%
# --- INITIALIZATION BLOCK ---
print("🚀 Initializing...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {torch.cuda.get_device_name(0)}")

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)

# Freeze Vision Encoder (Standard finetuning practice)
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad = False

# %%
# 2. Dataset Loading (Requires train.json / val.json from previous step)
# Ensure you have run the 'prepare_dataset.py' with the split logic first!
if not os.path.exists("sam_finetuning_dataset/train.json"):
    raise FileNotFoundError("❌ 'train.json' not found. Please run the split_dataset script first.")

train_dataset = CustomSAMDataset(
    json_file="sam_finetuning_dataset/train.json",
    root_dir="sam_finetuning_dataset",
    processor=processor
)
val_dataset = CustomSAMDataset(
    json_file="sam_finetuning_dataset/val.json",
    root_dir="sam_finetuning_dataset",
    processor=processor
)

# 3. Optimized DataLoaders
print(f"⚡ Creating High-Performance DataLoaders (Workers: {NUM_WORKERS})...")
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)

print(f"✅ Training on {len(train_dataset)} examples, Validating on {len(val_dataset)} examples.")

# %%
# 4. Training Setup
optimizer = AdamW(model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = GradScaler() # For Mixed Precision
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"🔥 Starting Training for {NUM_EPOCHS} Epochs on W7900...")

for epoch in range(NUM_EPOCHS):
    # --- TRAIN LOOP ---
    model.train()
    epoch_train_losses = []
    
    # Progress bar for visual feedback
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS}", leave=False)
    
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_boxes = batch["input_boxes"].to(device, non_blocking=True)
        ground_truth_masks = batch["ground_truth_mask"].to(device, non_blocking=True)

        # Mixed Precision Forward
        with autocast():
            outputs = model(
                pixel_values=pixel_values, 
                input_boxes=input_boxes, 
                multimask_output=False
            )
            # Fix shapes: [B, 1, 256, 256] -> [B, 256, 256]
            predicted_mask = outputs.pred_masks.squeeze(1)[:, 0, :, :]
            loss = F.binary_cross_entropy_with_logits(predicted_mask, ground_truth_masks)

        # Mixed Precision Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_losses.append(loss.item())
        pbar.set_postfix(train_loss=f"{loss.item():.4f}")

    avg_train = np.mean(epoch_train_losses)
    train_losses.append(avg_train)

    # --- VALIDATION LOOP ---
    model.eval()
    epoch_val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_boxes = batch["input_boxes"].to(device, non_blocking=True)
            ground_truth_masks = batch["ground_truth_mask"].to(device, non_blocking=True)

            with autocast():
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                predicted_mask = outputs.pred_masks.squeeze(1)[:, 0, :, :]
                v_loss = F.binary_cross_entropy_with_logits(predicted_mask, ground_truth_masks)
            
            epoch_val_losses.append(v_loss.item())
            
    avg_val = np.mean(epoch_val_losses)
    val_losses.append(avg_val)

    # --- LOGGING & SAVING ---
    print(f"Epoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f}")
    
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "sam_2_best_val_model.pth")
        print(f"   💾 Saved New Best Model! (Val Loss: {best_val_loss:.4f})")

# 5. Save Final Plot (Server Friendly)
print("📊 Saving Loss Curve to 'loss_curve.png'...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('SAM Finetuning: Train vs Val')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png') # <--- Saves to file instead of showing
print("🎉 Done! Check 'sam_2_best_val_model.pth' and 'loss_curve.png'.")

# %%
# Plotting Train vs Validation Loss
plt.figure(figsize=(10, 6))

# Plot Training Loss (Blue)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='b', label='Training Loss')

# Plot Validation Loss (Red)
plt.plot(range(1, num_epochs + 1), val_losses, marker='x', color='r', label='Validation Loss')

plt.title('SAM Finetuning: Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Loss')
plt.grid(True)
plt.legend()
plt.show()

# %%
# 1. Load the Best Saved Model
model.load_state_dict(torch.load("sam_2_best_val_model.pth"))
model.to(device)
model.eval()

# 2. Setup Test Loader
test_dataset = CustomSAMDataset(
    json_file="sam_finetuning_dataset/test.json",
    root_dir="sam_finetuning_dataset",
    processor=processor
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"🧪 Testing on {len(test_dataset)} unseen images...")

test_losses = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].to(device)

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)

        # Calculate Loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_mask = predicted_masks[:, 0, :, :]
        loss = F.binary_cross_entropy_with_logits(predicted_mask, ground_truth_masks)
        test_losses.append(loss.item())

        # Optional: Visualize every 5th test image
        if i % 5 == 0:
            pred_binary = (predicted_mask > 0).cpu().numpy()[0]
            gt_binary = ground_truth_masks.cpu().numpy()[0]

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(gt_binary, cmap='gray')
            ax[0].set_title("Ground Truth")
            ax[1].imshow(pred_binary, cmap='gray')
            ax[1].set_title(f"Prediction (Loss: {loss.item():.4f})")
            plt.show()

print(f"📊 Final Test Set Average Loss: {np.mean(test_losses):.4f}")

# %%


# # # --- NEW: Filter for Arrows ---
# # # Find all indices where the prompt text contains "arrow"
# # arrow_indices = [i for i, item in enumerate(dataset.data) if "arrow" in item.get("prompt_text", "").lower()]

# # if not arrow_indices:
# #     print("⚠️ No arrow prompts found in dataset metadata.")
# #     print("Falling back to random selection (update prepare_dataset.py to fix this).")
# #     idx = random.randint(0, len(dataset) - 1)
# # else:
# #     idx = random.choice(arrow_indices)
# #     print(f"✅ Selected Index {idx} (Prompt: '{dataset.data[idx]['prompt_text']}')")
# # ------------------------------
# idx = random.randint(0, len(dataset) - 1)
# item = dataset[idx]

# # Prepare Inputs
# image_path = os.path.join("sam_finetuning_dataset", dataset.data[idx]["image"])
# image = Image.open(image_path).convert("RGB")
# prompt_box = dataset.data[idx]["prompt_box"]
# ground_truth_mask = item["ground_truth_mask"].numpy()

# # Run Inference
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.eval()
# inputs = processor(image, input_boxes=[[prompt_box]], return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model(**inputs, multimask_output=False)

# # Process Result
# raw_mask = outputs.pred_masks.cpu().numpy()
# prediction = np.squeeze(raw_mask)
# if prediction.ndim == 3: prediction = prediction[0] # Safety check
# prediction = (prediction > 0.0).astype(np.uint8)

# # Visualization
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# def show_mask(mask, ax, color=None):
#     if color is None: color = np.array([30/255, 144/255, 255/255, 0.6])
#     if mask.ndim > 2: mask = mask.squeeze()
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Plot A: Input
# axes[0].imshow(image)
# show_box(prompt_box, axes[0])
# axes[0].set_title(f"Prompt: {dataset.data[idx].get('prompt_text', 'Box')}")
# axes[0].axis('off')

# # Plot B: Ground Truth
# axes[1].imshow(image)
# gt_resized = cv2.resize(ground_truth_mask.astype(np.float32), (image.width, image.height))
# gt_resized = (gt_resized > 0.5).astype(np.float32)
# show_mask(gt_resized, axes[1])
# axes[1].set_title("Ground Truth")
# axes[1].axis('off')

# # Plot C: Prediction
# axes[2].imshow(image)
# prediction_resized = cv2.resize(prediction.astype(np.float32), (image.width, image.height))
# prediction_resized = (prediction_resized > 0.5).astype(np.float32)
# show_mask(prediction_resized, axes[2])
# axes[2].set_title("Fine-Tuned Prediction")
# axes[2].axis('off')

# plt.show()

# # %%
# # Save the current state
# torch.save(model.state_dict(), "sam_finetuned_epoch.pth")
# print("✅ Saved checkpoint for Epoch 10")

# # %%


# # # Check if the file exists before trying to download
# # if os.path.exists("sam_best_model.pth"):
# #     print("⬇️ Downloading best model...")
# #     files.download("sam_best_model.pth")
# # else:
# #     print("⚠️ Model file not found. Make sure training finished and saved 'sam_best_model.pth'.")

# # %%
# model = SamModel.from_pretrained("facebook/sam-vit-base")
# # Load your custom weights
# model.load_state_dict(torch.load("sam_best_model.pth", map_location="cuda"))

# # %%


# # --- CONFIGURATION ---
# TEST_IMAGE_PATH = "my_new_test_image.jpg"  # Upload a NEW image to test!
# MODEL_PATH = "sam_best_model.pth"
# # ---------------------

# # 1. Load Model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SamModel.from_pretrained("facebook/sam-vit-base")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# # 2. Auto-Detect Prompt (Magic Inference)
# def magic_inference(image_path):
#     image_cv = cv2.imread(image_path)
#     if image_cv is None: return
#     image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

#     # Detect colorful markings (red, green, blue, yellow, etc.)
#     # We combine masks for common annotation colors
#     hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

#     # Tuning for "Not Grayscale" (Saturation > 40)
#     # This ignores the photo and finds the colorful paint
#     mask_color = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

#     contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("⚠️ No annotation detected! Draw a colorful arrow/box first.")
#         return

#     # Assume largest color blob is the prompt
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)
#     prompt_box = [x, y, x+w, y+h]

#     print(f"🎯 Detected Prompt Box: {prompt_box}")

#     # Run SAM
#     inputs = processor(Image.fromarray(image_rgb), input_boxes=[[prompt_box]], return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs, multimask_output=False)

#     # Visualise
#     prediction = outputs.pred_masks[0, 0].cpu().numpy() > 0.5

#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_rgb)
#     plt.imshow(prediction, alpha=0.5)
#     # Draw the box we found
#     plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='yellow', facecolor='none', lw=2))
#     plt.title("Did it work on a new image?")
#     plt.axis('off')
#     plt.show()

# # Run it
# magic_inference(TEST_IMAGE_PATH)


