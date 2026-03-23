import os
import json
import shutil
import cv2
import numpy as np
import random

# --- CONFIGURATION ---
RAW_DATA_DIR = "/home/ahma/unannotate/output" # Your path
OUTPUT_DIR = "sam_finetuning_dataset"
SPLIT_RATIOS = (0.8, 0.1, 0.1) # 80% Train, 10% Val, 10% Test
RANDOM_SEED = 42 # Ensures the split is the same every time you run it
# ---------------------

def fix_extension(folder_path, filename):
    base_name = os.path.splitext(filename)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_path = os.path.join(folder_path, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None

def prepare_dataset():
    # 1. Setup Output Structure
    images_dir = os.path.join(OUTPUT_DIR, "images")
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    prompt_masks_dir = os.path.join(OUTPUT_DIR, "prompt_masks")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(prompt_masks_dir, exist_ok=True)
    
    all_entries = []
    print(f"Processing data from '{RAW_DATA_DIR}'...")

    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: '{RAW_DATA_DIR}' not found.")
        return

    folder_names = sorted([f for f in os.listdir(RAW_DATA_DIR) 
                           if os.path.isdir(os.path.join(RAW_DATA_DIR, f))])
    
    # 2. Collect All Data
    for folder_name in folder_names:
        folder_path = os.path.join(RAW_DATA_DIR, folder_name)
        labels_path = os.path.join(folder_path, "labels.json")
        
        if not os.path.exists(labels_path): continue

        with open(labels_path, 'r') as f:
            labels_data = json.load(f)

        for key, value in labels_data.items():
            target_img_name = value.get("annotated_img_name")
            if not target_img_name: continue

            image_path = fix_extension(folder_path, target_img_name)
            if image_path is None: continue

            prompt_mask_name = f"annotation_mask_{key}.png"
            gt_mask_name = f"annotation_segmap_{key}.png"
            prompt_path = os.path.join(folder_path, prompt_mask_name)
            gt_path = os.path.join(folder_path, gt_mask_name)

            if not os.path.exists(prompt_path) or not os.path.exists(gt_path): continue

            # --- Save Files ---
            unique_base_name = f"{folder_name}_{key}"
            new_image_filename = f"{unique_base_name}.jpg"
            img = cv2.imread(image_path)
            if img is None: continue
            cv2.imwrite(os.path.join(images_dir, new_image_filename), img)

            p_mask = cv2.imread(prompt_path, cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            prompt_box = [x, y, x + w, y + h]

            new_mask_filename = f"{unique_base_name}_mask.png"
            shutil.copy(gt_path, os.path.join(masks_dir, new_mask_filename))
            
            new_prompt_mask_filename = f"{unique_base_name}_prompt_mask.png"
            shutil.copy(prompt_path, os.path.join(prompt_masks_dir, new_prompt_mask_filename))

            all_entries.append({
                "image": f"images/{new_image_filename}",
                "annotation": f"masks/{new_mask_filename}",
                "annotation_mask": f"prompt_masks/{new_prompt_mask_filename}",  # <-- added
                "prompt_box": prompt_box,
                "label": value.get("object", "unknown"),
                "prompt_text": value.get("annotation", "unknown"),
            })

            # # Add to list
            # all_entries.append({
            #     "image": f"images/{new_image_filename}",
            #     "annotation": f"masks/{new_mask_filename}",
            #     "prompt_box": prompt_box,
            #     "label": value.get("object", "unknown"),
            #     "prompt_text": value.get("annotation", "unknown"),
            #     "annotation_mask": prompt_path,
            # })

    # 3. Shuffle and Split
    random.seed(RANDOM_SEED)
    random.shuffle(all_entries)
    
    total = len(all_entries)
    n_train = int(total * SPLIT_RATIOS[0])
    n_val = int(total * SPLIT_RATIOS[1])
    
    train_data = all_entries[:n_train]
    val_data = all_entries[n_train:n_train+n_val]
    test_data = all_entries[n_train+n_val:]
    
    # 4. Save 3 JSON Files
    with open(os.path.join(OUTPUT_DIR, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "val.json"), 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "test.json"), 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"\n🎉 Done! Split Results:")
    print(f"   Train: {len(train_data)}")
    print(f"   Val:   {len(val_data)}")
    print(f"   Test:  {len(test_data)}")

if __name__ == "__main__":
    prepare_dataset()