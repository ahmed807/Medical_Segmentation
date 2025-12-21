# My_SAM_Project/convert.py

import os
import json
import shutil
import cv2

# Import from our other project files
import config
import coco_helpers

def run_conversion():
    """
    Main function to run the dataset conversion.
    """
    
    # --- 1. Setup Output Directories ---
    output_image_dir = os.path.join(config.OUTPUT_DIR, config.IMAGE_SUBDIR)
    output_json_path = os.path.join(config.OUTPUT_DIR, "_annotations.coco.json")

    os.makedirs(output_image_dir, exist_ok=True)
    
    print(f"Creating COCO dataset at: {config.OUTPUT_DIR}")

    # --- 2. Initialize COCO JSON Structure ---
    coco_output = coco_helpers.init_coco_structure()

    current_image_id = 1
    current_annotation_id = 1

    # --- 3. Main Loop: Iterate Through Data Folders ---
    folder_names = [f for f in os.listdir(config.RAW_DATA_DIR) 
                    if os.path.isdir(os.path.join(config.RAW_DATA_DIR, f))]
    
    for folder_name in folder_names:
        folder_path = os.path.join(config.RAW_DATA_DIR, folder_name)
        
        # --- 
        # --- ⚠️ THIS IS THE FIX ⚠️ ---
        # --- Look for original.png instead of original.jpg
        # ---
        original_image_path = os.path.join(folder_path, "original.png") 
        labels_path = os.path.join(folder_path, "labels.json")

        if not os.path.exists(original_image_path) or not os.path.exists(labels_path):
            # Updated error message to be correct
            print(f"Skipping folder (missing original.png or labels.json): {folder_name}")
            continue

        print(f"Processing: {folder_name}")

        # --- 4. Process Image ---
        image_cv = cv2.imread(original_image_path)
        if image_cv is None:
            print(f"  Error reading image: {original_image_path}")
            continue
        height, width, _ = image_cv.shape

        # Define the new image name and path
        new_image_name = f"{folder_name}.jpg" # Create a unique name
        new_image_path = os.path.join(output_image_dir, new_image_name)

        # --- 
        # --- ✨ IMPROVEMENT ✨ ---
        # --- Use cv2.imwrite to save and convert the image to JPG
        # --- This is more robust than shutil.copy
        # ---
        try:
            cv2.imwrite(new_image_path, image_cv)
        except Exception as e:
            print(f"  Error saving image {new_image_path}: {e}")
            continue

        # Create the COCO image entry
        image_entry = {
            "id": current_image_id,
            "file_name": new_image_name,
            "height": height,
            "width": width
        }
        coco_output["images"].append(image_entry)

        # --- 5. Process Annotations ---
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)

        for key, value in labels_data.items():
            object_name = value["object"]
            
            # Get the category ID (or create a new one)
            category_id = coco_helpers.get_or_create_category(coco_output, object_name)

            # Find and process the mask file
            mask_filename = f"annotation_segmap_{key}.png"
            mask_path = os.path.join(folder_path, mask_filename)

            if not os.path.exists(mask_path):
                print(f"  Skipping annotation (missing mask file): {mask_filename}")
                continue

            segmentation, area, bbox = coco_helpers.process_mask(mask_path)
            
            # If the mask was valid, create the annotation entry
            if segmentation:
                annotation_entry = {
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation_entry)
                current_annotation_id += 1
        
        current_image_id += 1

    # --- 6. Write Final JSON File ---
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print("\n--- Conversion Complete! ---")
    print(f"Total Images: {len(coco_output['images'])}")
    print(f"Total Annotations: {len(coco_output['annotations'])}")
    print(f"Total Categories ({len(coco_output['categories'])}): "
          f"{[cat['name'] for cat in coco_output['categories']]}")
    print(f"\nYour COCO dataset is ready in: {config.OUTPUT_DIR}")
    print(f"Next step: Upload this entire '{config.OUTPUT_DIR}' folder to Roboflow.")

if __name__ == "__main__":
    run_conversion()