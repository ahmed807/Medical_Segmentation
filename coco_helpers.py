# My_SAM_Project/coco_helpers.py

import cv2
import datetime

def init_coco_structure():
    """Returns the basic dictionary structure for a COCO JSON file."""
    return {
        "info": {
            "description": "My Custom Dataset in COCO format",
            "version": "1.0",
            "year": datetime.date.today().year,
            "date_created": datetime.date.today().isoformat()
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

def get_or_create_category(coco_data, object_name):
    """
    Finds an existing category or creates a new one.
    Returns the category_id.
    """
    # # --- ! DATA CORRECTION WARNING ! ---
    # if object_name == "horse":
    #     print(f"  WARNING: Correcting 'horse' to 'dog'. Please fix this in your original labels.json file!")
    #     object_name = "dog"
    # # -----------------------------------

    # Check if category already exists
    for category in coco_data["categories"]:
        if category["name"] == object_name:
            return category["id"]

    # Category not found, create a new one
    new_category_id = len(coco_data["categories"]) + 1
    category_entry = {
        "id": new_category_id,
        "name": object_name,
        "supercategory": object_name
    }
    coco_data["categories"].append(category_entry)
    
    return new_category_id

def process_mask(mask_path):
    """
    Loads a mask file and finds contours, area, and bounding box.
    Returns (segmentation, area, bbox) or None if invalid.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  Skipping: Cannot read mask file {mask_path}")
        return None, None, None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"  Skipping: No contours found in {mask_path}")
        return None, None, None

    segmentation = []
    total_area = 0
    
    # Use the mask itself to get the main bounding box
    # This is more reliable if contours are fragmented
    x, y, w, h = cv2.boundingRect(mask)
    bbox = [int(x), int(y), int(w), int(h)] # COCO format: [x, y, width, height]

    for contour in contours:
        # A valid polygon for COCO needs at least 3 points (6 values)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
            total_area += cv2.contourArea(contour)

    if not segmentation:
        print(f"  Skipping: No valid contours found in {mask_path}")
        return None, None, None

    return segmentation, float(total_area), bbox