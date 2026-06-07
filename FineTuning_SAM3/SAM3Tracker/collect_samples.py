import os
import shutil

DATASET = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset"
RESULTS = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/results_ep20"
OUT     = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/showcase_ep20"

samples = {
    "arrow": [
        "2176_0_0",
        "2952_0_1",
        "3755_2_2",
        "4206_1_4",
        "4466_0_1",
        "4734_11_11",
    ],
    "freeform_bbox": [
        "88_0_0",
        "245_2_4",
        "323_0_3",
        "323_1_6",
        "329_2_11",
        "406_1_1",
        "407_0_2",
        "586_0_1",
    ],
    "number_letter": [
        "297_0_0",
        "339_0_0",
        "613_7_7",
        "53_2_11",
        "251_0_1",
        "289_1_3",
    ],
}

for ann_type, entry_ids in samples.items():
    type_dir = os.path.join(OUT, ann_type)
    os.makedirs(type_dir, exist_ok=True)

    for entry_id in entry_ids:
        files = {
            "comparison": os.path.join(RESULTS, ann_type, f"{entry_id}_comparison.png"),
            "gt_mask":    os.path.join(DATASET, "masks", f"{entry_id}_segmap.png"),
            "pred_mask":  os.path.join(RESULTS, ann_type, f"{entry_id}_object_mask.png"),
        }

        for label, src in files.items():
            if os.path.exists(src):
                dst = os.path.join(type_dir, f"{entry_id}_{label}.png")
                shutil.copy2(src, dst)
                print(f"  {ann_type}/{entry_id}_{label}.png")
            else:
                print(f"  MISSING: {src}")

print(f"\nDone! Files saved to: {OUT}/")
