# Run: v3_arrow_fix_freeform_mask

**Timestamp:** 2026-04-05_15-51-30

## Description

Fixed arrow tip detection: two-endpoint algorithm picks the pointy end
(fewer nearby pixels) instead of max-distance-from-centroid which was
a coin flip between tip and tail.

Changed freeform_bbox prompting: instead of a rectangular box approximation,
the filled annotation contour is passed as input_masks to Sam3Tracker.
This preserves the exact freeform shape and prevents the model from
segmenting the annotation lines along with the object.

rect_bbox still uses a standard box prompt (no information loss).

## Prompt routing

| Annotation type | Prompt strategy |
|---|---|
| arrow | Point prompt at tip (two-endpoint pointy-end algorithm) |
| number_letter | Multiple positive point prompts on annotation pixels |
| rect_bbox | Box prompt from annotation bounding box |
| freeform_bbox | Filled contour as input_masks + box context |

## Hyperparameters

| Parameter | Value |
|---|---|
| Model | `facebook/sam3` (Sam3TrackerModel) |
| Batch size | 48 |
| Grad accumulation | 1 |
| Effective batch | 48 |
| Epochs | 30 |
| Learning rate | 5e-05 |
| Warmup epochs | 3 |
| Weight decay | 0.0001 |
| Frozen prefixes | ['vision_encoder'] |
| Loss | 20×Focal + Dice + IoU-MSE |
| GT mask size | 256×256 |
| N letter points | 5 |
| Checkpoint every | 5 epochs |
| Dataset | `/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset` |

## Results

*(filled automatically at end of training)*

## Final results

| Metric | Value | Epoch |
|---|---|---|
| Best val loss | 0.3556 | 11 |
| Best val IoU | 0.6759 | 30 |
| Total epochs trained | 30 | - |

## Model stats

| | Count | % |
|---|---|---|
| Frozen params | 454,038,784 | 99.1% |
| Trainable params | 4,221,585 | 0.9% |
| Total params | 458,260,369 | 100% |

## Output files

- `best_iou_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/best_by_iou`
- `best_iou_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/best_by_iou.pth`
- `best_loss_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/best_by_loss.pth`
- `best_model_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/best_by_loss`
- `loss_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/loss_curve.png`
- `loss_iou_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/loss_iou_curve.png`
- `run_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask`
- `tensorboard_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-05_15-51-30_v3_arrow_fix_freeform_mask/tensorboard`
