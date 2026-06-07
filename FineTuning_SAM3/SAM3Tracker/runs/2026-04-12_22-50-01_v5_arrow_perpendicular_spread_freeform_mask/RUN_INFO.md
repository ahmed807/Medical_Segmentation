# Run: v5_arrow_perpendicular_spread_freeform_mask

**Timestamp:** 2026-04-12_22-50-01

## Description

Replaced arrow tip detection with a perpendicular-width algorithm.

Previous approaches failed on arrows with thick shafts, where counting
pixels within a radius couldn't discriminate the arrowhead from the
tail — both ends had similar pixel counts.

New algorithm:
  1. Find the two convex hull points furthest apart (arrow endpoints).
  2. Compute the arrow's main axis (end_a to end_b).
  3. At each endpoint, measure how wide the shape is PERPENDICULAR to
     the main axis, within a band covering the outer 25% of the arrow.
  4. Tip = endpoint with LARGER perpendicular spread (arrowhead flare).

This correctly identifies the tip regardless of shaft thickness because
only the arrowhead triangle spreads sideways — the tail is always just
as wide as the shaft.

Freeform_bbox prompting unchanged: filled annotation contour passed as
input_masks to preserve the exact shape.

rect_bbox still uses a standard box prompt.

## Prompt routing

| Annotation type | Prompt strategy |
|---|---|
| arrow | Point prompt at tip (perpendicular-spread algorithm) |
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
| Best val loss | 0.3697 | 7 |
| Best val IoU | 0.6939 | 22 |
| Total epochs trained | 30 | - |

## Model stats

| | Count | % |
|---|---|---|
| Frozen params | 454,038,784 | 99.1% |
| Trainable params | 4,221,585 | 0.9% |
| Total params | 458,260,369 | 100% |

## Output files

- `best_iou_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/best_by_iou`
- `best_iou_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/best_by_iou.pth`
- `best_loss_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/best_by_loss.pth`
- `best_model_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/best_by_loss`
- `loss_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/loss_curve.png`
- `loss_iou_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/loss_iou_curve.png`
- `run_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask`
- `tensorboard_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-12_22-50-01_v5_arrow_perpendicular_spread_freeform_mask/tensorboard`
