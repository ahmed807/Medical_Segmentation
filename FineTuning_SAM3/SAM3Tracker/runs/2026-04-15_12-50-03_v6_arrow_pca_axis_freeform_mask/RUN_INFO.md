# Run: v6_arrow_pca_axis_freeform_mask

**Timestamp:** 2026-04-15_12-50-03

## Description

Replaced arrow tip detection with a PCA-based principal-axis algorithm.

The previous v5 (convex hull + perpendicular spread) failed on diagonal
arrows because the convex hull's "furthest pair" sometimes picked two
arrowhead corners (left vs right barb of the triangle) instead of tip
vs tail. It also occasionally returned hull vertices that fell OUTSIDE
the actual annotation pixels, producing prompts that missed the arrow.

New v6 algorithm:
  1. Compute PCA on all annotation pixels — main eigenvector = true
     arrow axis. This works robustly for any orientation because it
     uses the full pixel distribution, not just hull corners.
  2. Project pixels onto the axis. The min/max projections define the
     two endpoints of the arrow along its true direction.
  3. SNAP each endpoint to the nearest actual annotation pixel — this
     guarantees the prompt is always ON the arrow, never floating in
     empty space.
  4. Measure perpendicular spread at each end (outer 25% band). The
     arrowhead has wider perpendicular spread than the tail.
  5. Tip = end with LARGER perpendicular spread (arrowhead flare).

Freeform_bbox prompting unchanged: filled annotation contour passed as
input_masks to preserve the exact shape.

rect_bbox still uses a standard box prompt.

## Prompt routing

| Annotation type | Prompt strategy |
|---|---|
| arrow | Point prompt at tip (PCA-axis algorithm, snapped to mask) |
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
| Best val loss | 0.3626 | 16 |
| Best val IoU | 0.6987 | 16 |
| Total epochs trained | 30 | - |

## Model stats

| | Count | % |
|---|---|---|
| Frozen params | 454,038,784 | 99.1% |
| Trainable params | 4,221,585 | 0.9% |
| Total params | 458,260,369 | 100% |

## Output files

- `best_iou_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/best_by_iou`
- `best_iou_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/best_by_iou.pth`
- `best_loss_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/best_by_loss.pth`
- `best_model_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/best_by_loss`
- `loss_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/loss_curve.png`
- `loss_iou_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/loss_iou_curve.png`
- `run_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask`
- `tensorboard_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-15_12-50-03_v6_arrow_pca_axis_freeform_mask/tensorboard`
