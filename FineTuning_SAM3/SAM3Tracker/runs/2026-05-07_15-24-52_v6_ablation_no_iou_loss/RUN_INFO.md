# Run: v6_ablation_no_iou_loss

**Timestamp:** 2026-05-07_15-24-52

## Description

Loss ablation: drop the IoU-MSE term from the combined loss.

This run is identical to v6_arrow_pca_axis_freeform_mask EXCEPT the
training objective changes from
        20 * focal + dice + iou_mse
to
        20 * focal + dice

The focal weight (20) and dice weight (1) are NOT rebalanced. The
ablation measures the marginal contribution of the IoU-MSE term in
isolation — same dataset, same 30-epoch budget, same LR, same
warmup, same prompt routing, same frozen vision encoder.

The model still predicts iou_scores (architecturally unchanged); the
iou_scores head simply no longer receives a supervision signal, so
its weights drift only via downstream gradients into shared layers.
This matches how SAM-style ablations are typically reported.

Compare against the main run on:
  - best validation IoU and best validation loss (epoch numbers)
  - per-type test IoU + Boundary F1
  - convergence curve shape (faster / slower / different plateau)

Prompt routing, dataset, optimizer, scheduler, batch size and AMP
settings are all unchanged from the main run.

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
| Dataset | `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset` |

## Results

*(filled automatically at end of training)*

## Final results

| Metric | Value | Epoch |
|---|---|---|
| Best val loss | 0.3285 | 10 |
| Best val IoU | 0.7096 | 26 |
| Total epochs trained | 30 | - |

## Model stats

| | Count | % |
|---|---|---|
| Frozen params | 454,038,784 | 99.1% |
| Trainable params | 4,221,585 | 0.9% |
| Total params | 458,260,369 | 100% |

## Output files

- `best_iou_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/best_by_iou`
- `best_iou_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/best_by_iou.pth`
- `best_loss_pth`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/best_by_loss.pth`
- `best_model_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/best_by_loss`
- `loss_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/loss_curve.png`
- `loss_iou_curve`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/loss_iou_curve.png`
- `run_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss`
- `tensorboard_dir`: `/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/tensorboard`
