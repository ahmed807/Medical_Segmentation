"""
compose_freeform_stage_coupling.py
==================================

Builds Figure 6.5 of the thesis: a focused, two-row, four-column
illustration of the freeform stage-coupling failure on a single example.

Layout:
                annotated input | annotation mask | predicted clean | SAM3 object mask
  Pipeline A:   [boy + contour] | [dashed contour] | [boy preserved] | [boy silhouette]
  Pipeline B:   [boy + contour] | [filled silhouette] | [boy removed] | [boy silhouette]

USAGE
-----
1. Fill the one TODO path below (pb_annotation_mask) using the lookup
   command from the README.
2. Run:  python3 compose_freeform_stage_coupling.py
3. Output goes to figures/pipeline_a_vs_b_examples.png.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================================
# PATHS (PA side confirmed from test.json; PB annotation mask is TODO)
# =========================================================================
DATASET = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"
RUN     = "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask"

PATHS = {
    # 1. annotated input — note .jpg, and the filename uses the PA basename
    "annotated_input":
        f"{DATASET}/images/2182_1_4.jpg",

    # 2. PA annotation mask — GT dashed contour, note "_annot_mask" filename
    "pa_annotation_mask":
        f"{DATASET}/prompt_masks/2182_1_4_annot_mask.png",

    # 3. PA predicted clean (FLUX output, boy preserved)
    "pa_predicted_clean":
        f"{RUN}/inference_results_full_v2/freeform_bbox/2182_1_4_clean.png",

    # 4. PA SAM3TrackerModel object mask (boy silhouette)
    "pa_object_mask":
        f"{RUN}/inference_results_full_v2/freeform_bbox/2182_1_4_object_mask.png",

    # 5. PB annotation mask (SAM-2 filled silhouette of the boy).
    # Lives in the Grounded-SAM-2 output directory, NOT alongside the
    # FLUX outputs. Naming pattern: {source_stem}_{detection_idx}_sam.png
    "pb_annotation_mask":
        "/home/ahma/Medical_Segmentation/GroundingSAM/zero_shot_v2_full_promptD_for_pipeline_b/pred_masks/2182_1_0_sam.png",

    # 6. PB predicted clean (FLUX output, boy removed)
    "pb_predicted_clean":
        f"{RUN}/pipeline_b_default/freeform_bbox/2182_1_det0_clean.png",

    # 7. PB SAM3TrackerModel object mask (boy silhouette — same as PA's)
    "pb_object_mask":
        f"{RUN}/pipeline_b_default/freeform_bbox/2182_1_det0_object_mask.png",
}

OUTPUT = "figures/pipeline_a_vs_b_examples.png"
MARGIN = 24
DPI    = 300


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"could not read: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def annotation_bbox(mask, margin=24):
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 127
    if not binary.any():
        raise RuntimeError("empty mask: cannot compute bounding box")
    ys, xs = np.where(binary)
    H, W = binary.shape
    return (max(int(xs.min()) - margin, 0),
            max(int(ys.min()) - margin, 0),
            min(int(xs.max()) + 1 + margin, W),
            min(int(ys.max()) + 1 + margin, H))


def crop(img, bbox):
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


def show(ax, img, title=None):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")
        s.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=10, pad=4)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.dpi": DPI,
    })

    print("Loading images...")
    imgs = {}
    for k, p in PATHS.items():
        print(f"  {k:25s} {p}")
        imgs[k] = read_image(p)
    print()

    # Crop bounding box from PB annotation mask (the larger of the two
    # masks). The boy is fully inside the crop in every panel.
    bbox = annotation_bbox(imgs["pb_annotation_mask"], margin=MARGIN)
    print(f"crop bbox = {bbox}  ({bbox[2]-bbox[0]}w x {bbox[3]-bbox[1]}h)")

    crops = {k: crop(v, bbox) for k, v in imgs.items()}

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2))
    titles = ["annotated input", "annotation mask",
              "predicted clean", "SAM3TrackerModel object mask"]

    show(axes[0, 0], crops["annotated_input"],     titles[0])
    show(axes[0, 1], crops["pa_annotation_mask"],  titles[1])
    show(axes[0, 2], crops["pa_predicted_clean"],  titles[2])
    show(axes[0, 3], crops["pa_object_mask"],      titles[3])

    show(axes[1, 0], crops["annotated_input"])
    show(axes[1, 1], crops["pb_annotation_mask"])
    show(axes[1, 2], crops["pb_predicted_clean"])
    show(axes[1, 3], crops["pb_object_mask"])

    axes[0, 0].text(-0.10, 0.5, "Pipeline A",
                    transform=axes[0, 0].transAxes,
                    ha="right", va="center",
                    fontsize=11, fontweight="bold")
    axes[1, 0].text(-0.10, 0.5, "Pipeline B",
                    transform=axes[1, 0].transAxes,
                    ha="right", va="center",
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    plt.savefig(OUTPUT)
    plt.close(fig)
    print(f"\nwrote: {OUTPUT}")


if __name__ == "__main__":
    main()