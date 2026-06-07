"""
make_annotation_taxonomy_figure.py
Builds figures/annotation_taxonomy_examples.png:
three rows (arrow / number-letter / freeform), two columns
(annotated image, binary annotation mask). Each row uses a square
crop centered on the annotation so the grid is aspect-uniform.
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_ROOT = Path(
    "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"
)
OUT_PATH = Path("./figures/annotation_taxonomy_examples.png")

# Top -> bottom, matching the thesis caption
ROWS = [
    ("arrow",         "9027_1_1"),
    ("number_letter", "9559_12_12"),
    ("freeform_bbox", "4218_1_1"),
]
CROP_MARGIN = 32   # pixels of context around the annotation bbox

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
})


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def square_bbox(mask_2d, margin):
    """Square bbox centered on the annotation; side = max dim + 2*margin."""
    binary = mask_2d > 127
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    cy = (int(ys.min()) + int(ys.max())) // 2
    cx = (int(xs.min()) + int(xs.max())) // 2
    side = max(int(ys.max()) - int(ys.min()),
               int(xs.max()) - int(xs.min())) + 2 * margin
    H, W = mask_2d.shape
    half = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    if x1 - x0 < side: x0 = max(0, x1 - side)
    if y1 - y0 < side: y0 = max(0, y1 - side)
    return x0, y0, x1, y1


def crop(img, bbox):
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


def show(ax, img):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999"); s.set_linewidth(0.5)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 8.0))

    for r, (_cls, base) in enumerate(ROWS):
        img  = load_rgb(DATASET_ROOT / "images" / f"{base}.jpg")
        mask = cv2.imread(
            str(DATASET_ROOT / "prompt_masks" / f"{base}_annot_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(f"{base}_annot_mask.png")

        bbox = square_bbox(mask, CROP_MARGIN)
        show(axes[r, 0], crop(img,  bbox))
        show(axes[r, 1], crop(mask, bbox))

    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
    fig.savefig(OUT_PATH)
    fig.savefig(OUT_PATH.with_suffix(".pdf"))
    plt.close(fig)
    print(f"-> {OUT_PATH}")
    print(f"-> {OUT_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()