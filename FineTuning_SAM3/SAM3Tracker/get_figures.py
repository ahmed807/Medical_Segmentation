"""
compose_thesis_qualitative_figure.py
Stacks pre-rendered *_comparison.png strips from SAM3Tracker inference
into thesis-ready qualitative figures (per-class + one combined).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

INFERENCE_ROOT = Path(
    "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset/images"
)
OUT_DIR = Path("./thesis_figures_images")

SELECTIONS = {
    "freeform_bbox": ["4218_1_1", "89_0_0"],
    "arrow":         ["1893_7_7", "9027_1_1", "9514_1_1"],
    "number_letter": ["9426_3_6", "9559_12_12"],
}
CLASS_DISPLAY = {
    "freeform_bbox": "Freeform bbox",
    "arrow":         "Arrow",
    "number_letter": "Number / letter",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "savefig.dpi": 300,
})


def comp_path(cls, base):
    return INFERENCE_ROOT / cls / f"{base}.jpg"


def render(rows, out_path, with_class_label=False):
    """rows: list of (class, basename). Stacks vertically, one strip per row."""
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 1.7 * n))
    if n == 1:
        axes = [axes]
    prev_cls = None
    for ax, (cls, base) in zip(axes, rows):
        p = comp_path(cls, base)
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(0.5, 0.5, f"missing\n{p.name}", ha="center", va="center",
                    transform=ax.transAxes, color="#a44", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc"); s.set_linewidth(0.5)
        if with_class_label and cls != prev_cls:
            label = f"{CLASS_DISPLAY.get(cls, cls)}\n{base}"
        else:
            label = base
        ax.set_ylabel(label, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=45)
        prev_cls = cls

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = out_path.with_suffix(f".{ext}")
        fig.savefig(out)
        print(f"  -> {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== per-class figures ==")
    for cls, bases in SELECTIONS.items():
        render([(cls, b) for b in bases], OUT_DIR / f"qual_{cls}")

    print("\n== combined figure ==")
    combined = [(c, b) for c, bs in SELECTIONS.items() for b in bs]
    render(combined, OUT_DIR / "qual_combined", with_class_label=True)


if __name__ == "__main__":
    main()