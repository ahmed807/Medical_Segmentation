"""
extract_thesis_samples.py
Copies dataset files (image, object mask, annotation mask, original)
for hand-picked basenames into per-sample folders for thesis figures.
"""
import shutil
from pathlib import Path

DATASET_ROOT = Path(
    "/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"
)
OUT_ROOT = Path("./thesis_samples")

SELECTIONS = {
    "freeform_bbox": ["4218_1_1", "89_0_0"],
    "arrow":         ["1893_7_7", "9027_1_1", "9514_1_1"],
    "number_letter": ["9426_3_6", "9559_12_12"],
}

# (subdir-in-dataset, suffix-on-disk, output-filename)
COMPONENTS = [
    ("images",       ".jpg",              "image.jpg"),
    ("masks",        "_segmap.png",       "object_mask.png"),
    ("prompt_masks", "_annot_mask.png",   "annotation_mask.png"),
]


def copy_one(cls, base):
    out_dir = OUT_ROOT / cls / base
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for subdir, suffix, out_name in COMPONENTS:
        src = DATASET_ROOT / subdir / f"{base}{suffix}"
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, out_dir / out_name)

    # original clean image — keyed on source ID (e.g. "4218_1_1" -> "4218")
    source_id = base.split("_", 1)[0]
    orig_src = DATASET_ROOT / "originals" / f"{source_id}_original.png"
    if orig_src.exists():
        shutil.copy2(orig_src, out_dir / "original.png")
    else:
        missing.append(str(orig_src))

    status = "OK" if not missing else f"MISSING {len(missing)}"
    print(f"  [{cls:14s}] {base:14s} -> {out_dir}  [{status}]")
    for m in missing:
        print(f"      missing: {m}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cls, bases in SELECTIONS.items():
        print(f"\n== {cls} ==")
        for b in bases:
            copy_one(cls, b)
    print(f"\nDone. Files in: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()