"""
count_dataset.py
----------------
Verify dataset counts and consistency for sam_finetuning_dataset (or any
similarly-structured directory). Reports:
  - per-split JSON entry counts and per-type breakdown
  - per-folder file counts on disk (images, masks, prompt_masks, originals)
  - cross-check: are JSON-referenced files actually on disk? are there
    orphan files on disk not referenced by any JSON?

Usage:
    python3 count_dataset.py /path/to/sam_finetuning_dataset
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path


def count_dir(path):
    """Count files in a directory (1 level deep, no recursion)."""
    if not path.is_dir():
        return 0
    return sum(1 for f in path.iterdir() if f.is_file())


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 count_dataset.py /path/to/dataset_dir")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"ERROR: {root} not a directory")
        sys.exit(1)

    print(f"Dataset root: {root}\n")

    # ── 1. JSON entry counts ─────────────────────────────────────────
    print("=" * 70)
    print("JSON entry counts")
    print("=" * 70)
    total_entries = 0
    referenced_files = {"images": set(), "masks": set(),
                        "prompt_masks": set(), "originals": set()}
    referenced_types = Counter()
    type_per_split = {}

    for split in ("train", "val", "test"):
        jpath = root / f"{split}.json"
        if not jpath.exists():
            print(f"  {split:<6} (file not found)")
            continue
        with open(jpath) as f:
            data = json.load(f)
        total_entries += len(data)
        type_counts = Counter(e.get("annotation_type", "?") for e in data)
        type_per_split[split] = type_counts
        print(f"  {split:<6} {len(data):>7,} entries")
        for t, c in sorted(type_counts.items()):
            pct = 100 * c / len(data) if data else 0
            print(f"        {t:<18} {c:>6,}  ({pct:5.1f}%)")
            referenced_types[t] += c

        # Track referenced filenames per directory
        for e in data:
            for field, dir_key in [("image", "images"),
                                    ("annotation", "masks"),
                                    ("annotation_mask", "prompt_masks"),
                                    ("original_clean_image", "originals")]:
                rel = e.get(field)
                if rel:
                    fname = os.path.basename(rel)
                    referenced_files[dir_key].add(fname)

    print(f"  {'TOTAL':<6} {total_entries:>7,} entries\n")
    if total_entries:
        print("  Global type breakdown:")
        for t, c in sorted(referenced_types.items()):
            print(f"    {t:<18} {c:>6,}  ({100*c/total_entries:5.1f}%)")

    # ── 2. Files on disk per directory ───────────────────────────────
    print()
    print("=" * 70)
    print("Files on disk per directory")
    print("=" * 70)
    on_disk = {}
    for d in ("images", "masks", "prompt_masks", "originals"):
        sub = root / d
        if sub.is_symlink():
            target = sub.resolve()
            n = count_dir(target)
            print(f"  {d:<14} {n:>7,} files  (symlink -> {target})")
        else:
            n = count_dir(sub)
            print(f"  {d:<14} {n:>7,} files")
        on_disk[d] = n

    # ── 3. Cross-check JSON references vs disk ──────────────────────
    print()
    print("=" * 70)
    print("Cross-check: JSON references vs files on disk")
    print("=" * 70)
    for d in ("images", "masks", "prompt_masks", "originals"):
        ref = referenced_files[d]
        sub = root / d
        target = sub.resolve() if sub.is_symlink() else sub
        existing = set()
        if target.is_dir():
            existing = {f.name for f in target.iterdir() if f.is_file()}
        missing = ref - existing
        orphan = existing - ref
        status = "OK" if not missing else "MISSING FILES"
        print(f"  {d:<14} referenced={len(ref):>6,}  on_disk={len(existing):>6,}"
              f"  missing={len(missing):>5,}  orphan={len(orphan):>5,}  [{status}]")
        if missing and len(missing) <= 5:
            for m in list(missing)[:5]:
                print(f"        missing: {m}")

    # ── 4. Sanity verdict ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    if total_entries == 41143:
        print("  Production split: 41,143 entries — matches your reported v6 dataset.")
    elif total_entries == 82875:
        print("  82,875 entries — this is the FULL raw-data prepare; 2x production.")
    else:
        print(f"  {total_entries:,} entries — neither 41,143 nor 82,875.")
        print(f"  Compare against the count expected for this dataset version.")


if __name__ == "__main__":
    main()
