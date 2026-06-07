"""
export_split_manifest.py
========================

Run this ONCE on the production server. It reads the production split
JSONs and writes a single portable file, split_manifest.json, that pins
exactly which entries belong to train / val / test.

The key used is `unique_base` = "{folder_name}_{key}", which is the stem
of the `image` field (images/<unique_base>.jpg). Both prepare_dataset
scripts construct this string identically, so it is a stable join key
that does not depend on process-completion order, the random seed, or
the machine.

Copy split_manifest.json to the other server next to
prepare_dataset_generic.py and run that script with --split-manifest.

Usage
-----
    python export_split_manifest.py \
        --dataset-dir /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset \
        --out split_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def unique_base_from_entry(entry: dict) -> str:
    """images/6786_0_3.jpg -> 6786_0_3"""
    return Path(entry["image"]).stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True,
                    help="Production sam_finetuning_dataset directory "
                         "(must contain train.json, val.json, test.json)")
    ap.add_argument("--out", default="split_manifest.json")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    manifest: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    seen: dict[str, str] = {}        # unique_base -> split, to detect collisions
    collisions = 0

    for split in ("train", "val", "test"):
        path = ds / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        with open(path) as f:
            entries = json.load(f)

        ids: list[str] = []
        for e in entries:
            ub = unique_base_from_entry(e)
            if ub in seen:
                collisions += 1
                if collisions <= 10:
                    print(f"[warn] {ub} appears in both "
                          f"{seen[ub]} and {split}")
            else:
                seen[ub] = split
            ids.append(ub)

        manifest[split] = ids
        counts[split] = len(ids)

    # Sanity: total unique ids should equal sum of split sizes minus collisions.
    total_ids = sum(counts.values())
    unique_ids = len(seen)

    payload = {
        "format": "split_manifest_v1",
        "key": "unique_base (stem of the image field)",
        "counts": counts,
        "splits": manifest,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"  train {counts['train']:>8,}")
    print(f"  val   {counts['val']:>8,}")
    print(f"  test  {counts['test']:>8,}")
    print(f"  total {total_ids:>8,}  (unique ids: {unique_ids:,})")
    if collisions:
        print(f"  [WARNING] {collisions} unique_base collisions across "
              f"splits. The first-seen split wins on the generic side; "
              f"investigate before trusting the paired comparison.")
    else:
        print("  no cross-split id collisions")


if __name__ == "__main__":
    main()
