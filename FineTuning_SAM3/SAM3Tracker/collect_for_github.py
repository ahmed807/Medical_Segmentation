"""
collect_for_github.py
=====================

Build a clean, GitHub-ready directory of the thesis's final scripts.

Walks your source directory, copies the final used scripts into a
sensible folder layout, sanitises hard-coded local paths, and generates
README.md / requirements.txt / .gitignore / LICENSE / module __init__s
suitable for `git init && git push`.

The script does NOT touch your source files; it only reads and copies.
The output directory is built fresh each run (existing contents are
cleared first; the script asks for confirmation before doing this).

USAGE
-----
    python3 collect_for_github.py \\
        --source-root /home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker \\
        --grounded-sam-root /home/ahma/Medical_Segmentation/GroundingSAM \\
        --gs2-root /home/ahma/Grounded-SAM-2 \\
        --output-dir /home/ahma/medical-annotation-removal \\
        --repo-name medical-annotation-removal

Once it finishes, cd into the output directory and:

    git init
    git add -A
    git commit -m "Initial release"
    git branch -M main
    git remote add origin https://github.com/<your_user>/<repo>.git
    git push -u origin main
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


# =========================================================================
# What goes where in the new repo.
# =========================================================================
# Each entry maps {source_relative_path: destination_relative_path}.
# Source paths are relative to --source-root unless they start with a
# directory prefix indicating they live under one of the other roots.

# Files from the main SAM3Tracker project root
# Each value is the destination path in the new repo.
# Source paths can be either:
#   - a bare filename (looked up at source-root; auto-searched if missing)
#   - a subdirectory-relative path (looked up exactly at that location)
SAM3_TRACKER_FILES = {
    # Dataset preparation
    "prepare_dataset_tracker.py":          "data/prepare_dataset_tracker.py",
    # Training (snapshots are kept inside their run directories)
    "runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask/train_script_snapshot.py":
        "train/train_production.py",
    "runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss/train_script_snapshot.py":
        "train/train_ablation_no_iou.py",
    # Inference (production inference lives in tracker_arrow_opencv_fix_PRODUCTION/)
    "tracker_arrow_opencv_fix_PRODUCTION/inference_pipeline_b.py":
        "inference/inference_pipeline_b.py",
    "inference_zeroshot_pcs.py":           "inference/inference_zeroshot_pcs.py",
    "build_pipeline_b_input.py":           "inference/build_pipeline_b_input.py",
    # Evaluation
    "evaluate_boundary_f1.py":             "evaluate/evaluate_boundary_f1.py",
    "evaluate_fid_lpips.py":               "evaluate/evaluate_fid_lpips.py",
    "compute_fid_from_crops.py":           "evaluate/compute_fid_from_crops.py",
    "zero_shot_eval_v2_with_persistence.py": "evaluate/zero_shot_eval_v2_with_persistence.py",
    "eval_generic_per_true_class.py":      "evaluate/eval_generic_per_true_class.py",
    # Analysis (recent additions)
    "paired_rq2_comparison.py":            "analysis/paired_rq2_comparison.py",
    "analyze_pipeline_b_buckets.py":       "analysis/analyze_pipeline_b_buckets.py",
    "search_samples.py":                   "analysis/search_samples.py",
    "ref_lookup.py":                       "analysis/ref_lookup.py",
    # Figure-generation for the thesis
    "compose_chapter_6_figures.py":        "figures/compose_chapter_6_figures.py",
    "compose_freeform_stage_coupling.py":  "figures/compose_freeform_stage_coupling.py",
    "chapter_6_figure_samples.json":       "figures/chapter_6_figure_samples.json",
}

# Files from the GroundingSAM project (Stage 1 dataset preparation)
GROUNDING_SAM_FILES = {
    "prepare_dataset_gdino.py":            "data/prepare_dataset_gdino.py",
}

# Documentation files (collected if present, optional)
OPTIONAL_DOCS = {
    "HANDOFF.md":          "docs/HANDOFF.md",
    "README_references.md": "docs/README_references.md",
    "references_tracking.json": "docs/references_tracking.json",
}

# =========================================================================
# Path-sanitisation rules. These textual replacements are applied to
# every .py / .md / .json file that gets copied, so the repo doesn't
# leak your home-directory paths to the world.
# =========================================================================
PATH_SANITISERS = [
    # absolute user paths -> placeholder
    (re.compile(r"/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/?"),
     "${SAM3_TRACKER_ROOT}/"),
    (re.compile(r"/home/ahma/Medical_Segmentation/GroundingSAM/?"),
     "${GROUNDED_SAM_ROOT}/"),
    (re.compile(r"/home/ahma/Grounded-SAM-2/?"),
     "${GS2_ROOT}/"),
    (re.compile(r"/home/ahma/Medical_Segmentation/?"),
     "${PROJECT_ROOT}/"),
    (re.compile(r"/home/ahma/?"),
     "${HOME}/"),
    # specific run-directory names left in code: kept as configurable placeholders
    (re.compile(r"runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask"),
     "runs/${PRODUCTION_RUN}"),
    (re.compile(r"runs/2026-05-17_12-37-43_v6_arrow_pca_axis_freeform_mask"),
     "runs/${GENERIC_RUN}"),
]


# =========================================================================
README_TEMPLATE = """\
# {repo_name}

Code accompanying the M.Sc. thesis
**"Bridging the Annotation Distribution Gap in Medical Imaging:
A Three-Stage Pipeline for Automated Detection, Segmentation,
and Removal of Visual Annotations from Medical Educational Imagery."**

Technical University of Munich, M.Sc. Data Engineering and Analytics.

## Overview

The pipeline removes visual annotations (arrows, glyphs, freeform
contours) from annotated educational figures to produce clean image
pairs suitable for downstream medical vision-language model training.

It has three stages:

1. **Detection** &mdash; GroundedSAM 2 (open-vocabulary text-prompted).
2. **Segmentation** &mdash; fine-tuned SAM3TrackerModel with type-specific
   prompt routing (4.2 M of 458 M parameters trained; 0.9 %%).
3. **Inpainting** &mdash; FLUX.1 Fill on the dilated annotation mask.

Evaluated under a dual-pipeline protocol that supports per-stage error
attribution: Pipeline A uses ground-truth masks at Stage 1; Pipeline B
uses GroundedSAM 2 predictions.

Trained model checkpoints are released separately on HuggingFace:
👉 <https://huggingface.co/USERNAME/medical-annotation-removal>

## Repository layout

```
{repo_name}/
├── data/         dataset preparation scripts (SAM3Tracker, GroundedSAM views)
├── train/        training scripts (production + loss ablation)
├── inference/    zero-shot inference and Pipeline B input construction
├── evaluate/     IoU, Boundary F1, FID, LPIPS evaluation
├── analysis/    paired comparisons, bucket analysis, sample lookup
├── figures/      thesis figure generation
└── docs/         project notes and references
```

## Quick start

### 1. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure paths

Several scripts reference environment-variable placeholders for absolute
paths. Set these before running:

```bash
export SAM3_TRACKER_ROOT=/path/to/your/SAM3Tracker
export GROUNDED_SAM_ROOT=/path/to/your/GroundingSAM
export GS2_ROOT=/path/to/your/Grounded-SAM-2
export PRODUCTION_RUN=2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask
```

The placeholders that appear in the scripts (`${{SAM3_TRACKER_ROOT}}`,
`${{PROJECT_ROOT}}`, etc.) are not Python variables &mdash; they are markers
indicating which absolute paths you should substitute when running.

### 3. Run inference

Download the trained SAM3TrackerModel checkpoint from HuggingFace, then:

```bash
python3 inference/inference_pipeline_b.py \\
    --checkpoint /path/to/best_by_iou.pth \\
    --pb-input-json /path/to/pipeline_b_input_default.json \\
    --out-dir /path/to/output
```

See each script's `--help` for full argument details.

## Pipeline B input format

Pipeline B consumes a JSON file produced by `inference/build_pipeline_b_input.py`
with the following per-entry schema:

```json
{{
  "image": "/path/to/annotated_image.jpg",
  "annotation_mask": "/path/to/sam2_predicted_mask.png",
  "original_clean_image": "/path/to/original_unannotated.png",
  "prompt_box": [x_min, y_min, x_max, y_max],
  "prompt_text": "dashed line",
  "annotation_type": "freeform_bbox",
  "source_image": "images/<source_stem>.jpg",
  "detection_idx": 0
}}
```

## Citation

If you use this code or the released checkpoints, please cite:

```bibtex
@mastersthesis{{your_thesis_2026,
  title  = {{Bridging the Annotation Distribution Gap in Medical Imaging}},
  author = {{Your Name}},
  school = {{Technical University of Munich}},
  year   = {{2026}},
  type   = {{M.Sc. Thesis}}
}}
```

## License

Source code is released under the MIT License (see `LICENSE`).

The released checkpoints are derived from publicly available
foundation models (SAM 3 Tracker, GroundedSAM 2, FLUX.1 Fill); please
check the upstream licences of those models before any commercial use.
"""


REQUIREMENTS = """\
# Core ML stack
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0

# Foundation model loaders
transformers>=4.40.0
huggingface_hub>=0.20.0
safetensors>=0.4.0

# Evaluation
scikit-image>=0.21.0
scipy>=1.11.0
lpips>=0.1.4

# Inpainting (FLUX requires the diffusers main branch features)
diffusers>=0.27.0
accelerate>=0.27.0

# Data / IO
pandas>=2.0.0
tqdm>=4.65.0

# Visualisation (for thesis figures)
matplotlib>=3.7.0

# Optional: training
einops>=0.7.0
tensorboard>=2.14.0
"""


GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual envs
venv/
env/
.venv/

# Distribution / packaging
build/
dist/
*.egg-info/
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo

# Local data / model artefacts (released separately on HuggingFace)
runs/
checkpoints/
*.pth
*.pt
*.safetensors

# Dataset (not redistributable)
sam_finetuning_dataset/
sam_finetuning_dataset_generic/
gdino_finetuning_dataset/
pipeline_b_input*.json

# Generated outputs
figures/*.png
figures/*.pdf
inference_results*/
pipeline_b_default/

# OS
.DS_Store
Thumbs.db
"""


MIT_LICENSE = """\
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# Per-directory README content
SUBDIR_READMES = {
    "data": "## Dataset preparation\n\n"
            "- `prepare_dataset_tracker.py` &mdash; builds the per-(image, annotation) "
            "view used by the SAM3TrackerModel fine-tune.\n"
            "- `prepare_dataset_gdino.py` &mdash; builds the per-folder view used by "
            "the GroundedSAM 2 detection-stage evaluation.\n\n"
            "The two scripts produce slightly different annotation counts (82,875 "
            "vs 82,869 in this project) because their validity filters differ "
            "marginally; the 0.007 %% difference does not affect any reported "
            "metric.\n",

    "train": "## Training\n\n"
             "- `train_script_snapshot.py` &mdash; production training script "
             "(loss = 20 \u00b7 focal + Dice + IoU-MSE).\n"
             "- `train_ablation_no_iou.py` &mdash; same script with the IoU-MSE term "
             "removed, used for the loss-ablation comparison.\n\n"
             "All runs: AdamW with lr=5e-5 and weight decay 1e-4, batch 48, "
             "30 epochs in bfloat16 on a single A40.\n",

    "inference": "## Inference\n\n"
                 "- `inference_zeroshot_pcs.py` &mdash; runs the production model on "
                 "the SAM3Tracker test split (Pipeline A).\n"
                 "- `inference_pipeline_b.py` &mdash; consumes the GroundedSAM 2 "
                 "detection output and runs the full Pipeline B chain.\n"
                 "- `build_pipeline_b_input.py` &mdash; constructs the input JSON for "
                 "Pipeline B from the upstream Grounded-SAM-2 detection results.\n",

    "evaluate": "## Evaluation\n\n"
                "- `evaluate_boundary_f1.py` &mdash; per-instance Boundary F1 with a "
                "2-pixel tolerance.\n"
                "- `evaluate_fid_lpips.py` &mdash; per-class FID via torchvision "
                "InceptionV3 ImageNet1K_V1 weights; per-sample LPIPS.\n"
                "- `compute_fid_from_crops.py` &mdash; FID on cropped regions around "
                "the inpainted area (the per-class FID used in Pipeline B "
                "evaluation).\n"
                "- `zero_shot_eval_v2_with_persistence.py` &mdash; zero-shot baselines "
                "with intermediate results persisted between stages.\n"
                "- `eval_generic_per_true_class.py` &mdash; per-true-class metrics "
                "for the generic-routing baseline.\n",

    "analysis": "## Analysis\n\n"
                "- `paired_rq2_comparison.py` &mdash; paired comparison of "
                "type-specific vs generic routing on identical basenames.\n"
                "- `analyze_pipeline_b_buckets.py` &mdash; bucketed error-propagation "
                "analysis (zero / low / mid / high IoU buckets).\n"
                "- `search_samples.py` &mdash; CSV-driven sample lookup by class, "
                "metric, and threshold.\n"
                "- `ref_lookup.py` &mdash; citation key lookup.\n",

    "figures": "## Figure generation for the thesis\n\n"
               "- `compose_chapter_6_figures.py` &mdash; composite figures (Pipeline "
               "A example panels, Pipeline B failure modes).\n"
               "- `compose_freeform_stage_coupling.py` &mdash; the 2-row x 4-column "
               "figure that demonstrates the stage-coupling failure on a "
               "single representative example.\n"
               "- `chapter_6_figure_samples.json` &mdash; entry IDs for the figure "
               "composers.\n",

    "docs": "## Project documentation\n\n"
            "- `HANDOFF.md` &mdash; handoff notes describing the production run, "
            "the dataset layout, and known issues.\n"
            "- `README_references.md` &mdash; bibliographic notes for the cited "
            "works in the thesis.\n"
            "- `references_tracking.json` &mdash; machine-readable index of cited "
            "references.\n",
}


# =========================================================================
def sanitise_file_content(text):
    """Apply path-sanitiser rules to the text."""
    for pat, repl in PATH_SANITISERS:
        text = pat.sub(repl, text)
    return text


def copy_with_sanitisation(src_path, dst_path):
    """Copy a single file. For text files, apply path sanitisation."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.suffix in (".py", ".md", ".json", ".txt", ".cfg", ".yaml", ".yml"):
        try:
            text = src_path.read_text(encoding="utf-8")
            text = sanitise_file_content(text)
            dst_path.write_text(text, encoding="utf-8")
        except UnicodeDecodeError:
            # binary-ish file slipped through; just copy
            shutil.copy2(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def auto_search(src_root, rel_path, search_root):
    """If src_root/rel_path doesn't exist, search search_root for the
    bare filename. Returns the resolved Path or None.

    For bare filenames (no slashes), single matches are returned. For
    sub-path lookups (e.g. 'runs/X/foo.py'), no search is attempted —
    those are expected to live exactly where the mapping says, and a
    mismatch means the run dir name changed and the user needs to fix
    the mapping.
    """
    direct = src_root / rel_path
    if direct.exists():
        return direct, "primary"

    # only auto-search for bare filenames
    if "/" in rel_path:
        return None, None

    matches = list(search_root.rglob(rel_path))
    # exclude anything in venv / __pycache__ / .git
    matches = [m for m in matches if not any(
        part in {"venv", ".venv", "__pycache__", ".git", "node_modules"}
        for part in m.parts
    )]
    if len(matches) == 1:
        return matches[0], "found"
    if len(matches) > 1:
        return matches, "multiple"
    return None, None


def collect(args):
    src_sam3   = Path(args.source_root)
    src_gdino  = Path(args.grounded_sam_root) if args.grounded_sam_root else None
    out        = Path(args.output_dir)

    # Confirm overwrite
    if out.exists() and any(out.iterdir()):
        if not args.yes:
            ans = input(f"\n  {out} exists and is not empty. Clear it? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("aborted.")
                return 1
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Build the file plan: (src_path, dst_path, required, label)
    plan = []
    for rel, dst_rel in SAM3_TRACKER_FILES.items():
        plan.append((src_sam3 / rel, out / dst_rel, True, "SAM3Tracker"))
    if src_gdino:
        for rel, dst_rel in GROUNDING_SAM_FILES.items():
            plan.append((src_gdino / rel, out / dst_rel, True, "GroundedSAM"))
    for rel, dst_rel in OPTIONAL_DOCS.items():
        plan.append((src_sam3 / rel, out / dst_rel, False, "docs"))

    # Execute
    print(f"\nCollecting files into {out}/...\n")
    missing = []
    multiple = []
    copied = []
    for src, dst, required, label in plan:
        # If src doesn't exist, try auto-search for bare filenames
        if not src.exists():
            src_root_for_label = {
                "SAM3Tracker": src_sam3,
                "GroundedSAM": src_gdino if src_gdino else src_sam3,
                "docs": src_sam3,
            }.get(label, src_sam3)
            rel = src.relative_to(src_root_for_label) if src.is_absolute() and \
                  src_root_for_label in src.parents else None
            if rel is not None:
                resolved, status = auto_search(src_root_for_label, str(rel), src_root_for_label)
                if status == "found":
                    src = resolved
                    print(f"  [{label:12s}] {rel.name:50s} -> {dst.relative_to(out)}  (auto-found at {src.relative_to(src_root_for_label)})")
                    copy_with_sanitisation(src, dst)
                    copied.append(dst.relative_to(out))
                    continue
                elif status == "multiple":
                    multiple.append((rel.name, resolved))
                    print(f"  [{label:12s}] {rel.name:50s} AMBIGUOUS: {len(resolved)} matches")
                    for m in resolved:
                        print(f"                  candidate: {m.relative_to(src_root_for_label)}")
                    continue

        if src.exists():
            copy_with_sanitisation(src, dst)
            copied.append(dst.relative_to(out))
            print(f"  [{label:12s}] {src.name:50s} -> {dst.relative_to(out)}")
        else:
            if required:
                missing.append(src)
            tag = "MISSING (required)" if required else "skipped (optional)"
            print(f"  [{label:12s}] {src.name:50s} {tag}")

    # Top-level files
    print()
    (out / "README.md").write_text(
        README_TEMPLATE.format(repo_name=args.repo_name))
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / ".gitignore").write_text(GITIGNORE)
    (out / "LICENSE").write_text(MIT_LICENSE)
    print("  Wrote: README.md, requirements.txt, .gitignore, LICENSE")

    # Per-directory READMEs
    for subdir, content in SUBDIR_READMES.items():
        target = out / subdir
        if target.exists():
            (target / "README.md").write_text(content)
            print(f"  Wrote: {subdir}/README.md")

    print(f"\nDone. {len(copied)} files copied into {out}.")
    if multiple:
        print(f"\n{len(multiple)} files had AMBIGUOUS matches (multiple copies under source-root):")
        for name, paths in multiple:
            print(f"  - {name}")
            for p in paths:
                print(f"      candidate: {p}")
        print("  → Update SAM3_TRACKER_FILES with the specific subdirectory-relative path you want, then re-run.")
    if missing:
        print(f"\n{len(missing)} required files were not found anywhere under source-root:")
        for m in missing:
            print(f"  - {m}")
        print("  → If you know where they live, update SAM3_TRACKER_FILES with the correct subpath and re-run.")
        print("  → If the file genuinely doesn't exist, remove it from SAM3_TRACKER_FILES.")

    print("\nNext steps:")
    print(f"  cd {out}")
    print( "  git init")
    print( "  git add -A")
    print(f"  git commit -m 'Initial release: {args.repo_name}'")
    print( "  git branch -M main")
    print(f"  git remote add origin https://github.com/<your_user>/{args.repo_name}.git")
    print( "  git push -u origin main")

    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True,
                    help="root of the SAM3Tracker project on this machine")
    ap.add_argument("--grounded-sam-root", default=None,
                    help="root of the GroundingSAM project (for "
                         "prepare_dataset_gdino.py). Optional.")
    ap.add_argument("--gs2-root", default=None,
                    help="root of the Grounded-SAM-2 project. Optional; "
                         "currently used only for path sanitisation.")
    ap.add_argument("--output-dir", required=True,
                    help="where to build the clean repo directory")
    ap.add_argument("--repo-name", default="medical-annotation-removal",
                    help="name of the new repo (used in README and git commands)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="don't prompt before clearing the output directory")
    args = ap.parse_args()
    sys.exit(collect(args))


if __name__ == "__main__":
    main()
