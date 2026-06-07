"""
compute_fid_from_crops.py
=========================

Computes FID from crops already saved by evaluate_fid_lpips.py, then merges
the result with the existing per-sample LPIPS CSV to produce the final
summary.

WHY THIS EXISTS
---------------
clean-fid downloads its InceptionV3 checkpoint from Dropbox, which is
blocked on some networks (e.g. TUM LRZ). This script uses torchvision's
InceptionV3 instead — its weights live on PyTorch's CDN, which is the
same source that already worked for the LPIPS/AlexNet download.

The absolute FID values will differ slightly from clean-fid because of
internal preprocessing differences (PIL vs OpenCV resizing, weight
checkpoint version). What matters for the thesis is that the SAME
evaluator is used for Pipeline A and Pipeline B — that is guaranteed as
long as both runs use this script.

USAGE
-----
After evaluate_fid_lpips.py has finished cropping (even if it crashed
during the FID step), run:

    pip install torch torchvision numpy scipy pillow tqdm

    python compute_fid_from_crops.py \\
        --eval-dir   ./eval_fid_lpips \\
        --batch-size 32

The script will:
  1. Find the saved crops in <eval-dir>/crops/<type>/{predicted,original}/
  2. Compute FID per annotation type and overall
  3. Read <eval-dir>/lpips_per_sample.csv for LPIPS stats
  4. Write <eval-dir>/fid_lpips_summary.csv
  5. Print a markdown summary table
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor — 2048-d pool features (pre-classifier)
# ---------------------------------------------------------------------------

class InceptionV3Pool(nn.Module):
    """
    InceptionV3 with the final classifier replaced by Identity, so the
    forward pass returns the 2048-d pooled feature vector used for FID.
    """

    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        m = inception_v3(weights=weights, aux_logits=True)
        m.fc = nn.Identity()
        m.eval()
        self.model = m

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Crop dataset — preprocessing for InceptionV3
# ---------------------------------------------------------------------------

# torchvision ImageNet normalisation, matches Inception_V3_Weights.IMAGENET1K_V1.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class CropFolder(Dataset):
    def __init__(self, folder: Path):
        self.paths = sorted(folder.glob("*.png"))
        # Inception expects 299x299. Crops are 256x256 from
        # evaluate_fid_lpips.py, so we resize up bilinearly.
        self.tf = transforms.Compose([
            transforms.Resize(
                (299, 299),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


def extract_features(folder: Path, model: nn.Module, device: str,
                     batch_size: int, num_workers: int,
                     desc: str = "") -> np.ndarray | None:
    if not folder.exists():
        return None
    ds = CropFolder(folder)
    if len(ds) == 0:
        return None

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=desc, leave=False, unit="batch"):
            batch = batch.to(device, non_blocking=True)
            f = model(batch)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid(feats_a: np.ndarray, feats_b: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Standard Frechet Inception Distance:
        FID = ||mu_a - mu_b||^2  +  Tr(Sigma_a + Sigma_b - 2 * (Sigma_a Sigma_b)^0.5)
    """
    if feats_a.shape[0] < 2 or feats_b.shape[0] < 2:
        return float("nan")

    mu_a = feats_a.mean(axis=0)
    mu_b = feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b

    # Matrix square root of the product of covariances.
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)

    # Add a small ridge if the result is non-finite (numerically singular).
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = sqrtm((sigma_a + offset) @ (sigma_b + offset))

    # The product is symmetric in expectation; small imaginary components
    # are numerical noise. Warn if they're not small, then drop.
    if np.iscomplexobj(covmean):
        max_imag = float(np.max(np.abs(covmean.imag)))
        if max_imag > 1e-3:
            print(f"  [warn] non-trivial imaginary component in covmean: "
                  f"max |imag| = {max_imag:.4g}")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2.0 * covmean))


# ---------------------------------------------------------------------------
# LPIPS aggregation from the existing CSV
# ---------------------------------------------------------------------------

def load_lpips_csv(csv_path: Path) -> dict[str, list[float]]:
    """Returns {annotation_type: [scores]}."""
    by_type: dict[str, list[float]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_type[row["annotation_type"]].append(float(row["lpips"]))
    return by_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-dir", required=True,
                    help="The --out-dir from evaluate_fid_lpips.py")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    crops_dir = eval_dir / "crops"
    lpips_csv = eval_dir / "lpips_per_sample.csv"

    if not crops_dir.exists():
        raise SystemExit(f"crops directory not found: {crops_dir}")
    if not lpips_csv.exists():
        raise SystemExit(f"LPIPS per-sample CSV not found: {lpips_csv}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading InceptionV3 (torchvision, ImageNet weights)...")
    model = InceptionV3Pool().to(device).eval()

    annotation_types = sorted([
        d.name for d in crops_dir.iterdir()
        if d.is_dir() and d.name != "_all"
    ])
    if not annotation_types:
        raise SystemExit(f"No annotation type folders found under {crops_dir}")

    # ---------------- per-type FID ----------------
    fid_per_type: dict[str, float] = {}
    # Cache features so we don't re-extract them for the overall computation.
    feats_cache: dict[str, dict[str, np.ndarray]] = {}

    print()
    print("Computing FID per annotation type...")
    for ann_type in annotation_types:
        pred_dir = crops_dir / ann_type / "predicted"
        orig_dir = crops_dir / ann_type / "original"
        if not (pred_dir.exists() and orig_dir.exists()):
            print(f"  {ann_type}: skipped (missing predicted/original folder)")
            fid_per_type[ann_type] = float("nan")
            continue

        feats_pred = extract_features(
            pred_dir, model, device, args.batch_size, args.num_workers,
            desc=f"  {ann_type}/predicted",
        )
        feats_orig = extract_features(
            orig_dir, model, device, args.batch_size, args.num_workers,
            desc=f"  {ann_type}/original",
        )
        if feats_pred is None or feats_orig is None:
            fid_per_type[ann_type] = float("nan")
            continue

        feats_cache[ann_type] = {"predicted": feats_pred, "original": feats_orig}
        score = compute_fid(feats_pred, feats_orig)
        fid_per_type[ann_type] = score
        print(f"  {ann_type:<16}: FID = {score:.3f}  "
              f"(n_pred={len(feats_pred)}, n_orig={len(feats_orig)})")

    # ---------------- overall FID ----------------
    print("\nComputing overall FID (concatenating cached features)...")
    all_pred = [v["predicted"] for v in feats_cache.values()]
    all_orig = [v["original"] for v in feats_cache.values()]
    if all_pred and all_orig:
        fid_overall = compute_fid(
            np.concatenate(all_pred, axis=0),
            np.concatenate(all_orig, axis=0),
        )
    else:
        fid_overall = float("nan")
    fid_per_type["OVERALL"] = fid_overall
    print(f"  OVERALL         : FID = {fid_overall:.3f}")

    # ---------------- LPIPS aggregation ----------------
    print(f"\nLoading LPIPS values from {lpips_csv.name}...")
    lpips_by_type = load_lpips_csv(lpips_csv)
    all_lpips = [s for v in lpips_by_type.values() for s in v]

    # ---------------- summary CSV ----------------
    summary_csv = eval_dir / "fid_lpips_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "type", "n", "lpips_mean", "lpips_median", "lpips_std", "fid",
        ])
        for ann_type in sorted(lpips_by_type.keys()):
            scores = lpips_by_type[ann_type]
            w.writerow([
                ann_type, len(scores),
                f"{statistics.mean(scores):.4f}",
                f"{statistics.median(scores):.4f}",
                f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "",
                f"{fid_per_type.get(ann_type, float('nan')):.3f}",
            ])
        if all_lpips:
            w.writerow([
                "OVERALL", len(all_lpips),
                f"{statistics.mean(all_lpips):.4f}",
                f"{statistics.median(all_lpips):.4f}",
                f"{statistics.stdev(all_lpips):.4f}" if len(all_lpips) > 1 else "",
                f"{fid_per_type.get('OVERALL', float('nan')):.3f}",
            ])

    # ---------------- markdown table ----------------
    print()
    print("=" * 78)
    print("FID + LPIPS results (region-cropped, torchvision InceptionV3)")
    print("=" * 78)
    print()
    header = (f"| {'Type':<18} | {'N':>5} | {'LPIPS mean':>10} | "
              f"{'LPIPS med':>9} | {'LPIPS std':>9} | {'FID':>8} |")
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for ann_type in sorted(lpips_by_type.keys()):
        scores = lpips_by_type[ann_type]
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        fid_val = fid_per_type.get(ann_type, float("nan"))
        print(
            f"| {ann_type:<18} | {len(scores):>5} | "
            f"{statistics.mean(scores):>10.4f} | "
            f"{statistics.median(scores):>9.4f} | "
            f"{std:>9.4f} | {fid_val:>8.3f} |"
        )
    if all_lpips:
        std = statistics.stdev(all_lpips) if len(all_lpips) > 1 else 0.0
        print(
            f"| {'OVERALL':<18} | {len(all_lpips):>5} | "
            f"{statistics.mean(all_lpips):>10.4f} | "
            f"{statistics.median(all_lpips):>9.4f} | "
            f"{std:>9.4f} | {fid_per_type['OVERALL']:>8.3f} |"
        )
    print()
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
