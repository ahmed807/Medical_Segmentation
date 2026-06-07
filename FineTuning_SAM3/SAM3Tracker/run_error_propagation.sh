#!/usr/bin/env bash
# =====================================================================
# run_error_propagation.sh
# ---------------------------------------------------------------------
# Per-detection error-propagation analysis for §6.5.2.
#
# Pipeline:
#   1. Compute matched IoU for every Pipeline B detection by comparing
#      its predicted annotation mask against every GT annotation mask
#      on the same source image (best IoU wins).
#   2. Join with the per-detection LPIPS already produced by the
#      Pipeline B evaluator.
#   3. Bucket detections by matched IoU into four tiers (zero / low /
#      mid / high). Produce per-bucket per-class LPIPS stats and
#      per-class Pearson + Spearman correlations between IoU and LPIPS.
#   4. Symlink the existing FID crops into per-bucket subdirectories.
#   5. Run compute_fid_from_crops.py on each bucket to get per-bucket FID.
#
# Total wall-clock: ~10 minutes CPU + ~4 x 2 min GPU for the FID passes.
#
# Outputs land under:
#   <PB_RESULTS>/error_propagation/
#       detection_iou_per_sample.csv
#       bucketed_stats.csv
#       correlations.csv
#       bucket_crops/<bucket>/crops/<type>/{predicted,original}/
#       bucket_crops/<bucket>/lpips_per_sample.csv
#       bucket_crops/<bucket>/fid_lpips_summary.csv     # after compute_fid
#
# Edit CONFIG, then:
#   chmod +x run_error_propagation.sh
#   nohup ./run_error_propagation.sh > error_propagation.log 2>&1 &
#   disown
#   tail -f error_propagation.log
# =====================================================================

set -uo pipefail

SCRIPTS_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker"
RUN_DIR="${SCRIPTS_DIR}/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask"
PB_RESULTS="${RUN_DIR}/pipeline_b_default"

PB_INPUT_JSON="/home/ahma/Grounded-SAM-2/pipeline_b_input_default.json"
SAM3_TEST_JSON="${SCRIPTS_DIR}/sam_finetuning_dataset/test.json"
SAM3_ROOT="${SCRIPTS_DIR}/sam_finetuning_dataset"

LPIPS_CSV="${PB_RESULTS}/eval_fid_lpips/lpips_per_sample.csv"
CROPS_DIR="${PB_RESULTS}/eval_fid_lpips/crops"

ANALYSIS_PY="${SCRIPTS_DIR}/analyze_pipeline_b_buckets.py"
COMPUTE_FID="${SCRIPTS_DIR}/runs/compute_fid_from_crops.py"

OUT_DIR="${PB_RESULTS}/error_propagation"

log() { echo "[$(date -Iseconds)] $*"; }

for p in "${PB_INPUT_JSON}" "${SAM3_TEST_JSON}" "${LPIPS_CSV}" \
         "${CROPS_DIR}" "${ANALYSIS_PY}" "${COMPUTE_FID}"; do
    if [[ ! -e "${p}" ]]; then
        log "ABORT: required path missing: ${p}"
        exit 1
    fi
done

log "###################################################"
log "# PIPELINE B ERROR PROPAGATION  --  bucketed analysis"
log "# host       : $(hostname)"
log "# input json : ${PB_INPUT_JSON}"
log "# lpips csv  : ${LPIPS_CSV}"
log "# out dir    : ${OUT_DIR}"
log "###################################################"

# ----- step 1: compute matched IoU, bucket, correlate, symlink crops -----
log ""
log "=== STEP 1: detection IoU + bucketing + symlinks ==="
t0=$(date +%s)
python3 "${ANALYSIS_PY}" \
    --input-json   "${PB_INPUT_JSON}" \
    --test-json    "${SAM3_TEST_JSON}" \
    --sam3-root    "${SAM3_ROOT}" \
    --lpips-csv    "${LPIPS_CSV}" \
    --crops-dir    "${CROPS_DIR}" \
    --orig-root-replace "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset:/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset" \
    --out-dir      "${OUT_DIR}"
rc=$?
log "step 1 rc=${rc}, elapsed $(( $(date +%s) - t0 )) s"
[[ ${rc} -ne 0 ]] && exit ${rc}

# ----- step 2: per-bucket FID -----
log ""
log "=== STEP 2: per-bucket FID via compute_fid_from_crops.py ==="
for bucket_dir in "${OUT_DIR}"/bucket_crops/*; do
    [[ -d "${bucket_dir}/crops" ]] || continue
    b=$(basename "${bucket_dir}")
    log ""
    log ">>> bucket: ${b}"
    t0=$(date +%s)
    python3 "${COMPUTE_FID}" --eval-dir "${bucket_dir}" || \
        log "    (compute_fid failed for ${b}; continuing)"
    log "<<< ${b} elapsed $(( $(date +%s) - t0 )) s"
done

log ""
log "###################################################"
log "# DONE"
log "#"
log "# Results to send back for chapter 6.5.2:"
log "#   ${OUT_DIR}/bucketed_stats.csv"
log "#   ${OUT_DIR}/correlations.csv"
log "#   ${OUT_DIR}/bucket_crops/<bucket>/fid_lpips_summary.csv  (per bucket)"
log "###################################################"
