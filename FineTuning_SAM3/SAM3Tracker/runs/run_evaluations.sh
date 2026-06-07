#!/usr/bin/env bash
# =====================================================================
# run_evaluations.sh
# ---------------------------------------------------------------------
# Runs the three evaluation scripts on two run directories:
#   1) Pipeline A v2 (production v6 checkpoint, full test set 8,288)
#   2) No-IoU ablation (no_iou_best.pth, same test set)
#
# For each run dir, in order:
#   evaluate_boundary_f1.py      -> IoU + BF1 per type
#   evaluate_fid_lpips.py        -> crops + per-sample LPIPS  (--skip-fid)
#   compute_fid_from_crops.py    -> torchvision FID on the saved crops
#
# Outputs land inside each run dir:
#   <run_dir>/eval_boundary_f1/{per_sample.csv, summary.csv}
#   <run_dir>/eval_fid_lpips/{crops/, lpips_per_sample.csv, fid_lpips_summary.csv}
#
# Edit the CONFIG block to match your paths, then:
#   chmod +x run_evaluations.sh
#   nohup ./run_evaluations.sh > evaluations.log 2>&1 &
#   disown
#   tail -f evaluations.log
# =====================================================================

set -uo pipefail   # not -e: we want all six steps to attempt even if one fails

# ---------- CONFIG ----------
DATA_ROOT="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"
TEST_JSON="${DATA_ROOT}/test.json"

# Where the three evaluation scripts live. From screenshot 2 they sit
# next to the inference_results_full_v2 dir under the v6 run.
SCRIPTS_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs"
IOU_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask"

EVAL_BF1="${SCRIPTS_DIR}/evaluate_boundary_f1.py"
EVAL_FID_LPIPS="${SCRIPTS_DIR}/evaluate_fid_lpips.py"
COMPUTE_FID="${SCRIPTS_DIR}/compute_fid_from_crops.py"

    python3 "${EVAL_BF1}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${PIPELINE_A_RESULTS}" \
        --out-dir      "${PIPELINE_A_DIR}/eval_boundary_f1" \
        --tolerance-px "${BF1_TOL}"
TEST: /home/ahmed/Medical_Segmentation/SAM3Tracker/sam_finetuning_dataset_generic/test.json
DATA: /home/ahmed/Medical_Segmentation/SAM3Tracker/sam_finetuning_dataset_generic
RESULTS: /home/ahmed/Medical_Segmentation/SAM3Tracker/runs/2026-05-17_12-37-43_v6_arrow_pca_axis_freeform_mask/inference_results_generic_sam_only
OUT: /home/ahmed/Medical_Segmentation/SAM3Tracker/runs/2026-05-17_12-37-43_v6_arrow_pca_axis_freeform_mask/eval_boundary_f1
TOLERANCE: 2
# Two run directories and the inference output sub-dir inside each.
PIPELINE_A_DIR="${IOU_DIR}"
PIPELINE_A_RESULTS="${IOU_DIR}/inference_results_full_v2"

NO_IOU_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-05-07_15-24-52_v6_ablation_no_iou_loss"
NO_IOU_RESULTS="${NO_IOU_DIR}/inference_results_full"

# Boundary F1 distance tolerance (pixels).
BF1_TOL=2

# ---------- HELPERS ----------
log() {
    echo "[$(date -Iseconds)] $*"
}

run_step() {
    local name="$1"
    shift
    log "==================================================="
    log ">>> START: ${name}"
    log "    cmd: $*"
    log "==================================================="
    local t0; t0=$(date +%s)
    if "$@"; then
        local t1; t1=$(date +%s)
        log "<<< DONE : ${name}  (elapsed $((t1 - t0)) s)"
    else
        local rc=$?
        log "!!! FAIL : ${name}  rc=${rc}"
        log "    continuing to next step"
    fi
}

# Path sanity check up front — don't burn time discovering a typo at hour 1.
for p in "${DATA_ROOT}" "${TEST_JSON}" "${EVAL_BF1}" "${EVAL_FID_LPIPS}" \
         "${COMPUTE_FID}" "${PIPELINE_A_RESULTS}" "${NO_IOU_RESULTS}"; do
    if [[ ! -e "${p}" ]]; then
        log "ABORT: required path missing: ${p}"
        exit 1
    fi
done

log "###################################################"
log "# EVALUATIONS DRIVER START"
log "# host:           $(hostname)"
log "# gpu (if any):   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'cpu-only')"
log "# data_root:      ${DATA_ROOT}"
log "# test_json:      ${TEST_JSON}"
log "# pipeline A:     ${PIPELINE_A_RESULTS}"
log "# no-IoU:         ${NO_IOU_RESULTS}"
log "###################################################"

# =====================================================================
# Pipeline A (production v6, best_by_iou.pth)
# =====================================================================
log ""
log "############ PIPELINE A ############"

run_step "A1_pipelineA_bf1" \
    python3 "${EVAL_BF1}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${PIPELINE_A_RESULTS}" \
        --out-dir      "${PIPELINE_A_DIR}/eval_boundary_f1" \
        --tolerance-px "${BF1_TOL}"

run_step "A2_pipelineA_lpips_crops" \
    python3 "${EVAL_FID_LPIPS}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${PIPELINE_A_RESULTS}" \
        --out-dir      "${PIPELINE_A_DIR}/eval_fid_lpips" \
        --skip-fid

run_step "A3_pipelineA_fid" \
    python3 "${COMPUTE_FID}" \
        --eval-dir "${PIPELINE_A_DIR}/eval_fid_lpips"

# =====================================================================
# No-IoU ablation (v6 trained without IoU-MSE loss term)
# =====================================================================
log ""
log "############ NO-IOU ABLATION ############"

run_step "B1_noiou_bf1" \
    python3 "${EVAL_BF1}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${NO_IOU_RESULTS}" \
        --out-dir      "${NO_IOU_DIR}/eval_boundary_f1" \
        --tolerance-px "${BF1_TOL}"

run_step "B2_noiou_lpips_crops" \
    python3 "${EVAL_FID_LPIPS}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${NO_IOU_RESULTS}" \
        --out-dir      "${NO_IOU_DIR}/eval_fid_lpips" \
        --skip-fid

run_step "B3_noiou_fid" \
    python3 "${COMPUTE_FID}" \
        --eval-dir "${NO_IOU_DIR}/eval_fid_lpips"

# =====================================================================
log ""
log "###################################################"
log "# ALL EVALUATIONS FINISHED"
log "###################################################"
log ""
log "Result files to inspect:"
log "  Pipeline A:"
log "    ${PIPELINE_A_DIR}/eval_boundary_f1/boundary_f1_summary.csv"
log "    ${PIPELINE_A_DIR}/eval_fid_lpips/fid_lpips_summary.csv"
log "  No-IoU ablation:"
log "    ${NO_IOU_DIR}/eval_boundary_f1/boundary_f1_summary.csv"
log "    ${NO_IOU_DIR}/eval_fid_lpips/fid_lpips_summary.csv"
log ""
log "The markdown summary tables are at the END of this log — grep the"
log "log for '====' to find them."
