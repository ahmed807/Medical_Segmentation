#!/usr/bin/env bash
# =====================================================================
# run_evaluations_zeroshot.sh
# ---------------------------------------------------------------------
# Runs evaluate_boundary_f1.py on two zero-shot ablation runs:
#   1) PVS_ZERO_SHOT  — SAM3TrackerModel off-the-shelf (no fine-tuning),
#                       same geometric prompts as the fine-tuned baseline.
#   2) PCS_ZERO_SHOT  — Sam3Model off-the-shelf (text-prompted), with
#                       the operating concept text used during inference.
#
# Both runs are SEG-ONLY (no FLUX), so they produce _object_mask.png
# files but no _clean.png. Only IoU + Boundary F1 apply; FID and LPIPS
# do not, because there is no inpainted image to compare against.
#
# Output dirs land inside each run dir:
#   <run_dir>/eval_boundary_f1/{boundary_f1_per_sample.csv,
#                                boundary_f1_summary.csv}
#
# Edit the CONFIG block to match your paths, then:
#   chmod +x run_evaluations_zeroshot.sh
#   nohup ./run_evaluations_zeroshot.sh > evaluations_zeroshot.log 2>&1 &
#   disown
#   tail -f evaluations_zeroshot.log
# =====================================================================

set -uo pipefail   # not -e: we want both runs to attempt even if one fails

# ---------- CONFIG ----------
DATA_ROOT="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"
TEST_JSON="${DATA_ROOT}/test.json"

# Where the eval scripts live (SAM3Tracker root, per the screenshot).
SCRIPTS_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs"
EVAL_BF1="${SCRIPTS_DIR}/evaluate_boundary_f1.py"

# Zero-shot run directories.
PVS_DIR="${SCRIPTS_DIR}/PVS_ZERO_SHOT"
PCS_DIR="${SCRIPTS_DIR}/PCS_ZERO_SHOT"

# Boundary F1 distance tolerance (pixels). Match the fine-tuned eval.
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

# Path sanity check up front.
for p in "${DATA_ROOT}" "${TEST_JSON}" "${EVAL_BF1}" \
         "${PVS_DIR}" "${PCS_DIR}"; do
    if [[ ! -e "${p}" ]]; then
        log "ABORT: required path missing: ${p}"
        exit 1
    fi
done

log "###################################################"
log "# ZERO-SHOT EVALUATIONS DRIVER START"
log "# host:           $(hostname)"
log "# data_root:      ${DATA_ROOT}"
log "# test_json:      ${TEST_JSON}"
log "# PVS zero-shot:  ${PVS_DIR}"
log "# PCS zero-shot:  ${PCS_DIR}"
log "###################################################"

# =====================================================================
# Zero-shot PVS (SAM3TrackerModel off-the-shelf)
# =====================================================================
log ""
log "############ PVS ZERO-SHOT ############"

run_step "P1_pvs_bf1" \
    python3 "${EVAL_BF1}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${PVS_DIR}" \
        --out-dir      "${PVS_DIR}/eval_boundary_f1" \
        --tolerance-px "${BF1_TOL}"

# =====================================================================
# Zero-shot PCS (Sam3Model off-the-shelf)
# =====================================================================
log ""
log "############ PCS ZERO-SHOT ############"

run_step "P2_pcs_bf1" \
    python3 "${EVAL_BF1}" \
        --test-json    "${TEST_JSON}" \
        --data-root    "${DATA_ROOT}" \
        --results-dir  "${PCS_DIR}" \
        --out-dir      "${PCS_DIR}/eval_boundary_f1" \
        --tolerance-px "${BF1_TOL}"

# =====================================================================
log ""
log "###################################################"
log "# ZERO-SHOT EVALUATIONS FINISHED"
log "###################################################"
log ""
log "Result files to inspect:"
log "  PVS zero-shot:"
log "    ${PVS_DIR}/eval_boundary_f1/boundary_f1_summary.csv"
log "  PCS zero-shot:"
log "    ${PCS_DIR}/eval_boundary_f1/boundary_f1_summary.csv"
log ""
log "Markdown summary tables are at the END of this log."
log "Grep for '====' to find them."
