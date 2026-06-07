#!/usr/bin/env bash
# =====================================================================
# run_evaluations_pipeline_b.sh
# ---------------------------------------------------------------------
# Pipeline B evaluation. ONLY FID + LPIPS apply:
#   - Pipeline B detections have no matching GT object mask, so IoU /
#     Boundary F1 against GT are undefined. evaluate_boundary_f1.py is
#     deliberately NOT run here.
#   - FID / LPIPS compare the predicted clean image against the original
#     clean reference, which every Pipeline B entry still carries.
#
# Two steps:
#   1. evaluate_fid_lpips_pipeline_b.py  -> crops + per-sample LPIPS
#      (Pipeline-B-aware: handles the <stem>_det<idx> entry-id naming)
#   2. compute_fid_from_crops.py         -> torchvision FID on the crops
#      (same script as Pipeline A, so the numbers are comparable)
#
# Edit CONFIG, then:
#   chmod +x run_evaluations_pipeline_b.sh
#   nohup ./run_evaluations_pipeline_b.sh > eval_pipeline_b.log 2>&1 &
#   disown
#   tail -f eval_pipeline_b.log
# =====================================================================

set -uo pipefail

# ---------- CONFIG ----------
SCRIPTS_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask/pipeline_b_default"
RUN_DIR="/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/runs/2026-04-19_16-03-50_v6_arrow_pca_axis_freeform_mask"

# Pipeline B inference outputs (the dir you gave).
PB_RESULTS="${RUN_DIR}/pipeline_b_default"

# The JSON that was fed to inference_pipeline_b.py for this run.
# This is the *default* config build. If you evaluated the tuned build
# instead, point this at pipeline_b_input_tuned.json.
PB_INPUT_JSON="/home/ahma/Grounded-SAM-2/pipeline_b_input_default.json"

# Evaluation scripts.
EVAL_FL_PB="${SCRIPTS_DIR}/evaluate_fid_lpips_pipeline_b.py"
COMPUTE_FID="${SCRIPTS_DIR}/compute_fid_from_crops.py"

OUT_DIR="${PB_RESULTS}/eval_fid_lpips"

# ---------- HELPERS ----------
log() { echo "[$(date -Iseconds)] $*"; }

run_step() {
    local name="$1"; shift
    log "==================================================="
    log ">>> START: ${name}"
    log "    cmd: $*"
    log "==================================================="
    local t0; t0=$(date +%s)
    if "$@"; then
        log "<<< DONE : ${name}  (elapsed $(( $(date +%s) - t0 )) s)"
    else
        log "!!! FAIL : ${name}  rc=$?"
        log "    stopping (step 2 depends on step 1's crops)"
        exit 1
    fi
}

for p in "${PB_RESULTS}" "${PB_INPUT_JSON}" "${EVAL_FL_PB}" \
         "${COMPUTE_FID}"; do
    if [[ ! -e "${p}" ]]; then
        log "ABORT: required path missing: ${p}"
        log "  (if PB_INPUT_JSON is wrong, set it to the JSON you"
        log "   actually fed to inference for pipeline_b_default)"
        exit 1
    fi
done

log "###################################################"
log "# PIPELINE B EVALUATION START"
log "# host:        $(hostname)"
log "# gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo cpu)"
log "# results dir: ${PB_RESULTS}"
log "# input json:  ${PB_INPUT_JSON}"
log "# out dir:     ${OUT_DIR}"
log "###################################################"

run_step "PB1_lpips_crops" \
    python3 "${EVAL_FL_PB}" \
        --input-json   "${PB_INPUT_JSON}" \
        --results-dir  "${PB_RESULTS}" \
        --out-dir      "${OUT_DIR}" \
        --crop-size    256 \
        --margin-px    16 \
        --orig-root-replace "/home/ahma/Medical_Segmentation/FineTuning_SAM3/sam_finetuning_dataset:/home/ahma/Medical_Segmentation/FineTuning_SAM3/SAM3Tracker/sam_finetuning_dataset"

run_step "PB2_fid" \
    python3 "${COMPUTE_FID}" \
        --eval-dir "${OUT_DIR}"

log ""
log "###################################################"
log "# PIPELINE B EVALUATION FINISHED"
log "###################################################"
log "Result files:"
log "  ${OUT_DIR}/lpips_per_sample.csv"
log "  ${OUT_DIR}/fid_lpips_summary.csv   <- send this back"
log ""
log "NOTE: IoU / Boundary F1 are intentionally absent for Pipeline B."
log "Pipeline B detections have no matching GT object mask, so"
log "segmentation-vs-GT metrics are undefined. Detection quality is"
log "reported separately via the threshold-sweep precision/recall"
log "numbers already in hand (chapter 6 detection section)."
