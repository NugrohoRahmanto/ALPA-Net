#!/bin/bash

set -euo pipefail

PROJECT_DIR="/home/nugee/code-program/code-thesis/hibah/myocardial-infarction-localization"
cd "$PROJECT_DIR"

LOG_FILE="${LOG_FILE:-output_revise_plan.log}"
PID_FILE="${PID_FILE:-revise_plan_pipeline.pid}"
NOTEBOOK_ROOT="notebook_revise_plan"

PRIORITIES=(
  "priority_01_temporal_context"
  "priority_02_learnable_priors"
  "priority_03_mi_specific_features"
  "priority_04_spatial_anatomical_encoding"
  "priority_05_class_imbalance"
  "priority_06_interpretability_validation"
)

run_notebook() {
  local notebook_path="$1"
  local output_name="$2"
  local output_dir="$3"

  echo ""
  echo "Running: $notebook_path"
  echo "Output : $output_dir/$output_name"
  echo "Time   : $(date)"

  .venv/bin/jupyter nbconvert \
    --to notebook \
    --execute "$notebook_path" \
    --ExecutePreprocessor.timeout=-1 \
    --output "$output_name" \
    --output-dir "$output_dir"

  echo "Finished: $notebook_path"
  echo "Time    : $(date)"
}

echo $$ > "$PID_FILE"

echo "============================================================"
echo "ALPA-Net Architecture Revision Plan Pipeline"
echo "Start time : $(date)"
echo "Project    : $PROJECT_DIR"
echo "PID file   : $PID_FILE"
echo "Log file   : $LOG_FILE"
echo "============================================================"

for priority in "${PRIORITIES[@]}"; do
  if [[ "${ONLY_PRIORITY:-}" != "" && "${ONLY_PRIORITY}" != "$priority" ]]; then
    continue
  fi

  priority_dir="$NOTEBOOK_ROOT/$priority"
  pretrain_nb="$priority_dir/pretrain_${priority}.ipynb"
  finetune_nb="$priority_dir/finetune_${priority}.ipynb"

  echo ""
  echo "============================================================"
  echo "Priority started: $priority"
  echo "============================================================"

  run_notebook "$pretrain_nb" "pretrain_${priority}_executed.ipynb" "$priority_dir"
  run_notebook "$finetune_nb" "finetune_${priority}_executed.ipynb" "$priority_dir"

  echo ""
  echo "Priority finished: $priority"
done

echo ""
echo "============================================================"
echo "ALPA-Net Architecture Revision Plan Pipeline Completed"
echo "Finish time: $(date)"
echo "============================================================"
