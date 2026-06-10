#!/bin/bash

set -e

cd /home/nugee/code-program/code-thesis/hibah/myocardial-infarction-localization

echo "====================================="
echo "ALPA-Net Final Pipeline Started"
echo "Start Time: $(date)"
echo "====================================="

echo ""
echo "[1/3] DATA PREPARATION START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/1.data_prep.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 1.data_prep_executed.ipynb \
  --output-dir notebook

echo ""
echo "[1/3] DATA PREPARATION FINISHED"
echo ""

echo ""
echo "[2/3] PRETRAIN START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/2.pretrain_alpanet.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 2.pretrain_alpanet_executed.ipynb \
  --output-dir notebook

echo ""
echo "[2/3] PRETRAIN FINISHED"
echo ""

echo ""
echo "[3/3] FINETUNE START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/3.finetune_alpanet_transfer_cv.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 3.finetune_alpanet_transfer_cv_executed.ipynb \
  --output-dir notebook

echo ""
echo "====================================="
echo "ALPA-Net Final Pipeline Completed"
echo "Finish Time: $(date)"
echo "====================================="
