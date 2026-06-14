#!/bin/bash

set -e

cd /home/nugee/code-program/code-thesis/hibah/myocardial-infarction-localization

echo "====================================="
echo "ALPA-Net Notebook 2 to 7 Pipeline Started"
echo "Start Time: $(date)"
echo "====================================="


.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/1_data_prepa.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 1_data_prep_executed.ipynb \
  --output-dir notebook/executed


echo ""
echo "[1/6] PRETRAIN ALPA-NET START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/2_pretrain_alpanet.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 2_pretrain_alpanet_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[1/6] PRETRAIN ALPA-NET FINISHED"
echo ""

echo ""
echo "[2/6] FINETUNE ALPA-NET TRANSFER CV START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/3_finetune_alpanet_transfer_cv.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 3_finetune_alpanet_transfer_cv_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[2/6] FINETUNE ALPA-NET TRANSFER CV FINISHED"
echo ""

echo ""
echo "[3/6] PRETRAIN ALPA-NET NO ATTENTION START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/4_pretrain_alpanet_no_attention.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 4_pretrain_alpanet_no_attention_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[3/6] PRETRAIN ALPA-NET NO ATTENTION FINISHED"
echo ""

echo ""
echo "[4/6] FINETUNE ALPA-NET NO ATTENTION START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/5_finetune_alpanet_no_attention.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 5_finetune_alpanet_no_attention_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[4/6] FINETUNE ALPA-NET NO ATTENTION FINISHED"
echo ""

echo ""
echo "[5/6] PRETRAIN MODEL COMPARISON START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/6_pretrain_model_comparison.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 6_pretrain_model_comparison_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[5/6] PRETRAIN MODEL COMPARISON FINISHED"
echo ""

echo ""
echo "[6/6] FINETUNE MODEL COMPARISON START"
echo ""

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebook/7_finetune_model_comparison.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 7_finetune_model_comparison_executed.ipynb \
  --output-dir notebook/executed

echo ""
echo "[6/6] FINETUNE MODEL COMPARISON FINISHED"
echo ""

echo ""
echo "====================================="
echo "ALPA-Net Notebook 2 to 7 Pipeline Completed"
echo "Finish Time: $(date)"
echo "====================================="