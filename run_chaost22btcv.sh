#!/bin/bash
# SRPL-SFDA: ABDOMINAL (CHAOST2 → BTCV) — MR→CT Adaptation
#
# Reverse direction: train on CHAOST2 (MRI), adapt to BTCV (CT).
# Same pipeline as BTCV→CHAOST2 but with swapped source/target domains.
set -e
export CUDA_VISIBLE_DEVICES=0
ROOT=/opt/data/private/SRPL-SFDA
cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

DIRECTION=CHAOST22BTCV
SOURCE_DOMAIN=CHAOST2
ENHANCED_DIR=$ROOT/data/ABDOMINAL/$DIRECTION/enhanced_pl
SRC_MODEL_DIR=$ROOT/results/$DIRECTION/source_model
SRC_MODEL=$SRC_MODEL_DIR/best_model.pth

echo "========================================="
echo "SRPL-SFDA: ABDOMINAL ($DIRECTION)"
echo "========================================="

# Phase 0: Source Model Training
echo "[Phase 0] Training source model on $SOURCE_DOMAIN..."
python train_code/abdominal/0_source_train.py \
    --source_domain $SOURCE_DOMAIN \
    --batch_size 32 \
    --max_epochs 100

# Phase 1: Test-Time Tri-branch Intensity Enhancement + Ensemble Pseudo-Labels
echo "[Phase 1] Tri-branch intensity enhancement + ensemble PL generation..."
python train_code/abdominal/1_1_intensity_enhancement.py \
    --source_domain $SOURCE_DOMAIN \
    --source_model $SRC_MODEL \
    --output_dir $ENHANCED_DIR \
    --split train
python train_code/abdominal/1_1_intensity_enhancement.py \
    --source_domain $SOURCE_DOMAIN \
    --source_model $SRC_MODEL \
    --output_dir $ENHANCED_DIR \
    --split val

# Phase 2: SFDA Step 3 — RPL Selection + Fine-tune
echo "[Phase 2] SFDA Step 3: RPL selection + fine-tune..."
python train_code/abdominal/3_train_RPL_selectRPL.py \
    --source_domain $SOURCE_DOMAIN \
    --source_model $SRC_MODEL \
    --exp $DIRECTION/step3_RPL \
    --max_iterations 3000 \
    --pseudo_label_dir $ENHANCED_DIR/train

# Phase 3: SFDA Step 4 — RPL + EM (full SRPL-SFDA method)
echo "[Phase 3] SFDA Step 4: RPL + EM (full method)..."
python train_code/abdominal/4_train_RPL_add_EM.py \
    --source_domain $SOURCE_DOMAIN \
    --source_model $SRC_MODEL \
    --exp $DIRECTION/step4_EM \
    --max_iterations 3000 \
    --lameta_fix 0.1 \
    --pseudo_label_dir $ENHANCED_DIR/train

# Evaluation
echo "========================================="
echo "Evaluating..."
echo "========================================="
echo ""
echo "--- Source-only ---"
python train_code/abdominal/test_metrics.py \
    --source_domain $SOURCE_DOMAIN \
    --model_path $SRC_MODEL \
    --domain target --split test
echo ""
echo "--- SFDA Step 3 ---"
python train_code/abdominal/test_metrics.py \
    --source_domain $SOURCE_DOMAIN \
    --model_path results/$DIRECTION/step3_RPL/best_model.pth \
    --domain target --split test
echo ""
echo "--- SFDA Step 4 (Full) ---"
python train_code/abdominal/test_metrics.py \
    --source_domain $SOURCE_DOMAIN \
    --model_path results/$DIRECTION/step4_EM/best_model.pth \
    --domain target --split test
echo ""
echo "Done!"
