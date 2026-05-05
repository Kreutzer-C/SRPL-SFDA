#!/bin/bash
# SRPL-SFDA v2: Full pipeline with MemProp-compatible UNet (31M)
# BTCV→CHAOST2 + CHAOST2→BTCV bidirectional
# Phase 1 → Phase 2 (MedSAM) → Step 4
set -e
ROOT=/opt/data/private/SRPL-SFDA
cd 
export PYTHONPATH=:
GPU=0

# MemProp source model paths
BTCV_SRC=/opt/data/private/MemProp-SFDA/results/ABDOMINAL/BTCV_to_BTCV/source_pretrain_UNet_1/checkpoints/best_checkpoint.pth
CH2_SRC=/opt/data/private/MemProp-SFDA/results/ABDOMINAL/CHAOST2_to_CHAOST2/source_pretrain_UNet_5/checkpoints/epoch_0_dice_0.7180.pth

LOG=/pipeline_v2.log
echo "===== SRPL-SFDA v2 Full Pipeline =====" | tee 
echo "Start: Tue May  5 14:09:22     2026" | tee -a 

# ============================================================
# Phase 1: Tri-branch Enhancement + Ensemble PL
# ============================================================
echo "" | tee -a 
echo "[Phase 1] BTCV→CHAOST2 train..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/1_1_intensity_enhancement.py     --source_domain BTCV --source_model      --output_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl --split train 2>&1 | tee -a 

echo "[Phase 1] BTCV→CHAOST2 val..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/1_1_intensity_enhancement.py     --source_domain BTCV --source_model      --output_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl --split val 2>&1 | tee -a 

echo "[Phase 1] CHAOST2→BTCV train..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/1_1_intensity_enhancement.py     --source_domain CHAOST2 --source_model      --output_dir data/ABDOMINAL/CHAOST22BTCV/enhanced_pl --split train 2>&1 | tee -a 

echo "[Phase 1] CHAOST2→BTCV val..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/1_1_intensity_enhancement.py     --source_domain CHAOST2 --source_model      --output_dir data/ABDOMINAL/CHAOST22BTCV/enhanced_pl --split val 2>&1 | tee -a 

# ============================================================
# Phase 2: MedSAM Bbox Refinement
# ============================================================
echo "" | tee -a 
echo "[Phase 2] BTCV→CHAOST2 train..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/2_medsam_bbox_refine.py     --source_domain BTCV     --pl_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl/train     --output_dir data/ABDOMINAL/BTCV2CHAOST2/medsam_pl/train     --split train 2>&1 | tee -a 

echo "[Phase 2] BTCV→CHAOST2 val..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/2_medsam_bbox_refine.py     --source_domain BTCV     --pl_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl/val     --output_dir data/ABDOMINAL/BTCV2CHAOST2/medsam_pl/val     --split val 2>&1 | tee -a 

echo "[Phase 2] CHAOST2→BTCV train..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/2_medsam_bbox_refine.py     --source_domain CHAOST2     --pl_dir data/ABDOMINAL/CHAOST22BTCV/enhanced_pl/train     --output_dir data/ABDOMINAL/CHAOST22BTCV/medsam_pl/train     --split train 2>&1 | tee -a 

echo "[Phase 2] CHAOST2→BTCV val..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/2_medsam_bbox_refine.py     --source_domain CHAOST2     --pl_dir data/ABDOMINAL/CHAOST22BTCV/enhanced_pl/val     --output_dir data/ABDOMINAL/CHAOST22BTCV/medsam_pl/val     --split val 2>&1 | tee -a 

# ============================================================
# Step 4: RPL + EM training
# ============================================================
echo "" | tee -a 
echo "[Step 4] BTCV→CHAOST2 with MedSAM PLs..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/4_train_RPL_add_EM.py     --source_domain BTCV --source_model      --exp BTCV2CHAOST2/step4_EM_medsam_v2 --max_iterations 3000 --lameta_fix 0.1     --pseudo_label_dir data/ABDOMINAL/BTCV2CHAOST2/medsam_pl/train 2>&1 | tee -a 

echo "[Step 4] CHAOST2→BTCV with MedSAM PLs..." | tee -a 
CUDA_VISIBLE_DEVICES= python3 train_code/abdominal/4_train_RPL_add_EM.py     --source_domain CHAOST2 --source_model      --exp CHAOST22BTCV/step4_EM_medsam_v2 --max_iterations 3000 --lameta_fix 0.1     --pseudo_label_dir data/ABDOMINAL/CHAOST22BTCV/medsam_pl/train 2>&1 | tee -a 

echo "" | tee -a 
echo "===== Pipeline Complete: Tue May  5 14:09:22     2026 =====" | tee -a 
