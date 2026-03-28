#!/usr/bin/env bash
# ============================================================
#  SRPL-SFDA  ——  CHAOS 腹部器官分割训练流水线
#  依赖数据格式：MemProp-SFDA processed_DDFP  (NPZ 切片)
#
#  使用方法：
#    cd /workspace/SRPL-SFDA
#    bash train_code/chaos/train.sh
# ============================================================

set -e  # 出错即停

# -------- 用户配置 (修改这里) --------
DATA_DIR="./datasets/ABDOMINAL/processed_DDFP"   # processed_DDFP 根目录 (含 metadata.json)
DOMAIN="CHAOST2"                                  # 目标域名称
SOURCE_MODEL=""                                   # 源域预训练模型路径 (.pth)
SAM_CHECKPOINT=""                                 # MedSAM 检查点路径 (.pth)
GPU="0"
NUM_CLASSES=5
PATCH_SIZE=256
# -------- 超参数 --------
T_FIX=0.5              # Stage 3: 可靠性阈值乘数
MAX_ITER_3=3000        # Stage 3 迭代次数
LR_3=1e-4
THRESHOLD_4=0.6        # Stage 4: 可靠性阈值
LAMBDA_4=0.1           # Stage 4: 熵最小化损失权重
MAX_ITER_4=3000
LR_4=1e-3
# ------------------------------------

export PYTHONPATH="$PYTHONPATH:$(pwd)"
SCRIPT_DIR="$(dirname "$0")"

check_arg() {
    if [ -z "$1" ]; then
        echo "[ERROR] $2 is not set. Please edit train.sh."
        exit 1
    fi
}
check_arg "$DATA_DIR"    "DATA_DIR"
check_arg "$SOURCE_MODEL" "SOURCE_MODEL"
check_arg "$SAM_CHECKPOINT" "SAM_CHECKPOINT"

echo "============================================================"
echo "  SRPL-SFDA CHAOS Pipeline"
echo "  DATA_DIR      : $DATA_DIR"
echo "  DOMAIN        : $DOMAIN"
echo "  SOURCE_MODEL  : $SOURCE_MODEL"
echo "============================================================"

# (a) ---- Stage 1: Tri-branch Pseudo-Label Generation ----
echo "[Stage 1-1] Histogram equalization + source model inference..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/1_1_image_trans_equal.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --source_model "$SOURCE_MODEL" --num_classes $NUM_CLASSES \
    --patch_size $PATCH_SIZE --gpu "$GPU"

echo "[Stage 1-2] Gamma-dark (γ=0.6) + source model inference..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/1_2_image_trans_rD.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --source_model "$SOURCE_MODEL" --num_classes $NUM_CLASSES \
    --patch_size $PATCH_SIZE --gpu "$GPU"

echo "[Stage 1-3] Gamma-bright (γ=1.4) + source model inference..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/1_3_image_trans_rS.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --source_model "$SOURCE_MODEL" --num_classes $NUM_CLASSES \
    --patch_size $PATCH_SIZE --gpu "$GPU"

echo "[Stage 1-4] Average 3 predictions → pseudo-label + uncertainty map..."
python $SCRIPT_DIR/1_4_average_pl_uncertainty.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" --num_classes $NUM_CLASSES

# (b) ---- Stage 2: SAM Pseudo-Label Refinement ----
echo "[Stage 2-2] Create 3-channel RGB PNGs for SAM input..."
python $SCRIPT_DIR/2_2_concat_image.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN"

echo "[Stage 2-3] MedSAM per-class BBox refinement..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/2_3_Med_SAM_bbox_seg.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --sam_checkpoint "$SAM_CHECKPOINT" --gpu "$GPU"

# (c) ---- Stage 3: RPL Reliable Pseudo-Label Fine-tuning ----
echo "[Stage 3] RPL fine-tuning..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/3_train_RPL_selectRPL_fine_tune.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --source_model "$SOURCE_MODEL" \
    --num_classes $NUM_CLASSES \
    --max_iterations $MAX_ITER_3 \
    --T_fix $T_FIX \
    --base_lr $LR_3 \
    --exp chaos_rpl_ft \
    --gpu "$GPU"

# After stage 3, use the best model for stage 4
STAGE3_BEST="../../results/chaos_rpl/chaos_rpl_ft/${DOMAIN}/iters${MAX_ITER_3}_lr${LR_3}_T${T_FIX}/unet2d_best_model.pth"

# (d) ---- Stage 4: RPL + Entropy Minimization ----
echo "[Stage 4] RPL + Entropy Minimization fine-tuning..."
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/4_train_RPL_selectRPL_add_EM_fine_tune.py \
    --data_dir "$DATA_DIR" --domain "$DOMAIN" \
    --source_model "$STAGE3_BEST" \
    --num_classes $NUM_CLASSES \
    --max_iterations $MAX_ITER_4 \
    --threshold $THRESHOLD_4 \
    --lameta_fix $LAMBDA_4 \
    --base_lr $LR_4 \
    --exp chaos_rpl_em \
    --gpu "$GPU"

echo "============================================================"
echo "  SRPL-SFDA CHAOS pipeline completed!"
echo "  Final model: results/chaos_rpl_em/chaos_rpl_em/${DOMAIN}/"
echo "============================================================"
