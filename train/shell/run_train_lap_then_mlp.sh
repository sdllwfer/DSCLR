#!/bin/bash

# ============================================================
# DeIR 分阶段解耦训练脚本
# 先训练 LAP（冻结编码器），再训练 MLP（冻结 LAP）
# ============================================================

eval "$(conda shell.bash hook)"
conda activate dsclr
cd /home/luwa/Documents/DSCLR

# ============================================================
# 配置参数
# ============================================================

# 模型选择: "bge", "mistral" 或 "repllama"
ENCODER_TYPE="repllama"

# GPU 设备
GPU_ID=1

# 输出目录（默认）
OUTPUT_DIR="train/output/Hybrid"

# 自定义输出目录（可选，如果设置则覆盖上面的 OUTPUT_DIR）
# 例如：CUSTOM_OUTPUT_DIR="/path/to/my/output"
CUSTOM_OUTPUT_DIR="/home/luwa/Documents/DSCLR/train/output/Hybrid/lap_then_mlp/03.30-傍晚-repllama-防崩塌修复"

# 实验备注（可选，会记录到训练配置文件中）
EXPERIMENT_NOTE="LAP防崩塌修复: λ_push=1.4/λ_pull=0.6+正交正则化+随机正交锚点+Margin调整(Push<0.05/Pull>0.3)"

# LAP 训练参数
LAP_EPOCHS=8
LAP_LR=5e-5

# MLP 训练参数
MLP_EPOCHS=10
MLP_LR=1e-4

# 数据参数
BATCH_SIZE=256
ENCODE_BATCH_SIZE=128
NUM_NEG=15
VAL_RATIO=0.1
PATIENCE=5

# 数据目录
DATA_DIR="dataset/FollowIR_train"

# ============================================================
# 根据编码器类型自动设置参数
# ============================================================

if [ "$ENCODER_TYPE" = "mistral" ]; then
    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    MODEL_SHORT="e5-mistral-7b"
    EMBED_DIM=4096
    ENCODE_BATCH_SIZE=16
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    MODEL_SHORT="bge-large-en"
    EMBED_DIM=1024
    ENCODE_BATCH_SIZE=32
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    MODEL_SHORT="repllama-v1-7b"
    EMBED_DIM=4096
    ENCODE_BATCH_SIZE=16
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

# 缓存路径
# LAP 阶段使用难负样本数据集
# MLP 阶段使用原始数据集（混合样本）
if [ "$ENCODER_TYPE" = "mistral" ]; then
    LAP_CACHE_PATH="dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_hard_negatives_embeddings.pt"
    MLP_CACHE_PATH="dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt"
    CACHE_PATH="$LAP_CACHE_PATH"  # 默认使用 LAP 的缓存
elif [ "$ENCODER_TYPE" = "bge" ]; then
    LAP_CACHE_PATH="dataset/FollowIR_train/embeddings/bge-large-en/dsclr_train_hard_negatives_embeddings.pt"
    MLP_CACHE_PATH="dataset/FollowIR_train/embeddings/bge-large-en/dsclr_train_embeddings_bge-large-en.pt"
    CACHE_PATH="$LAP_CACHE_PATH"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    LAP_CACHE_PATH="dataset/FollowIR_train/embeddings/repllama-v1-7b/dsclr_train_hard_negatives_embeddings.pt"
    MLP_CACHE_PATH="dataset/FollowIR_train/embeddings/repllama-v1-7b/dsclr_train_embeddings_repllama-v1-7b.pt"
    CACHE_PATH="$LAP_CACHE_PATH"
fi

TIMESTAMP=$(date +%m.%d-%H%M)

# 如果设置了自定义输出目录，则使用它（作为最终完整路径）
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
    # 标记为完整路径，Python 不会再拼接子目录
    USE_CUSTOM_OUTPUT_AS_FINAL="True"
else
    USE_CUSTOM_OUTPUT_AS_FINAL="False"
fi

# ============================================================
# 打印配置信息
# ============================================================

echo "============================================================"
echo "DeIR 分阶段解耦训练配置"
echo "============================================================"
echo "编码器: $MODEL_NAME"
echo "嵌入维度: $EMBED_DIM"
echo "GPU: $GPU_ID"
echo "训练策略: 分阶段解耦训练 (LAP -> MLP)"
echo "  - LAP 阶段: $LAP_EPOCHS epochs, LR=$LAP_LR"
echo "    使用难负样本: $LAP_CACHE_PATH"
echo "  - MLP 阶段: $MLP_EPOCHS epochs, LR=$MLP_LR"
echo "    使用混合样本: $MLP_CACHE_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

# ============================================================
# 检查数据
# ============================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "⚠️  数据目录不存在: $DATA_DIR"
    exit 1
fi
echo "使用原始数据集: $DATA_DIR"

# ============================================================
# Wandb 配置
# ============================================================
USE_WANDB="False"

# ============================================================
# 开始训练
# ============================================================

echo ""
echo "🚀 开始分阶段训练..."
echo ""

# 保存训练配置到文件
CONFIG_FILE="$OUTPUT_DIR/training_config.txt"
mkdir -p "$OUTPUT_DIR"
cat > "$CONFIG_FILE" << EOF
========================================
训练配置参数
========================================
生成时间: $(date '+%Y-%m-%d %H:%M:%S')

[实验备注]
$EXPERIMENT_NOTE

[基础配置]
编码器类型: $ENCODER_TYPE
模型名称: $MODEL_NAME
GPU ID: $GPU_ID
输出目录: $OUTPUT_DIR

[LAP 阶段]
Epochs: $LAP_EPOCHS
Learning Rate: $LAP_LR
Cache Path: $LAP_CACHE_PATH

[MLP 阶段]
Epochs: $MLP_EPOCHS
Learning Rate: $MLP_LR
Cache Path: $MLP_CACHE_PATH

[数据参数]
Batch Size: $BATCH_SIZE
Encode Batch Size: $ENCODE_BATCH_SIZE
Num Negatives: $NUM_NEG
Validation Ratio: $VAL_RATIO
Patience: $PATIENCE

[其他]
Use WandB: $USE_WANDB
========================================
EOF
echo "✅ 训练配置已保存: $CONFIG_FILE"
echo ""

python model/train_hybrid.py \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --encode_batch_size $ENCODE_BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --output_dir_is_final $USE_CUSTOM_OUTPUT_AS_FINAL \
    --data_dir "$DATA_DIR" \
    --cache_path "$CACHE_PATH" \
    --mlp_cache_path "$MLP_CACHE_PATH" \
    --lr $MLP_LR \
    --lap_lr $LAP_LR \
    --lap_epochs $LAP_EPOCHS \
    --epochs $MLP_EPOCHS \
    --num_neg $NUM_NEG \
    --val_ratio $VAL_RATIO \
    --patience $PATIENCE \
    --embed_dim $EMBED_DIM \
    --model_type $ENCODER_TYPE \
    --use_lap True \
    --use_mlp True \
    --lap_then_mlp True \
    --use_wandb $USE_WANDB

echo ""
echo "✅ 训练完成!"