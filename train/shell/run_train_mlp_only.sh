#!/bin/bash

# ============================================================
# DeIR MLP 阶段单独训练脚本
# 从已有的 LAP checkpoint 开始，只训练 MLP 阶段
# ============================================================

eval "$(conda shell.bash hook)"
conda activate dsclr
cd /home/luwa/Documents/DSCLR

# ============================================================
# 配置参数
# ============================================================

# LAP checkpoint 路径（必须设置）
# 例如：LAP_CHECKPOINT="train/output/Hybrid/lap_then_mlp/03.28-1121-repllama/lap_phase/deir_best_lap_phase.pt"
LAP_CHECKPOINT="train/output/Hybrid/lap_then_mlp/03.28-1145-repllama-LAPepoch5/lap_phase/deir_best_lap_phase.pt"

# 模型选择: "bge", "mistral" 或 "repllama"
ENCODER_TYPE="repllama"

# GPU 设备
GPU_ID=2

# 输出目录（默认）
OUTPUT_DIR="train/output/Hybrid"

# 自定义输出目录（可选，如果设置则覆盖上面的 OUTPUT_DIR）
CUSTOM_OUTPUT_DIR="/home/luwa/Documents/DSCLR/train/output/Hybrid/lap_then_mlp/03.28-2200-repllama-ShieldLoss重构版"

# 实验备注（可选，会记录到训练配置文件中）
EXPERIMENT_NOTE="使用DeIRFriendlyFireLoss重构版：引入Shield Loss强制提升tau，解决正样本误伤问题。shield_epsilon=0.05, shield_weight=2.0"

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
if [ "$ENCODER_TYPE" = "mistral" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/bge-large-en/dsclr_train_embeddings_bge-large-en.pt"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/repllama-v1-7b/dsclr_train_embeddings_repllama-v1-7b.pt"
fi

TIMESTAMP=$(date +%m.%d-%H%M)

# 如果设置了自定义输出目录，则使用它（作为最终完整路径）
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
    USE_CUSTOM_OUTPUT_AS_FINAL="True"
else
    USE_CUSTOM_OUTPUT_AS_FINAL="False"
fi

# ============================================================
# 打印配置信息
# ============================================================

echo "============================================================"
echo "DeIR MLP 阶段单独训练配置"
echo "============================================================"
echo "编码器: $MODEL_NAME"
echo "嵌入维度: $EMBED_DIM"
echo "GPU: $GPU_ID"
echo "训练策略: 只训练 MLP 阶段（LAP 已冻结）"
echo "LAP Checkpoint: $LAP_CHECKPOINT"
echo "  - MLP 阶段: $MLP_EPOCHS epochs, LR=$MLP_LR"
echo "Batch Size: $BATCH_SIZE"
echo "Encode Batch Size: $ENCODE_BATCH_SIZE"
echo "Num Negatives: $NUM_NEG"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

# 检查 LAP checkpoint 是否存在
if [ ! -f "$LAP_CHECKPOINT" ]; then
    echo "❌ 错误: LAP checkpoint 不存在: $LAP_CHECKPOINT"
    exit 1
fi
echo "✅ LAP checkpoint 已确认存在"

# ============================================================
# 开始训练
# ============================================================

echo ""
echo "🚀 开始 MLP 阶段训练..."
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
Checkpoint: $LAP_CHECKPOINT
状态: 已冻结，从 checkpoint 加载

[MLP 阶段]
Epochs: $MLP_EPOCHS
Learning Rate: $MLP_LR

[数据参数]
Batch Size: $BATCH_SIZE
Encode Batch Size: $ENCODE_BATCH_SIZE
Num Negatives: $NUM_NEG
Validation Ratio: $VAL_RATIO
Patience: $PATIENCE
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
    --lr $MLP_LR \
    --epochs $MLP_EPOCHS \
    --num_neg $NUM_NEG \
    --val_ratio $VAL_RATIO \
    --patience $PATIENCE \
    --embed_dim $EMBED_DIM \
    --model_type $ENCODER_TYPE \
    --use_lap True \
    --use_mlp True \
    --lap_then_mlp True \
    --lap_checkpoint "$LAP_CHECKPOINT" \
    --lap_epochs 0

echo ""
echo "✅ MLP 阶段训练完成！"
echo ""
