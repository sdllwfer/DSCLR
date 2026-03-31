#!/bin/bash

# ============================================================
# DeIR 混合检索器训练脚本
# 支持四种配置模式：
#   1. baseline: 传统双塔 Baseline
#   2. mlp_only: 纯 MLP 动态路由
#   3. lap_only: 纯 LAP 矩阵投影
#   4. hybrid: 终极完全体 Hybrid
# ============================================================

eval "$(conda shell.bash hook)"
conda activate dsclr
cd /home/luwa/Documents/DSCLR

# ============================================================
# 配置参数
# ============================================================

# 模型选择: "bge", "mistral" 或 "repllama"
ENCODER_TYPE="repllama"

# 训练模式: "baseline", "mlp_only", "lap_only", "hybrid"
TRAIN_MODE="lap_only"

# GPU 设备
GPU_ID=0

# 训练参数
BATCH_SIZE=32
ENCODE_BATCH_SIZE=16
LR=1e-4
EPOCHS=50
NUM_NEG=15
VAL_RATIO=0.1
PATIENCE=5

# 数据目录
DATA_DIR="dataset/FollowIR_train"

# 静态超参（当 use_mlp=False 时使用）
STATIC_ALPHA=1.0
STATIC_TAU=0.5

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

# 缓存路径（根据模型自动选择）
if [ "$ENCODER_TYPE" = "mistral" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/bge-large-en/dsclr_train_embeddings_bge-large-en.pt"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    CACHE_PATH="dataset/FollowIR_train/embeddings/repllama-v1-7b/dsclr_train_embeddings_repllama-v1-7b.pt"
fi

# 输出目录
TIMESTAMP=$(date +%m.%d-%H%M)
OUTPUT_DIR="train/output/Hybrid/${TRAIN_MODE}/${TIMESTAMP}-${MODEL_SHORT}"

# ============================================================
# 根据训练模式设置模块开关
# ============================================================

case "$TRAIN_MODE" in
    baseline)
        USE_LAP="False"
        USE_MLP="False"
        echo "模式: 传统双塔 Baseline"
        ;;
    mlp_only)
        USE_LAP="False"
        USE_MLP="True"
        echo "模式: 纯 MLP 动态路由"
        ;;
    lap_only)
        USE_LAP="True"
        USE_MLP="False"
        echo "模式: 纯 LAP 矩阵投影"
        ;;
    hybrid)
        USE_LAP="True"
        USE_MLP="True"
        echo "模式: 终极完全体 Hybrid"
        ;;
    *)
        echo "错误: 未知的训练模式 '$TRAIN_MODE'"
        echo "支持的模式: baseline, mlp_only, lap_only, hybrid"
        exit 1
        ;;
esac

# ============================================================
# 打印配置信息
# ============================================================

echo "============================================================"
echo "DeIR 混合检索器训练配置"
echo "============================================================"
echo "编码器: $MODEL_NAME"
echo "嵌入维度: $EMBED_DIM"
echo "GPU: $GPU_ID"
echo "训练模式: $TRAIN_MODE"
echo "  - use_lap: $USE_LAP"
echo "  - use_mlp: $USE_MLP"
echo "  - static_alpha: $STATIC_ALPHA"
echo "  - static_tau: $STATIC_TAU"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

# ============================================================
# 检查数据/缓存
# ============================================================

# LAP 模式需要原始数据集，其他模式需要缓存
if [ "$TRAIN_MODE" = "lap_only" ] || [ "$TRAIN_MODE" = "hybrid" ]; then
    # LAP/Hybrid 模式需要原始数据集进行动态编码
    if [ ! -d "$DATA_DIR" ]; then
        echo "⚠️  数据目录不存在: $DATA_DIR"
        exit 1
    fi
    echo "使用原始数据集: $DATA_DIR"
else
    # baseline/mlp_only 模式使用缓存
    if [ ! -f "$CACHE_PATH" ]; then
        echo "⚠️  缓存文件不存在: $CACHE_PATH"
        echo "请先运行数据准备脚本生成缓存"
        exit 1
    fi
fi

# ============================================================
# Wandb 配置
# ============================================================
USE_WANDB="False"

# ============================================================
# 开始训练
# ============================================================

echo ""
echo "🚀 开始训练..."
echo ""

python model/train_hybrid.py \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --encode_batch_size $ENCODE_BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --cache_path "$CACHE_PATH" \
    --lr $LR \
    --epochs $EPOCHS \
    --num_neg $NUM_NEG \
    --val_ratio $VAL_RATIO \
    --patience $PATIENCE \
    --embed_dim $EMBED_DIM \
    --model_type $ENCODER_TYPE \
    --use_lap $USE_LAP \
    --use_mlp $USE_MLP \
    --static_alpha $STATIC_ALPHA \
    --static_tau $STATIC_TAU \
    --use_wandb $USE_WANDB

echo ""
echo "✅ 训练完成!"
echo "输出目录: $OUTPUT_DIR"
