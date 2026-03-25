#!/bin/bash

# ============================================================
# DSCLR MLP 训练脚本
# 支持 BGE-large 和 E5-Mistral-7B 两种编码器
# ============================================================

eval "$(conda shell.bash hook)"
conda activate dsclr
cd /home/luwa/Documents/DSCLR

# ============================================================
# 配置参数（修改这里切换模型）
# ============================================================

# 模型选择: "bge", "mistral" 或 "repllama"
ENCODER_TYPE="repllama"

# GPU 设备
GPU_ID=0

# 训练参数
BATCH_SIZE=32
LR=5e-5
EPOCHS=50
NUM_NEG=15
VAL_RATIO=0.1
PATIENCE=5

# ============================================================
# 根据编码器类型自动设置参数
# ============================================================

if [ "$ENCODER_TYPE" = "mistral" ]; then
    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    MODEL_SHORT="e5-mistral-7b"
    # E5-Mistral 使用 4096 维嵌入
    EMBED_DIM=4096
    # E5-Mistral 显存限制，batch_size 不宜过大
    BATCH_SIZE=28
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    MODEL_SHORT="bge-large-en"
    # BGE 使用 1024 维嵌入
    EMBED_DIM=1024
    BATCH_SIZE=32
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    MODEL_SHORT="repllama-v1-7b"
    # RepLLaMA 使用 4096 维嵌入 (LLaMA-2 7B)
    EMBED_DIM=4096
    # RepLLaMA 也是大模型，batch_size 不宜过大
    BATCH_SIZE=28
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

# 缓存路径（根据模型自动选择）
CACHE_PATH="dataset/FollowIR_train/embeddings/repllama-v1-7b/dsclr_train_embeddings_repllama-v1-7b.pt"

# 输出目录
OUTPUT_DIR="train/output/${MODEL_SHORT}/$(date +%m.%d)-dscrl_mlp"

# ============================================================
# 检查缓存是否存在
# ============================================================

echo "============================================================"
echo "DSCLR MLP 训练配置"
echo "============================================================"
echo "编码器: $MODEL_NAME"
echo "嵌入维度: $EMBED_DIM"
echo "GPU: $GPU_ID"
echo "Batch Size: $BATCH_SIZE"
echo "缓存路径: $CACHE_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

if [ ! -f "$CACHE_PATH" ]; then
    echo "⚠️  缓存文件不存在: $CACHE_PATH"
    echo ""
    echo "请先运行数据准备脚本生成缓存:"
    echo "  python model/prepare_train_data.py --model $MODEL_NAME --device cuda:$GPU_ID --batch_size $BATCH_SIZE"
    echo ""
    read -p "是否现在运行数据准备脚本? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在准备训练数据..."
        python model/prepare_train_data.py \
            --model "$MODEL_NAME" \
            --device "cuda:$GPU_ID" \
            --batch_size $BATCH_SIZE
        
        if [ $? -ne 0 ]; then
            echo "❌ 数据准备失败"
            exit 1
        fi
    else
        echo "退出"
        exit 1
    fi
fi

# ============================================================
# 开始训练
# ============================================================

echo ""
echo "🚀 开始训练..."
echo ""

python model/train_dscrl_mlp.py \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --cache_path "$CACHE_PATH" \
    --lr $LR \
    --epochs $EPOCHS \
    --num_neg $NUM_NEG \
    --val_ratio $VAL_RATIO \
    --patience $PATIENCE \
    --embed_dim $EMBED_DIM \
    --model_type $ENCODER_TYPE

echo ""
echo "✅ 训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "权重文件: dsclr_best_mlp_${ENCODER_TYPE}.pt"
