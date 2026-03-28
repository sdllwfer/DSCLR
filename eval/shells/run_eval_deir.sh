#!/bin/bash
# ============================================================
# DeIR 双流检索评测系统启动脚本
# 支持 DeIR 架构（LAP + 升级 MLP）的评估
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dsclr

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 设置 Python 路径
export PYTHONPATH="/home/luwa/Documents/DSCLR:$PYTHONPATH"

# 验证环境
if [ "$CONDA_DEFAULT_ENV" != "dsclr" ]; then
    echo "❌ 错误: 无法激活 dsclr 环境，当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"
echo "✅ Python 路径: $(which python)"

# ============================================================
# 编码器类型配置: "bge", "mistral" 或 "repllama"
# ============================================================
ENCODER_TYPE="repllama"  # 修改这里切换模型

# ============================================================
# 根据编码器类型自动设置参数
# ============================================================
if [ "$ENCODER_TYPE" = "mistral" ]; then
    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MLP_HIDDEN_DIM=256
    MODEL_SHORT="e5-mistral-7b"
    # DeIR 模型路径配置
    LAP_MODEL_PATH="train/output/Hybrid/lap_then_mlp/03.28-1145-repllama-LAPepoch5/lap_phase/deir_best_lap_phase.pt"
    MLP_MODEL_PATH="train/output/Hybrid/lap_then_mlp/03.28-1145-repllama-LAPepoch5/mlp_phase/deir_best_mlp_phase.pt"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    BATCH_SIZE=256
    EMBED_DIM=1024
    MLP_HIDDEN_DIM=256
    MODEL_SHORT="bge-large-en"
    LAP_MODEL_PATH=""
    MLP_MODEL_PATH=""
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MLP_HIDDEN_DIM=256
    MODEL_SHORT="repllama-v1-7b"
    # DeIR 模型路径配置（默认使用最新训练的模型）
    LAP_MODEL_PATH="train/output/Hybrid/lap_then_mlp/03.28-1145-repllama-LAPepoch5/lap_phase/deir_best_lap_phase.pt"
    MLP_MODEL_PATH="train/output/Hybrid/lap_then_mlp/03.28-1145-repllama-LAPepoch5/mlp_phase/deir_best_mlp_phase.pt"
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

GPU_ID=2
SEED=42
VERBOSE=false
TASK="Core17InstructionRetrieval"
# TASK="Robust04InstructionRetrieval"
# TASK="News21InstructionRetrieval"

# 输出路径 (指定 CUSTOM_OUTPUT_PATH 后直接使用，否则自动生成)
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/deir"
CUSTOM_OUTPUT_PATH=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DeIR 双流检索评测系统启动脚本
支持 LAP + 升级 MLP 架构

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge, mistral, repllama (默认: repllama)
    -m, --model <name>        模型名称或路径 (默认根据encoder自动设置)
    -t, --task <name>         评测任务 (必选)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号: 0/1/2/3 (默认: 2)
    -b, --batch_size <size>  批处理大小 (默认根据encoder自动设置)
    -s, --seed <num>         随机种子 (默认: 42)
    -L, --lap <path>         LAP模型路径 (必须)
    -M, --mlp <path>         MLP模型路径 (必须)
    -H, --mlp_hidden_dim <n> MLP隐藏层维度 (默认: 256)
    -v, --verbose            显示详细日志
    -h, --help               显示帮助信息

编码器类型:
    bge      - BAAI/bge-large-en-v1.5 (1024维, batch_size=256)
    mistral  - intfloat/e5-mistral-7b-instruct (4096维, batch_size=28)
    repllama - castorini/repllama-v1-7b-lora-passage (4096维, batch_size=28)

可用任务:
    Core17InstructionRetrieval
    Robust04InstructionRetrieval
    News21InstructionRetrieval

示例:
    # 使用 RepLLaMA 评测 (默认)
    $0 --task Core17InstructionRetrieval

    # 使用自定义模型路径
    $0 --task Core17InstructionRetrieval \
        --lap train/output/Hybrid/lap_phase/deir_best_lap_phase.pt \
        --mlp train/output/Hybrid/mlp_phase/deir_best_mlp_phase.pt

    # 使用第1张GPU评测
    $0 --task Core17InstructionRetrieval --gpu 1

EOF
}

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--encoder)
            ENCODER_TYPE="$2"
            # 根据编码器类型重新设置参数
            if [ "$ENCODER_TYPE" = "mistral" ]; then
                MODEL_NAME="intfloat/e5-mistral-7b-instruct"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MLP_HIDDEN_DIM=256
                MODEL_SHORT="e5-mistral-7b"
            elif [ "$ENCODER_TYPE" = "bge" ]; then
                MODEL_NAME="BAAI/bge-large-en-v1.5"
                BATCH_SIZE=256
                EMBED_DIM=1024
                MLP_HIDDEN_DIM=256
                MODEL_SHORT="bge-large-en"
            elif [ "$ENCODER_TYPE" = "repllama" ]; then
                MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MLP_HIDDEN_DIM=256
                MODEL_SHORT="repllama-v1-7b"
            else
                echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
                echo "支持的类型: bge, mistral, repllama"
                exit 1
            fi
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -O|--output)
            CUSTOM_OUTPUT_PATH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -L|--lap)
            LAP_MODEL_PATH="$2"
            shift 2
            ;;
        -M|--mlp)
            MLP_MODEL_PATH="$2"
            shift 2
            ;;
        -H|--mlp_hidden_dim)
            MLP_HIDDEN_DIM="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# 参数校验
# ============================================================
VALID_TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK} " ]]; then
    echo "❌ 错误: 无效任务 '$TASK'"
    echo "可用任务: ${VALID_TASKS[@]}"
    exit 1
fi

# 检查 LAP 和 MLP 模型路径
if [ -z "$LAP_MODEL_PATH" ] || [ ! -f "$LAP_MODEL_PATH" ]; then
    echo "❌ 错误: LAP 模型路径无效或未设置: $LAP_MODEL_PATH"
    exit 1
fi

if [ -z "$MLP_MODEL_PATH" ] || [ ! -f "$MLP_MODEL_PATH" ]; then
    echo "❌ 错误: MLP 模型路径无效或未设置: $MLP_MODEL_PATH"
    exit 1
fi

# 生成输出目录
if [ -n "$CUSTOM_OUTPUT_PATH" ]; then
    OUTPUT_DIR="${CUSTOM_OUTPUT_PATH}"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_SHORT}/${TASK}_${TIMESTAMP}"
    echo "📂 使用自动生成路径: ${OUTPUT_DIR}"
fi

# ============================================================
# 显示配置
# ============================================================
echo "============================================================"
echo "DeIR 双流检索评测系统"
echo "============================================================"
echo "编码器类型: ${ENCODER_TYPE}"
echo "模型: ${MODEL_NAME}"
echo "嵌入维度: ${EMBED_DIM}"
echo "MLP隐藏层: ${MLP_HIDDEN_DIM}"
echo "查询重构: LLM API (实时解耦)"
echo "任务: ${TASK}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "批处理: ${BATCH_SIZE}"
echo "种子: ${SEED}"
echo "LAP模型: ${LAP_MODEL_PATH}"
echo "MLP模型: ${MLP_MODEL_PATH}"
[ "$VERBOSE" = true ] && echo "日志: 详细模式"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python eval/engine_deir.py \
    --model_name "$MODEL_NAME" \
    --task_name "$TASK" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --batch_size "$BATCH_SIZE" \
    --use_cache True \
    --lap_model_path "$LAP_MODEL_PATH" \
    --mlp_model_path "$MLP_MODEL_PATH" \
    --mlp_hidden_dim "$MLP_HIDDEN_DIM" \
    --save_analysis True

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ DeIR 评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
