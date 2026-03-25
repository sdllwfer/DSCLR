#!/bin/bash
# ============================================================
# PAT (Positive-Aware Tolerance) 极限 Grid Search 评测脚本
# 基于正向得分自适应调整阈值: dynamic_tau = tau_base + lambda * S_base
# 目标: 冲击 p-MRR > 0.154 基线
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
# PAT 默认配置 - 极限搜索空间
# ============================================================
ENCODER_TYPE="repllama"

# PAT 极限搜索空间
ALPHA_LIST="3.0,4.0,5.0,6.0"
TAU_BASE_LIST="0.4,0.5"
LAMBDA_LIST="0.3,0.5,0.7"

# ============================================================
# 根据编码器类型自动设置参数
# ============================================================
if [ "$ENCODER_TYPE" = "mistral" ]; then
    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MODEL_SHORT="e5-mistral-7b"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    BATCH_SIZE=256
    EMBED_DIM=1024
    MODEL_SHORT="bge-large-en"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MODEL_SHORT="repllama-v1-7b"
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

GPU_ID=0
SEED=42
VERBOSE=false
TASK="Core17InstructionRetrieval"

# 输出路径
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/pat_search"
CUSTOM_OUTPUT_PATH=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
PAT (Positive-Aware Tolerance) 极限 Grid Search 评测脚本
基于正向得分自适应调整阈值，冲击 p-MRR > 0.154 基线

核心公式:
    dynamic_tau = tau_base + lambda_weight * S_base
    penalty = alpha * relu(S_neg - dynamic_tau)
    S_final = S_base - penalty

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge, mistral, repllama (默认: repllama)
    -m, --model <name>        模型名称或路径 (默认根据encoder自动设置)
    -t, --task <name>         评测任务 (默认: Core17InstructionRetrieval)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号: 0/1/2/3 (默认: 0)
    -b, --batch_size <size>  批处理大小 (默认根据encoder自动设置)
    -s, --seed <num>         随机种子 (默认: 42)
    -v, --verbose            显示详细日志

    # PAT 专用参数 (极限搜索空间)
    --alpha-list <list>      Alpha参数列表 (默认: "3.0,4.0,5.0,6.0")
    --tau-base-list <list>   Tau基础阈值列表 (默认: "0.4,0.5")
    --lambda-list <list>     Lambda权重列表 (默认: "0.3,0.5,0.7")

    -h, --help               显示帮助信息

PAT 极限搜索空间:
    alpha = [3.0, 4.0, 5.0, 6.0]
    tau_base = [0.4, 0.5]
    lambda = [0.3, 0.5, 0.7]
    总组合数: 4 × 2 × 3 = 24

示例:
    # 使用默认参数运行 PAT 极限评测
    $0 --encoder repllama --task Core17InstructionRetrieval

    # 自定义参数列表
    $0 --encoder repllama --task Core17InstructionRetrieval --alpha-list "3.0,5.0" --lambda-list "0.3,0.7"

    # 指定 GPU 和输出目录
    $0 --encoder repllama --task Core17InstructionRetrieval --gpu 1 --output /path/to/output

EOF
}

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--encoder)
            ENCODER_TYPE="$2"
            if [ "$ENCODER_TYPE" = "mistral" ]; then
                MODEL_NAME="intfloat/e5-mistral-7b-instruct"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MODEL_SHORT="e5-mistral-7b"
            elif [ "$ENCODER_TYPE" = "bge" ]; then
                MODEL_NAME="BAAI/bge-large-en-v1.5"
                BATCH_SIZE=256
                EMBED_DIM=1024
                MODEL_SHORT="bge-large-en"
            elif [ "$ENCODER_TYPE" = "repllama" ]; then
                MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
                BATCH_SIZE=28
                EMBED_DIM=4096
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
        --alpha-list)
            ALPHA_LIST="$2"
            shift 2
            ;;
        --tau-base-list)
            TAU_BASE_LIST="$2"
            shift 2
            ;;
        --lambda-list)
            LAMBDA_LIST="$2"
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
echo "🎯 PAT (Positive-Aware Tolerance) 极限 Grid Search"
echo "============================================================"
echo "编码器类型: ${ENCODER_TYPE}"
echo "模型: ${MODEL_NAME}"
echo "嵌入维度: ${EMBED_DIM}"
echo "任务: ${TASK}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "批处理: ${BATCH_SIZE}"
echo "种子: ${SEED}"
echo "------------------------------------------------------------"
echo "PAT 极限搜索空间:"
echo "  Alpha 列表: [${ALPHA_LIST}]"
echo "  Tau_base 列表: [${TAU_BASE_LIST}]"
echo "  Lambda 列表: [${LAMBDA_LIST}]"
N_ALPHA=$(echo "${ALPHA_LIST//,/ }" | wc -w)
N_TAU=$(echo "${TAU_BASE_LIST//,/ }" | wc -w)
N_LAMBDA=$(echo "${LAMBDA_LIST//,/ }" | wc -w)
echo "  总组合数: ${N_ALPHA} × ${N_TAU} × ${N_LAMBDA} = $(( N_ALPHA * N_TAU * N_LAMBDA ))"
echo "------------------------------------------------------------"
echo "目标: p-MRR > 0.154 且 nDCG 不崩盘"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# 构建Python命令参数
# ============================================================
PYTHON_ARGS="--model_name ${MODEL_NAME} \
    --task_name ${TASK} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda:${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --use_cache True \
    --alpha_list ${ALPHA_LIST} \
    --tau_base_list ${TAU_BASE_LIST} \
    --lambda_list ${LAMBDA_LIST}"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo ""
echo "🚀 启动 PAT 极限搜索评测..."
echo ""

python eval/pat_scorer.py ${PYTHON_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ PAT 极限搜索完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
