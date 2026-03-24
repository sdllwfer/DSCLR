#!/bin/bash
# ============================================================
# DADT (Distribution-Aware Dynamic Threshold) 动态阈值评测脚本
# 基于负样本统计分布自动计算动态阈值 tau
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
# DADT 默认配置
# ============================================================
ENCODER_TYPE="repllama"
GAMMA_LIST="0.0,0.5,1.0,1.5,2.0"
ALPHA_LIST="1.5,2.0,2.5"

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
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/dadt_search"
# 注意：如果需要自定义路径，请在运行脚本时通过 -O 参数传入
# 例如：bash run_dadt_search.sh -O "/path/to/custom/output"
CUSTOM_OUTPUT_PATH=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DADT (Distribution-Aware Dynamic Threshold) 动态阈值评测脚本
基于负样本统计分布自动计算动态阈值 tau = mu + gamma * sigma

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
    
    # DADT 专用参数
    --gamma-list <list>      Gamma参数列表,逗号分隔 (默认: "0.0,0.5,1.0,1.5,2.0")
    --alpha-list <list>      Alpha参数列表,逗号分隔 (默认: "1.5,2.0,2.5")
    
    -h, --help               显示帮助信息

编码器类型:
    bge       - BAAI/bge-large-en-v1.5 (1024维, batch_size=256)
    mistral   - intfloat/e5-mistral-7b-instruct (4096维, batch_size=28)
    repllama  - castorini/repllama-v1-7b-lora-passage (4096维, batch_size=28)

可用任务:
    Core17InstructionRetrieval
    Robust04InstructionRetrieval
    News21InstructionRetrieval

DADT 原理:
    tau = mu + gamma * sigma
    其中:
    - mu: 负样本相似度(S_neg)的均值
    - sigma: 负样本相似度的标准差
    - gamma: 标准差乘数(控制阈值偏离程度)

示例:
    # 使用默认参数运行 DADT 评测
    $0 --encoder repllama --task Core17InstructionRetrieval

    # 自定义 Gamma 和 Alpha 列表
    $0 --encoder repllama --task Core17InstructionRetrieval --gamma-list "0.0,1.0,2.0" --alpha-list "2.0,3.0,4.0"

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
            # 根据编码器类型重新设置参数
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
        --gamma-list)
            GAMMA_LIST="$2"
            shift 2
            ;;
        --alpha-list)
            ALPHA_LIST="$2"
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
    # 用户通过 -O 参数自定义了路径
    OUTPUT_DIR="${CUSTOM_OUTPUT_PATH}"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
else
    # 使用默认路径 + 时间戳
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_SHORT}/${TASK}_${TIMESTAMP}"
    echo "📂 使用自动生成路径: ${OUTPUT_DIR}"
fi

# ============================================================
# 显示配置
# ============================================================
echo "============================================================"
echo "🔬 DADT (Distribution-Aware Dynamic Threshold) 评测系统"
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
echo "DADT 参数:"
echo "  Gamma列表: [${GAMMA_LIST}]"
echo "  Alpha列表: [${ALPHA_LIST}]"
echo "  总组合数: $(echo "${GAMMA_LIST//,/ }" | wc -w) × $(echo "${ALPHA_LIST//,/ }" | wc -w) = $(( $(echo "${GAMMA_LIST//,/ }" | wc -w) * $(echo "${ALPHA_LIST//,/ }" | wc -w) ))"
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
    --dadt \
    --gamma_list ${GAMMA_LIST} \
    --alpha_list ${ALPHA_LIST}"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo ""
echo "🚀 启动 DADT 动态阈值评测..."
echo ""

python eval/engine_grid_search.py ${PYTHON_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ DADT 评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
