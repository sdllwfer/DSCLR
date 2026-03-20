#!/bin/bash
# ============================================================
# DSCLR 双流检索评测系统启动脚本
# 支持 DSCLR 架构的网格搜索超参数调优
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
# 默认配置
# ============================================================
MODEL_NAME="BAAI/bge-large-en-v1.5"
GPU_ID=0
BATCH_SIZE=256
SEED=42
VERBOSE=false
# TASK="Core17InstructionRetrieval"
TASK="Robust04InstructionRetrieval"
# 输出路径 (指定 CUSTOM_OUTPUT_PATH 后直接使用，否则自动生成)
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/DSCLR/evaluation/dsclr/${TASK}_test"

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DSCLR 双流检索评测系统启动脚本

用法: $0 [选项]

选项:
    -m, --model <name>        模型名称或路径 (默认: BAAI/bge-large-en-v1.5)
    -t, --task <name>         评测任务 (必选)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号: 0/1/2/3 (默认: 0)
    -b, --batch_size <size>  批处理大小 (默认: 256)
    -s, --seed <num>         随机种子 (默认: 42)
    -v, --verbose            显示详细日志
    -h, --help               显示帮助信息

可用任务:
    Core17InstructionRetrieval
    Robust04InstructionRetrieval
    News21InstructionRetrieval

示例:
    # 评测单个任务 (使用第1张GPU)
    $0 --task Core17InstructionRetrieval

    # 使用第2张GPU评测
    $0 --task Core17InstructionRetrieval --gpu 1

    # 指定输出目录
    $0 -t Core17InstructionRetrieval -O /tmp/dsclr_eval

EOF
}

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -O|--output)
            OUTPUT_DIR="$2"
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
    MODEL_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_DIR_NAME}_${TIMESTAMP}"
    echo "📂 使用自动生成路径: ${OUTPUT_DIR}"
fi

# ============================================================
# 显示配置
# ============================================================
echo "============================================================"
echo "DSCLR 双流检索评测系统"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "查询重构: LLM API (实时解耦)"
echo "任务: ${TASK}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "批处理: ${BATCH_SIZE}"
echo "种子: ${SEED}"
[ "$VERBOSE" = true ] && echo "日志: 详细模式"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python eval/engine_dscrl.py \
    --model_name "$MODEL_NAME" \
    --task_name "$TASK" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --batch_size "$BATCH_SIZE" \
    --use_cache True

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ DSCLR 评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
