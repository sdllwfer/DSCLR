#!/bin/bash
# ============================================================
# FollowIR 评测系统启动脚本
# 支持灵活配置模型路径、评测任务和参数
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dsclr

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

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
DEVICE="cuda"
BATCH_SIZE=64
SEED=42
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation"
TASKS=()
VERBOSE=false

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
FollowIR 评测系统启动脚本

用法: $0 [选项]

选项:
    -m, --model <name>        模型名称或路径 (默认: BAAI/bge-large-en-v1.5)
    -t, --task <name>         单个评测任务
    -T, --tasks <names>      多个评测任务 (用空格分隔)
    -o, --output <dir>       输出目录 (默认自动生成)
    -d, --device <device>    计算设备: cuda 或 cpu (默认: cuda)
    -b, --batch_size <size>  批处理大小 (默认: 64)
    -s, --seed <num>         随机种子 (默认: 42)
    -v, --verbose            显示详细日志
    -h, --help               显示帮助信息

可用任务:
    Core17InstructionRetrieval
    Robust04InstructionRetrieval
    News21InstructionRetrieval

示例:
    # 评测单个任务
    $0 --model BAAI/bge-large-en-v1.5 --task Core17InstructionRetrieval

    # 评测多个任务
    $0 -m BAAI/bge-large-en-v1.5 -T Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval

    # 指定输出目录和参数
    $0 --model BAAI/bge-base-en-v1.5 --task Core17 --output /tmp/eval --batch_size 128

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
            TASKS+=("$2")
            shift 2
            ;;
        -T|--tasks)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                TASKS+=("$1")
                shift
            done
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
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
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "❌ 错误: 未指定评测任务"
    show_help
    exit 1
fi

VALID_TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
for task in "${TASKS[@]}"; do
    if [[ ! " ${VALID_TASKS[@]} " =~ " ${task} " ]]; then
        echo "❌ 错误: 无效任务 '$task'"
        echo "可用任务: ${VALID_TASKS[@]}"
        exit 1
    fi
done

# 生成输出目录
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_DIR_NAME}_${TIMESTAMP}"
fi

# ============================================================
# 显示配置
# ============================================================
echo "============================================================"
echo "FollowIR 评测系统"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "任务: ${TASKS[*]}"
echo "输出: ${OUTPUT_DIR}"
echo "设备: ${DEVICE}"
echo "批处理: ${BATCH_SIZE}"
echo "种子: ${SEED}"
[ "$VERBOSE" = true ] && echo "日志: 详细模式"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# 构建 Python 命令
# ============================================================
PYTHON_ARGS=()

PYTHON_ARGS+=("--model" "$MODEL_NAME")
PYTHON_ARGS+=("--output" "$OUTPUT_DIR")
PYTHON_ARGS+=("--device" "$DEVICE")
PYTHON_ARGS+=("--batch_size" "$BATCH_SIZE")
PYTHON_ARGS+=("--seed" "$SEED")

if [ ${#TASKS[@]} -eq 1 ]; then
    PYTHON_ARGS+=("--task" "${TASKS[0]}")
else
    PYTHON_ARGS+=("--tasks" "${TASKS[@]}")
fi

[ "$VERBOSE" = true ] && PYTHON_ARGS+=("--verbose")

# ============================================================
# 运行评测
# ============================================================
cd /home/luwa/Documents/DSCLR

python -u -m eval.main "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ 评测完成!"
    echo "📁 结果保存至: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
