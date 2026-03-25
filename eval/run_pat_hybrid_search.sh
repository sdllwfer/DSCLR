#!/bin/bash
# ============================================================
# PAT (Positive-Aware Tolerance) 混合策略版 Grid Search 评测脚本
# 策略: 保护 top-k 文档 + boost 6-20 排名文档
# 目标: 突破 p-MRR 和 nDCG 的 trade-off
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dsclr

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 设置 Python 路径
export PYTHONPATH="/home/luwa/Documents/DSCLR:$PYTHONPATH"

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"

# ============================================================
# 默认配置
# ============================================================
ENCODER_TYPE="repllama"

# 基础搜索空间
ALPHA_LIST="3.0,4.0,5.0"
TAU_BASE_LIST="0.4,0.5"
LAMBDA_LIST="0.3,0.4,0.5"

# 混合策略参数
TOP_K=5
PROTECTION_FACTOR=0.0
BOOST_NDCG_FACTOR=0.5

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
fi

GPU_ID=0
SEED=42
TASK="Core17InstructionRetrieval"

# 输出路径
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/pat_hybrid_search"
CUSTOM_OUTPUT_PATH=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
PAT (混合策略) Grid Search 评测脚本

核心策略:
    - og_rank <= top_k: 保护文档 (不惩罚)
    - og_rank 6-20: 轻微 boost (帮助进入 top-5)

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge, mistral, repllama (默认: repllama)
    -t, --task <name>         评测任务 (默认: Core17InstructionRetrieval)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号 (默认: 0)

    # PAT 搜索参数
    --alpha-list <list>      Alpha参数列表 (默认: "3.0,4.0,5.0")
    --tau-base-list <list>   Tau基础阈值列表 (默认: "0.4,0.5")
    --lambda-list <list>     Lambda权重列表 (默认: "0.3,0.4,0.5")

    # 混合策略参数
    --top-k <num>            保护阈值 (默认: 5)
    --protection-factor <num> 保护系数 (默认: 0.0，完全不惩罚)
    --boost-factor <num>     Boost系数 (默认: 0.5)

    -h, --help               显示帮助信息

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
                MODEL_SHORT="e5-mistral-7b"
            elif [ "$ENCODER_TYPE" = "bge" ]; then
                MODEL_NAME="BAAI/bge-large-en-v1.5"
                BATCH_SIZE=256
                MODEL_SHORT="bge-large-en"
            elif [ "$ENCODER_TYPE" = "repllama" ]; then
                MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
                BATCH_SIZE=28
                MODEL_SHORT="repllama-v1-7b"
            fi
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
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --protection-factor)
            PROTECTION_FACTOR="$2"
            shift 2
            ;;
        --boost-factor)
            BOOST_NDCG_FACTOR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

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
echo "🚀 PAT (混合策略) Extreme Grid Search"
echo "============================================================"
echo "编码器类型: ${ENCODER_TYPE}"
echo "模型: ${MODEL_NAME}"
echo "任务: ${TASK}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "------------------------------------------------------------"
echo "PAT 搜索空间:"
echo "  Alpha 列表: [${ALPHA_LIST}]"
echo "  Tau_base 列表: [${TAU_BASE_LIST}]"
echo "  Lambda 列表: [${LAMBDA_LIST}]"
echo "------------------------------------------------------------"
echo "混合策略参数:"
echo "  Top_k: ${TOP_K}"
echo "  Protection Factor: ${PROTECTION_FACTOR}"
echo "  Boost Factor: ${BOOST_NDCG_FACTOR}"
N_ALPHA=$(echo "${ALPHA_LIST//,/ }" | wc -w)
N_TAU=$(echo "${TAU_BASE_LIST//,/ }" | wc -w)
N_LAMBDA=$(echo "${LAMBDA_LIST//,/ }" | wc -w)
echo "  总组合数: ${N_ALPHA} × ${N_TAU} × ${N_LAMBDA} = $(( N_ALPHA * N_TAU * N_LAMBDA ))"
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
    --lambda_list ${LAMBDA_LIST} \
    --top_k ${TOP_K} \
    --protection_factor ${PROTECTION_FACTOR} \
    --boost_ndcg_factor ${BOOST_NDCG_FACTOR}"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo ""
echo "🚀 启动 PAT 混合策略评测..."
echo ""

python -c "
import sys
sys.path.insert(0, '/home/luwa/Documents/DSCLR')
from eval.pat_scorer import run_pat_hybrid_grid_search_evaluation
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--task_name', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--use_cache', type=bool)
parser.add_argument('--alpha_list', type=str)
parser.add_argument('--tau_base_list', type=str)
parser.add_argument('--lambda_list', type=str)
parser.add_argument('--top_k', type=int)
parser.add_argument('--protection_factor', type=float)
parser.add_argument('--boost_ndcg_factor', type=float)
args = parser.parse_args()

run_pat_hybrid_grid_search_evaluation(
    model_name=args.model_name,
    task_name=args.task_name,
    output_dir=args.output_dir,
    device=args.device,
    batch_size=args.batch_size,
    use_cache=args.use_cache,
    alpha_list=[float(x) for x in args.alpha_list.split(',')],
    tau_base_list=[float(x) for x in args.tau_base_list.split(',')],
    lambda_list=[float(x) for x in args.lambda_list.split(',')],
    top_k=args.top_k,
    protection_factor=args.protection_factor,
    boost_ndcg_factor=args.boost_ndcg_factor
)
" ${PYTHON_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ PAT 混合策略评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
