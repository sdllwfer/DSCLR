#!/bin/bash
# TRACE Secondary Design Choices Ablation Experiment
# 运行三个消融实验：OLS fit, Mean/std scaling, Uncentered residual

set -e  # 遇到错误立即退出

# 配置参数
MODEL="samaya-ai/RepLLaMA-reproduced"
DUAL_QUERIES_BASE="dataset/FollowIR_test/dual_queries_v6/dual_queries_v6"
OUTPUT_BASE="results/trace_secondary_ablation"
DEVICE="cuda"
USE_CACHE="true"

# 固定参数
HUBER_DELTA="1.345"
LAMBDA="1.0"
TAU="0.2"

echo "=============================================="
echo "TRACE Secondary Design Choices Ablation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Use cache: $USE_CACHE"
echo "  Fixed params: huber_delta=$HUBER_DELTA, lambda=$LAMBDA, tau=$TAU"
echo ""

# 数据集列表
DATASETS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")

# 函数：运行单个实验
run_experiment() {
    local VARIANT=$1
    local DATASET=$2
    local REG_MODE=$3
    local NORM_MODE=$4
    local UNCENTERED=$5

    echo ""
    echo "========================================"
    echo "Running: $VARIANT on $DATASET"
    echo "========================================"
    echo "  regression_mode: $REG_MODE"
    echo "  normalization_mode: $NORM_MODE"
    echo "  uncentered_residual: $UNCENTERED"
    echo ""

    /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_trace \
        --task_name "$DATASET" \
        --model_name "$MODEL" \
        --dual_queries_path "${DUAL_QUERIES_BASE}_${DATASET}.jsonl" \
        --output_dir "${OUTPUT_BASE}/${VARIANT}/${DATASET}" \
        --huber_delta "$HUBER_DELTA" \
        --lambda_boundary "$LAMBDA" \
        --tau_decay "$TAU" \
        --regression_mode "$REG_MODE" \
        --normalization_mode "$NORM_MODE" \
        --uncentered_residual "$UNCENTERED" \
        --use_cache "$USE_CACHE" \
        --device "$DEVICE" \
        --batch_size 64

    echo ""
    echo "✅ Completed: $VARIANT on $DATASET"
    echo ""
}

# 主实验流程
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing dataset: $DATASET"
    echo "######################################################################"

    # 1. 基准实验（验证表格值）
    run_experiment "default" "$DATASET" "huber" "median_mad" "false"

    # 2. OLS fit ablation
    run_experiment "ols_fit" "$DATASET" "ols" "median_mad" "false"

    # 3. Mean/std scaling ablation
    run_experiment "mean_std_scaling" "$DATASET" "huber" "mean_std" "false"

    # 4. Uncentered residual ablation
    run_experiment "uncentered_residual" "$DATASET" "huber" "median_mad" "true"
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Next step: Run the summary script to extract metrics"