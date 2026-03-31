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
# 编码器类型配置: "bge" 或 "mistral"
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
    MLP_MODEL_PATH="train/output/e5-mistral-7b/03.22-dscrl_mlp_test/dsclr_best_mlp_mistral.pt"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    BATCH_SIZE=256
    EMBED_DIM=1024
    MLP_HIDDEN_DIM=256
    MODEL_SHORT="bge-large-en"
    MLP_MODEL_PATH="train/output/bge-large-en/03.21-dscrl_mlp_test/dsclr_best_mlp_bge.pt"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MLP_HIDDEN_DIM=256
    MODEL_SHORT="repllama-v1-7b"
    MLP_MODEL_PATH="train/output/repllama-v1-7b/03.23-dscrl_mlp/dsclr_best_mlp_repllama.pt"
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

GPU_ID=1
SEED=42
VERBOSE=false
TASK="Core17InstructionRetrieval"
# TASK="Robust04InstructionRetrieval"
# TASK="News21InstructionRetrieval"

# 实验备注
EXPERIMENT_NOTE=""

# 硬负文档嵌入路径（可选）
HARD_NEG_DOC_EMBED_PATH=""

# 模型架构参数
EMBED_DIM=4096
NUM_LAYERS=3
LAP_RANK=128
DROPOUT=0.1

# 输出路径 (指定 CUSTOM_OUTPUT_PATH 后直接使用，否则自动生成)
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr"
CUSTOM_OUTPUT_PATH=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DSCLR 双流检索评测系统启动脚本

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge 或 mistral (默认: repllama)
    -m, --model <name>        模型名称或路径 (默认根据encoder自动设置)
    -t, --task <name>         评测任务 (必选)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号: 0/1/2/3 (默认: 1)
    -b, --batch_size <size>  批处理大小 (默认根据encoder自动设置)
    -s, --seed <num>         随机种子 (默认: 42)
    -M, --mlp <path>         MLP模型路径 (可选，使用动态MLP推理)
    -H, --mlp_hidden_dim <n> MLP隐藏层维度 (默认: 256)
    -N, --note <text>         实验备注 (可选，记录到配置文件中)
    -D, --doc_embed <path>   硬负文档嵌入路径 (可选，用于硬负数据集)
    --embed_dim <n>           模型嵌入维度 (默认: 4096)
    --num_layers <n>          MLP层数 (默认: 3)
    --lap_rank <n>            LAP秩 (默认: 128)
    --dropout <rate>          Dropout率 (默认: 0.1)
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
    # 使用 Repllama 评测 (默认)
    $0 --task Core17InstructionRetrieval

    # 使用 BGE 评测
    $0 --encoder bge --task Core17InstructionRetrieval

    # 使用第2张GPU评测
    $0 --task Core17InstructionRetrieval --gpu 1

    # 使用动态MLP推理
    $0 --task Core17InstructionRetrieval --mlp train/output/e5-mistral-7b/03.21-dscrl_mlp_test/dsclr_best_mlp_mistral.pt

    # 使用硬负文档嵌入
    $0 --task Core17InstructionRetrieval --doc_embed /path/to/hard_neg_doc_embeddings.pt

    # 添加实验备注
    $0 --task Core17InstructionRetrieval --note "测试原始query+instruction作为S_base"
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
                MLP_MODEL_PATH="train/output/e5-mistral-7b/03.21-dscrl_mlp_test/dsclr_best_mlp_mistral.pt"
            elif [ "$ENCODER_TYPE" = "bge" ]; then
                MODEL_NAME="BAAI/bge-large-en-v1.5"
                BATCH_SIZE=256
                EMBED_DIM=1024
                MLP_HIDDEN_DIM=256
                MODEL_SHORT="bge-large-en"
                MLP_MODEL_PATH="train/output/bge-large-en/03.21-dscrl_mlp_test/dsclr_best_mlp_bge.pt"
            elif [ "$ENCODER_TYPE" = "repllama" ]; then
                MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MLP_HIDDEN_DIM=256
                MODEL_SHORT="repllama-v1-7b"
                MLP_MODEL_PATH="train/output/repllama-v1-7b/03.23-dscrl_mlp/dsclr_best_mlp_repllama.pt"
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
        -M|--mlp)
            MLP_MODEL_PATH="$2"
            shift 2
            ;;
        -H|--mlp_hidden_dim)
            MLP_HIDDEN_DIM="$2"
            shift 2
            ;;
        -N|--note)
            EXPERIMENT_NOTE="$2"
            shift 2
            ;;
        -D|--doc_embed)
            HARD_NEG_DOC_EMBED_PATH="$2"
            shift 2
            ;;
        --embed_dim)
            EMBED_DIM="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --lap_rank)
            LAP_RANK="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ============================================================
# 保存实验配置到文件（便于复现）
# ============================================================
CONFIG_FILE="${OUTPUT_DIR}/experiment_config.txt"
cat > "${CONFIG_FILE}" << EOF
============================================================
DSCLR 评测实验配置
============================================================

实验时间: $(date '+%Y-%m-%d %H:%M:%S')
输出目录: ${OUTPUT_DIR}

编码器配置:
  ENCODER_TYPE: ${ENCODER_TYPE}
  MODEL_NAME: ${MODEL_NAME}
  EMBED_DIM: ${EMBED_DIM}

MLP配置:
  MLP_HIDDEN_DIM: ${MLP_HIDDEN_DIM}
  MLP_MODEL_PATH: ${MLP_MODEL_PATH:-未使用}

模型架构:
  NUM_LAYERS: ${NUM_LAYERS}
  LAP_RANK: ${LAP_RANK}
  DROPOUT: ${DROPOUT}

文档嵌入:
  HARD_NEG_DOC_EMBED: ${HARD_NEG_DOC_EMBED_PATH:-未使用}

运行配置:
  TASK: ${TASK}
  GPU_ID: ${GPU_ID}
  BATCH_SIZE: ${BATCH_SIZE}
  SEED: ${SEED}

实验备注:
  ${EXPERIMENT_NOTE:-无}

复现命令:
  cd /home/luwa/Documents/DSCLR
  bash eval/run_eval_dscrl.sh \\
    --encoder ${ENCODER_TYPE} \\
    --task ${TASK} \\
    --gpu ${GPU_ID} \\
    --batch_size ${BATCH_SIZE} \\
    --seed ${SEED} \\
    --mlp ${MLP_MODEL_PATH:-none} \\
    --mlp_hidden_dim ${MLP_HIDDEN_DIM} \\
    --output "${OUTPUT_DIR}" \\
    ${HARD_NEG_DOC_EMBED_PATH:+--doc_embed "${HARD_NEG_DOC_EMBED_PATH}" \\}
    --embed_dim ${EMBED_DIM} \\
    --num_layers ${NUM_LAYERS} \\
    --lap_rank ${LAP_RANK} \\
    --dropout ${DROPOUT} \\
    ${EXPERIMENT_NOTE:+--note "${EXPERIMENT_NOTE}"}
EOF

echo "📝 实验配置已保存到: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

# ============================================================
# 显示配置
# ============================================================
echo "============================================================"
echo "DSCLR 双流检索评测系统"
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
[ -n "$MLP_MODEL_PATH" ] && echo "MLP模型: ${MLP_MODEL_PATH}"
[ -n "$HARD_NEG_DOC_EMBED_PATH" ] && echo "硬负文档嵌入: ${HARD_NEG_DOC_EMBED_PATH}"
[ "$VERBOSE" = true ] && echo "日志: 详细模式"
echo "============================================================"

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 构建命令参数
CMD_ARGS=(
    --model_name "$MODEL_NAME"
    --task_name "$TASK"
    --output_dir "$OUTPUT_DIR"
    --device "cuda"
    --batch_size "$BATCH_SIZE"
    --use_cache True
    --mlp_model_path "$MLP_MODEL_PATH"
    --mlp_hidden_dim "$MLP_HIDDEN_DIM"
    --embed_dim "$EMBED_DIM"
    --num_layers "$NUM_LAYERS"
    --lap_rank "$LAP_RANK"
    --dropout "$DROPOUT"
)

# 如果指定了硬负文档嵌入路径，添加到参数中
if [ -n "$HARD_NEG_DOC_EMBED_PATH" ]; then
    CMD_ARGS+=(--hard_neg_doc_embed_path "$HARD_NEG_DOC_EMBED_PATH")
fi

# 执行评测
python eval/engine_dscrl.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ DSCLR 评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "📝 实验配置: ${CONFIG_FILE}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
