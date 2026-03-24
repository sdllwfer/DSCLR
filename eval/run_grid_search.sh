#!/bin/bash
# ============================================================
# DSCLR 网格搜索评测系统启动脚本
# 支持使用固定参数组合进行网格搜索（替代MLP动态计算）
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
ENCODER_TYPE="mistral"  # 修改这里切换模型

# ============================================================
# 网格搜索参数配置
# ============================================================
# 格式: "alpha1,tau1;alpha2,tau2;..."
# 示例: "0.5,0.45;1.0,0.5;1.2,0.55"
GRID_PARAMS=""

# 是否使用预定义的网格参数组合
USE_PREDEFINED_GRID=false

# 预定义网格类型: "conservative", "aggressive", "balanced"
PREDEFINED_GRID_TYPE="balanced"

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

# 输出路径 (网格搜索结果存储路径)
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/grid_search"
CUSTOM_OUTPUT_PATH=""

# 断点续跑配置
RESUME_FROM_CHECKPOINT=false
CHECKPOINT_FILE=""

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DSCLR 网格搜索评测系统启动脚本
使用固定参数组合替代MLP动态计算

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge, mistral, repllama (默认: mistral)
    -m, --model <name>        模型名称或路径 (默认根据encoder自动设置)
    -t, --task <name>         评测任务 (默认: Core17InstructionRetrieval)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>           GPU编号: 0/1/2/3 (默认: 0)
    -b, --batch_size <size>  批处理大小 (默认根据encoder自动设置)
    -s, --seed <num>         随机种子 (默认: 42)
    -v, --verbose            显示详细日志
    
    # 网格搜索专用参数
    --grid-params <params>   自定义网格参数,格式: "alpha1,tau1;alpha2,tau2;..."
    --predefined-grid <type> 使用预定义网格: conservative, aggressive, balanced
    --resume <checkpoint>    从检查点文件恢复测试
    
    # DADT 模式参数（与上述参数互斥）
    --dadt                   启用 DADT 动态阈值模式
    --gamma-list <list>      DADT gamma参数,默认: "0.0,0.5,1.0,1.5,2.0"
    --alpha-list <list>      DADT alpha参数,默认: "1.5,2.0,2.5"
    
    -h, --help               显示帮助信息

编码器类型:
    bge       - BAAI/bge-large-en-v1.5 (1024维, batch_size=256)
    mistral   - intfloat/e5-mistral-7b-instruct (4096维, batch_size=28)
    repllama  - castorini/repllama-v1-7b-lora-passage (4096维, batch_size=28)

预定义网格类型:
    conservative   - 保守参数: Alpha=[0.5,0.8,1.0], Tau=[0.5,0.55,0.6]
    balanced       - 平衡参数: Alpha=[0.8,1.0,1.2], Tau=[0.45,0.5,0.55]
    aggressive     - 激进参数: Alpha=[1.0,1.2,1.5], Tau=[0.4,0.45,0.5]
    repllama_25    - RepLLaMA专用25组: Alpha=[0.5,0.8,1.2,1.5,2.0], Tau=[0.50,0.60,0.70,0.80,0.90]

可用任务:
    Core17InstructionRetrieval
    Robust04InstructionRetrieval
    News21InstructionRetrieval

示例:
    # 使用预定义网格评测（原始网格搜索）
    $0 --encoder repllama --task Core17InstructionRetrieval --predefined-grid balanced

    # 使用自定义参数网格（原始网格搜索）
    $0 --encoder repllama --task Core17InstructionRetrieval --grid-params "0.5,0.45;1.0,0.5;1.2,0.55"

    # DADT 动态阈值模式（新模式）
    $0 --encoder repllama --task Core17InstructionRetrieval --dadt
    
    # DADT 自定义参数
    $0 --encoder repllama --task Core17InstructionRetrieval --dadt --gamma-list "0.0,1.0,2.0" --alpha-list "2.0,3.0"

    # 断点续跑
    $0 --encoder repllama --task Core17InstructionRetrieval --predefined-grid balanced --resume /path/to/checkpoint.json

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
        --grid-params)
            GRID_PARAMS="$2"
            USE_PREDEFINED_GRID=false
            shift 2
            ;;
        --predefined-grid)
            PREDEFINED_GRID_TYPE="$2"
            USE_PREDEFINED_GRID=true
            shift 2
            ;;
        --resume)
            RESUME_FROM_CHECKPOINT=true
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        --dadt)
            USE_DADT=true
            shift
            ;;
        --gamma-list)
            DADT_GAMMA_LIST="$2"
            shift 2
            ;;
        --alpha-list)
            DADT_ALPHA_LIST="$2"
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

# 默认DADT参数
USE_DADT=${USE_DADT:-false}
DADT_GAMMA_LIST=${DADT_GAMMA_LIST:-"0.0,0.5,1.0,1.5,2.0"}
DADT_ALPHA_LIST=${DADT_ALPHA_LIST:-"1.5,2.0,2.5"}

# ============================================================
# 参数校验
# ============================================================
VALID_TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK} " ]]; then
    echo "❌ 错误: 无效任务 '$TASK'"
    echo "可用任务: ${VALID_TASKS[@]}"
    exit 1
fi

# 校验网格参数配置
if [ "$USE_PREDEFINED_GRID" = false ] && [ -z "$GRID_PARAMS" ] && [ "$RESUME_FROM_CHECKPOINT" = false ]; then
    echo "⚠️  警告: 未指定网格参数，将使用默认平衡网格"
    USE_PREDEFINED_GRID=true
    PREDEFINED_GRID_TYPE="balanced"
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
echo "DSCLR 网格搜索评测系统"
echo "============================================================"
echo "编码器类型: ${ENCODER_TYPE}"
echo "模型: ${MODEL_NAME}"
echo "嵌入维度: ${EMBED_DIM}"
echo "任务: ${TASK}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "批处理: ${BATCH_SIZE}"
echo "种子: ${SEED}"

if [ "$USE_DADT" = true ]; then
    echo "网格类型: DADT (Distribution-Aware Dynamic Threshold)"
    echo "Gamma列表: ${DADT_GAMMA_LIST}"
    echo "Alpha列表: ${DADT_ALPHA_LIST}"
elif [ "$USE_PREDEFINED_GRID" = true ]; then
    echo "网格类型: 预定义-${PREDEFINED_GRID_TYPE}"
elif [ -n "$GRID_PARAMS" ]; then
    echo "网格类型: 自定义"
    echo "参数组合: ${GRID_PARAMS}"
fi

if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    echo "断点续跑: 是"
    echo "检查点: ${CHECKPOINT_FILE}"
fi

[ "$VERBOSE" = true ] && echo "日志: 详细模式"
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
    --use_cache True"

# 添加网格搜索参数（DADT模式优先）
if [ "$USE_DADT" = true ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --dadt --gamma_list ${DADT_GAMMA_LIST} --alpha_list ${DADT_ALPHA_LIST}"
elif [ "$USE_PREDEFINED_GRID" = true ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --predefined_grid ${PREDEFINED_GRID_TYPE}"
elif [ -n "$GRID_PARAMS" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --grid_params ${GRID_PARAMS}"
fi

# 添加断点续跑参数
if [ "$RESUME_FROM_CHECKPOINT" = true ] && [ -n "$CHECKPOINT_FILE" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --resume_checkpoint ${CHECKPOINT_FILE}"
fi

# ============================================================
# 执行评测
# ============================================================
cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo ""
echo "🚀 启动网格搜索评测..."
echo ""

python eval/engine_grid_search.py ${PYTHON_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ 网格搜索评测完成！"
    echo "📁 结果保存于: ${OUTPUT_DIR}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 评测失败，退出码: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
