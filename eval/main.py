"""
FollowIR 评测系统命令行入口
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_args(args):
    """验证命令行参数"""
    valid_tasks = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval"
    ]
    
    for task in args.tasks:
        if task not in valid_tasks:
            raise ValueError(f"无效任务: {task}. 可用任务: {', '.join(valid_tasks)}")
    
    if args.device == "cuda" and not os.path.exists("/dev/nvidiactl"):
        logger.warning("CUDA 不可用，将使用 CPU")
        args.device = "cpu"
    
    return args


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="FollowIR 评测系统 - 稠密检索模型评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评测单个任务
  python -m eval.main --model BAAI/bge-large-en-v1.5 --task Core17InstructionRetrieval
  
  # 评测多个任务
  python -m eval.main --model BAAI/bge-large-en-v1.5 --tasks Core17 Robust04 News21
  
  # 指定输出目录和批处理大小
  python -m eval.main --model BAAI/bge-large-en-v1.5 --task Core17 --output /tmp/eval --batch_size 128
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="模型名称或路径 (default: BAAI/bge-large-en-v1.5)"
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="Core17InstructionRetrieval",
        help="评测任务 (default: Core17InstructionRetrieval)"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[],
        help="批量评测多个任务"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录 (默认自动生成)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="计算设备 (default: cuda)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=64,
        help="批处理大小 (default: 64)"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="最大序列长度 (默认使用模型默认值)"
    )
    
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不进行向量归一化"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        args = validate_args(args)
    except ValueError as e:
        logger.error(f"参数验证失败: {e}")
        sys.exit(1)
    
    tasks = args.tasks if args.tasks else [args.task]
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/home/luwa/Documents/DSCLR/evaluation/{args.model.replace('/', '_')}_{timestamp}"
    
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FollowIR 评测系统")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"任务: {', '.join(tasks)}")
    logger.info(f"输出: {args.output}")
    logger.info(f"设备: {args.device}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"随机种子: {args.seed}")
    logger.info("=" * 60)
    
    from eval.engine import FollowIREvaluatorEngine, EvaluationRunner
    
    if len(tasks) == 1:
        evaluator = FollowIREvaluatorEngine(
            model_name=args.model,
            task_name=tasks[0],
            output_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
            max_seq_length=args.max_seq_length,
            seed=args.seed
        )
        results = evaluator.run()
    else:
        runner = EvaluationRunner(
            model_name=args.model,
            tasks=tasks,
            output_base_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
            max_seq_length=args.max_seq_length,
            seed=args.seed
        )
        results = runner.run_all()
    
    logger.info("✅ 评测完成!")
    logger.info(f"📁 结果保存至: {args.output}")


if __name__ == "__main__":
    main()
