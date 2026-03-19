"""
结果输出模块
负责生成 TREC 格式文件和评估报告
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TRECWriter:
    """TREC 格式文件写入器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str,
        run_name: str = "eval_run"
    ) -> str:
        """写入 TREC 格式文件
        
        Args:
            results: 检索结果 {qid: {doc_id: score}}
            filename: 输出文件名
            run_name: 运行名称
            
        Returns:
            输出文件路径
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for q_id, doc_scores in results.items():
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                for rank_idx, (doc_id, score) in enumerate(sorted_docs, start=1):
                    f.write(f"{q_id} Q0 {doc_id} {rank_idx} {score:.6f} {run_name}\n")
        
        logger.info(f"💾 TREC 文件已保存: {output_path}")
        return output_path
    
    def write_og(self, results_og: Dict[str, Dict[str, float]], task_name: str, run_name: str = "eval") -> str:
        """写入 og 查询的 TREC 文件"""
        filename = f"run_{task_name}_og.trec"
        return self.write(results_og, filename, f"{run_name}_og")
    
    def write_changed(self, results_changed: Dict[str, Dict[str, float]], task_name: str, run_name: str = "eval") -> str:
        """写入 changed 查询的 TREC 文件"""
        filename = f"run_{task_name}_changed.trec"
        return self.write(results_changed, filename, f"{run_name}_changed")


class TRECReader:
    """TREC 格式文件读取器"""
    
    @staticmethod
    def read(trec_path: str) -> Dict[str, Dict[str, float]]:
        """读取 TREC 文件
        
        Args:
            trec_path: TREC 文件路径
            
        Returns:
            检索结果 {qid: {doc_id: score}}
        """
        results = {}
        
        with open(trec_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = parts[0]
                    docid = parts[2]
                    score = float(parts[4])
                    
                    if qid not in results:
                        results[qid] = {}
                    results[qid][docid] = score
        
        return results


class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_json_report(
        self,
        metrics: Dict[str, Any],
        task_name: str,
        model_name: str,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成 JSON 格式报告
        
        Args:
            metrics: 评估指标
            task_name: 任务名称
            model_name: 模型名称
            extra_info: 额外信息
            
        Returns:
            报告文件路径
        """
        report = {
            "task": task_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        if extra_info:
            report["extra_info"] = extra_info
        
        filename = f"results_{task_name}.json"
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 JSON 报告已保存: {output_path}")
        return output_path
    
    def generate_summary_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        output_filename: str = "summary.json"
    ) -> str:
        """生成汇总报告
        
        Args:
            all_results: 所有任务的评估结果
            output_filename: 输出文件名
            
        Returns:
            报告文件路径
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }
        
        for task_name, metrics in all_results.items():
            summary["tasks"][task_name] = metrics
        
        avg_pmrr = 0.0
        task_count = len(all_results)
        if task_count > 0:
            total_pmrr = sum(m.get("p-MRR", 0.0) for m in all_results.values())
            avg_pmrr = total_pmrr / task_count
        
        summary["average_p-MRR"] = avg_pmrr
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 汇总报告已保存: {output_path}")
        return output_path
    
    def generate_markdown_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        model_name: str,
        output_filename: str = "report.md"
    ) -> str:
        """生成 Markdown 格式报告
        
        Args:
            all_results: 所有任务的评估结果
            model_name: 模型名称
            output_filename: 输出文件名
            
        Returns:
            报告文件路径
        """
        lines = []
        lines.append(f"# FollowIR 评估报告")
        lines.append("")
        lines.append(f"**模型**: {model_name}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## 评测结果")
        lines.append("")
        lines.append("| 任务 | p-MRR | og nDCG@5 | changed nDCG@5 |")
        lines.append("|------|-------|-----------|----------------|")
        
        for task_name, metrics in all_results.items():
            pmrr = metrics.get("p-MRR", 0.0)
            og_ndcg = metrics.get("original", {}).get("ndcg_at_5", 0.0)
            changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)
            lines.append(f"| {task_name} | {pmrr:.4f} | {og_ndcg:.4f} | {changed_ndcg:.4f} |")
        
        lines.append("")
        avg_pmrr = sum(m.get("p-MRR", 0.0) for m in all_results.values()) / len(all_results) if all_results else 0
        lines.append(f"**平均 p-MRR**: {avg_pmrr:.4f}")
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.info(f"💾 Markdown 报告已保存: {output_path}")
        return output_path


class OutputManager:
    """输出管理器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.trec_writer = TRECWriter(os.path.join(output_dir, "trec"))
        self.report_generator = ReportGenerator(output_dir)
    
    def save_results(
        self,
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        task_name: str,
        model_name: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """保存所有结果
        
        Args:
            results_og: og 查询结果
            results_changed: changed 查询结果
            task_name: 任务名称
            model_name: 模型名称
            metrics: 评估指标
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {}
        
        trec_og_path = self.trec_writer.write_og(results_og, task_name, model_name)
        saved_files["trec_og"] = trec_og_path
        
        trec_changed_path = self.trec_writer.write_changed(results_changed, task_name, model_name)
        saved_files["trec_changed"] = trec_changed_path
        
        json_path = self.report_generator.generate_json_report(metrics, task_name, model_name)
        saved_files["json"] = json_path
        
        return saved_files
