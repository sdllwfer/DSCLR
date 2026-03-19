"""
FollowIR 评测系统单元测试
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mteb._evaluators.retrieval_metrics import get_rank_from_dict, rank_score


class TestMetrics(unittest.TestCase):
    """测试指标计算函数"""
    
    def test_get_rank_from_dict(self):
        """测试排名获取"""
        rank_dict = {
            "doc1": 1.0,
            "doc2": 0.8,
            "doc3": 0.5
        }
        
        rank, score = get_rank_from_dict(rank_dict, "doc1")
        self.assertEqual(rank, 1)
        self.assertEqual(score, 1.0)
        
        rank, score = get_rank_from_dict(rank_dict, "doc2")
        self.assertEqual(rank, 2)
        self.assertEqual(score, 0.8)
    
    def test_rank_score_improvement(self):
        """测试排名数字变小（变好）时的分数计算"""
        score = rank_score({"og_rank": 10, "new_rank": 1})
        self.assertLess(score, 0)
    
    def test_rank_score_degradation(self):
        """测试排名数字变大（变差）时的分数计算"""
        score = rank_score({"og_rank": 1, "new_rank": 10})
        self.assertGreater(score, 0)
    
    def test_rank_score_no_change(self):
        """测试排名不变时的分数计算"""
        score = rank_score({"og_rank": 5, "new_rank": 5})
        self.assertEqual(score, 0.0)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def test_task_mapping(self):
        """测试任务名称映射"""
        from eval.metrics.evaluator import DataLoader
        
        loader = DataLoader("Core17InstructionRetrieval")
        self.assertIn("jhu-clsp", loader.dataset_path)
        
        loader = DataLoader("Robust04InstructionRetrieval")
        self.assertIn("jhu-clsp", loader.dataset_path)
        
        loader = DataLoader("News21InstructionRetrieval")
        self.assertIn("jhu-clsp", loader.dataset_path)


if __name__ == "__main__":
    unittest.main()
