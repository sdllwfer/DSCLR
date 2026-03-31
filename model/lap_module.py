"""
LAP (Lightweight Asymmetric Projection) 模块

核心功能：对负向查询向量进行空间投影，使其远离正样本、靠近负样本

设计原则：
1. 单层无偏置线性变换，参数量极小
2. 单位矩阵初始化，保证 Epoch 0 输入输出等价
3. L2 归一化输出，确保余弦相似度空间一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPProjection(nn.Module):
    """
    Lightweight Asymmetric Projection Module
    
    对负向查询向量 (q_neg_emb) 进行空间扭曲投影
    
    Args:
        hidden_dim: 输入向量维度 (如 4096 for RepLLaMA, 1024 for BGE)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 单层无偏置线性变换
        # 参数量: hidden_dim * hidden_dim (如 4096*4096 = 16M 参数)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 核心Trick: 单位矩阵初始化
        # 保证 Epoch 0 时，输入输出等价，防止初始 Loss 爆炸
        with torch.no_grad():
            nn.init.eye_(self.proj.weight)
    
    def forward(self, q_neg_emb: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        """
        前向传播：投影 + L2归一化

        Args:
            q_neg_emb: 负向查询向量
                Shape: [batch_size, hidden_dim]
                要求: 输入应已 L2 归一化
            return_raw: 是否返回归一化前的原始投影向量

        Returns:
            q_neg_proj: 投影后的负向查询向量
                Shape: [batch_size, hidden_dim]
                保证: 输出已 L2 归一化
            raw_q_neg_proj: 归一化前的投影向量 (if return_raw=True)
                Shape: [batch_size, hidden_dim]
        """
        weight_dtype = self.proj.weight.dtype
        q_neg_emb = q_neg_emb.to(weight_dtype)
        raw_q_neg_proj = self.proj(q_neg_emb)

        if return_raw:
            return raw_q_neg_proj

        q_neg_proj = F.normalize(raw_q_neg_proj, p=2, dim=-1)
        return q_neg_proj
    
    def get_weight(self) -> torch.Tensor:
        """获取投影矩阵权重（用于分析和可视化）"""
        return self.proj.weight.data
    
    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, params={self.hidden_dim ** 2:,}"
