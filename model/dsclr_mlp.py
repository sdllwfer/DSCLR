import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCLR_MLP(nn.Module):
    """
    DSCLR 动态参数预测器

    支持两种调用方式：
    1. 单参数模式: mlp(q_minus, encoder_type='bge') -> 用于 train_dscrl_mlp.py 和 engine_dscrl.py
    2. 双参数模式: mlp(q_pos_emb, q_neg_proj) -> 用于 train_hybrid.py 和 deir_hybrid_retriever.py

    MLP 输出两个标量:
    - alpha: 惩罚力度，范围 [0, MAX_ALPHA]
    - tau: 容忍底线，范围 [TAU_MIN, TAU_MAX]
    """
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )

        self.mlp[-1].bias.data[0] = 0.5
        self.mlp[-1].bias.data[1] = 0.5

    def forward(self, *args, encoder_type='bge', **kwargs):
        """
        支持两种调用方式：
        
        方式1 (单参数模式):
            alpha, tau = mlp(q_minus, encoder_type='bge')
            Args:
                q_minus: 负向意图特征 [batch_size, input_dim]
                encoder_type: 编码器类型 ('bge', 'mistral', 'repllama')
        
        方式2 (双参数模式):
            alpha, tau = mlp(q_pos_emb, q_neg_proj)
            Args:
                q_pos_emb: 正向意图特征 [batch_size, input_dim]
                q_neg_proj: 投影后的负向意图特征 [batch_size, input_dim]

        Returns:
            alpha: 惩罚力度 [batch_size]
            tau: 容忍底线 [batch_size]
        """
        if len(args) == 1:
            q_minus = args[0]
            return self._forward_single(q_minus, encoder_type)
        elif len(args) == 2:
            q_pos_emb, q_neg_proj = args
            return self._forward_dual(q_pos_emb, q_neg_proj)
        else:
            raise ValueError(f"Expected 1 or 2 positional arguments, got {len(args)}")

    def _forward_single(self, q_minus, encoder_type='bge'):
        """单参数模式：直接使用 q_minus 预测 alpha 和 tau"""
        raw_output = self.mlp(q_minus)

        raw_alpha = raw_output[:, 0]
        raw_tau = raw_output[:, 1]

        if encoder_type == 'repllama':
            MAX_ALPHA = 1.2
            TAU_MIN = 0.45
            TAU_MAX = 0.85
        elif encoder_type == 'mistral':
            MAX_ALPHA = 1.5
            TAU_MIN = 0.4
            TAU_MAX = 0.8
        else:
            MAX_ALPHA = 2.0
            TAU_MIN = 0.3
            TAU_MAX = 0.7

        alpha = torch.sigmoid(raw_alpha) * MAX_ALPHA
        tau = TAU_MIN + torch.sigmoid(raw_tau) * (TAU_MAX - TAU_MIN)

        return alpha, tau

    def _forward_dual(self, q_pos_emb, q_neg_proj):
        """双参数模式：使用 q_pos_emb 和 q_neg_proj 预测 alpha 和 tau
        
        输入：q_neg_proj（投影后的负向意图特征）
        注意：q_pos_emb 用于计算相关性，但当前设计直接使用 q_neg_proj 预测
        """
        raw_output = self.mlp(q_neg_proj)

        raw_alpha = raw_output[:, 0]
        raw_tau = raw_output[:, 1]

        MAX_ALPHA = 2.0
        TAU_MIN = 0.3
        TAU_MAX = 0.7

        alpha = torch.sigmoid(raw_alpha) * MAX_ALPHA
        tau = TAU_MIN + torch.sigmoid(raw_tau) * (TAU_MAX - TAU_MIN)

        return alpha, tau
