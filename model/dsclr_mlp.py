import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCLR_MLP(nn.Module):
    """
    基于语义相关度的认知路由器

    MLP 的输入是拼接后的特征：
    - q_neg_proj: 投影后的负向意图特征 (hidden_dim)
    - correlation: q_pos 与 q_neg_proj 的余弦相似度 (1)

    MLP 输出两个标量：
    - raw_alpha: 经 SoftPlus 激活得到 alpha（惩罚力度）
    - raw_tau: 经 Sigmoid 激活得到 tau（容忍底线）
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

        self.mlp[-1].bias.data[0] = 0.5
        self.mlp[-1].bias.data[1] = -0.5

    def forward(self, q_pos_emb: torch.Tensor, q_neg_proj: torch.Tensor):
        """
        Args:
            q_pos_emb: 正向意图特征，Shape: [batch_size, hidden_dim]
            q_neg_proj: 投影后的负向意图特征，Shape: [batch_size, hidden_dim]

        Returns:
            alpha: 惩罚力度，Shape: [batch_size]
            tau: 容忍底线，Shape: [batch_size]
        """
        correlation = F.cosine_similarity(q_pos_emb, q_neg_proj, dim=-1).unsqueeze(-1)

        mlp_input = torch.cat([q_neg_proj, correlation], dim=-1)

        raw_output = self.mlp(mlp_input)

        raw_alpha = raw_output[:, 0]
        raw_tau = raw_output[:, 1]

        # 使用 Sigmoid 限制 alpha 的范围在 (0, MAX_ALPHA)，防止模型通过无限放大惩罚来作弊
        MAX_ALPHA = 2.0
        alpha = torch.sigmoid(raw_alpha) * MAX_ALPHA

        tau = torch.sigmoid(raw_tau)

        return alpha, tau