"""
LAP (Lightweight Asymmetric Projection) 逆向对比损失

核心思想：
- 让负向查询向量远离正样本（好文档）
- 让负向查询向量靠近负样本（踩雷文档）
- 这与传统对比学习方向相反！

损失函数设计：
1. Push Away Loss: 惩罚 q_neg_proj 与 d_pos 的相似度超过 margin_pos
2. Pull Closer Loss: 惩罚 q_neg_proj 与 d_neg 的相似度低于 margin_neg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPContrastiveLoss(nn.Module):
    """
    LAP 逆向对比损失函数 (非对称加权版本)
    
    目标：让负向指令向量远离好文档，主动靠近踩雷文档
    
    Args:
        margin_pos: 正样本避让安全线（相似度上限）
                    默认 0.1，即要求 q_neg_proj 与 d_pos 相似度 < 0.1
        margin_neg: 负样本贴合目标线（相似度下限）
                    默认 0.8，即要求 q_neg_proj 与 d_neg 相似度 > 0.8
        use_in_batch: 是否使用 In-Batch Hard Negatives
                      默认 True，利用批次内其他样本的负样本
        lambda_push: Push Away 损失的权重，默认 1.4 (提高优先级，先让绿色曲线下降)
        lambda_pull: Pull Closer 损失的权重，默认 0.6 (降低优先级，防止友军被误伤)
        temperature: 温度系数，用于放大难负样本差异，默认 0.05
        margin_pos: 正样本安全线，默认 0.05 (必须要求正样本分数 < 0.05)
        margin_neg: 难负样本目标线，默认 0.3 (只要难负样本分数 > 0.3 即可，不要追求 1.0)
        use_orthogonal_reg: 是否使用正交约束正则化，默认 True
        orthogonal_reg_weight: 正交正则化权重，默认 0.01
    """
    
    def __init__(
        self,
        margin_pos: float = 0.05,
        margin_neg: float = 0.3,
        use_in_batch: bool = True,
        lambda_push: float = 1.4,
        lambda_pull: float = 0.6,
        temperature: float = 0.05,
        use_orthogonal_reg: bool = True,
        orthogonal_reg_weight: float = 0.01
    ):
        super().__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.use_in_batch = use_in_batch
        self.lambda_push = lambda_push
        self.lambda_pull = lambda_pull
        self.temperature = temperature
        self.use_orthogonal_reg = use_orthogonal_reg
        self.orthogonal_reg_weight = orthogonal_reg_weight
    
    def forward(
        self,
        q_neg_proj: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor
    ) -> tuple:
        """
        计算逆向对比损失（防崩塌版本）
        
        Args:
            q_neg_proj: 投影后的负向查询向量
                Shape: [batch_size, hidden_dim]
                要求: 已 L2 归一化
            d_pos: 正样本文档向量（好文档）
                Shape: [batch_size, hidden_dim]
                要求: 已 L2 归一化
            d_neg: 负样本文档向量（踩雷文档）
                Shape: [batch_size, hidden_dim]
                要求: 已 L2 归一化
        
        Returns:
            total_loss: 总损失
            loss_pos: Push Away 损失（远离好文档）
            loss_neg: Pull Closer 损失（靠近踩雷文档）
            metrics: 监控指标字典
        """
        batch_size = q_neg_proj.size(0)
        hidden_dim = q_neg_proj.size(1)
        device = q_neg_proj.device
        
        # ============================================================
        # 1. Push Away Loss: 让 q_neg_proj 远离 d_pos
        # ============================================================
        sim_pos = torch.matmul(q_neg_proj, d_pos.T)
        diag_sim_pos = torch.diag(sim_pos)  # [batch_size]
        loss_pos = torch.mean(F.relu(diag_sim_pos - self.margin_pos))
        
        # ============================================================
        # 2. Pull Closer Loss: 让 q_neg_proj 靠近 d_neg
        # ============================================================
        sim_neg = torch.matmul(q_neg_proj, d_neg.T)
        
        if self.use_in_batch:
            diag_sim_neg = torch.diag(sim_neg)
            loss_neg = torch.mean(F.relu(self.margin_neg - diag_sim_neg))
        else:
            diag_sim_neg = torch.diag(sim_neg)
            loss_neg = torch.mean(F.relu(self.margin_neg - diag_sim_neg))
        
        # ============================================================
        # 3. 正交约束正则化 (Orthogonal Regularization)
        # ============================================================
        reg_loss = torch.tensor(0.0, device=device)
        if self.use_orthogonal_reg:
            lap_matrix = q_neg_proj
            gramian = torch.matmul(lap_matrix, lap_matrix.t())
            identity = torch.eye(batch_size, device=device)
            reg_loss = torch.mean((gramian - identity) ** 2)
        
        # ============================================================
        # 4. 随机正交锚点 (Random Orthogonal Anchoring)
        # ============================================================
        ortho_loss = torch.tensor(0.0, device=device)
        num_anchors = 4
        if self.training:
            with torch.no_grad():
                random_anchors = torch.randn(batch_size, num_anchors, hidden_dim, device=device)
                random_anchors = F.normalize(random_anchors, p=2, dim=-1)
            
            for i in range(num_anchors):
                anchor = random_anchors[:, i, :]
                sim_anchor = torch.sum(q_neg_proj * anchor, dim=-1)
                ortho_loss += torch.mean(sim_anchor ** 2)
            ortho_loss = ortho_loss / num_anchors
        
        # ============================================================
        # 5. 非对称加权总损失
        # ============================================================
        total_loss = (
            self.lambda_push * loss_pos + 
            self.lambda_pull * loss_neg + 
            self.orthogonal_reg_weight * reg_loss +
            0.1 * ortho_loss
        )
        
        # ============================================================
        # 6. 监控指标
        # ============================================================
        max_sim_neg_per_row = sim_neg.max(dim=1).values
        score_gap = (max_sim_neg_per_row - diag_sim_pos).mean().item()
        
        metrics = {
            'loss_pos': loss_pos.item(),
            'loss_neg': loss_neg.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'ortho_loss': ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss,
            'score_gap': score_gap,
            'diag_sim_pos_mean': diag_sim_pos.mean().item(),
            'diag_sim_neg_mean': diag_sim_neg.mean().item()
        }
        
        return total_loss, loss_pos, loss_neg, metrics
    
    def extra_repr(self) -> str:
        return (f"margin_pos={self.margin_pos}, margin_neg={self.margin_neg}, "
                f"use_in_batch={self.use_in_batch}, lambda_push={self.lambda_push}, "
                f"lambda_pull={self.lambda_pull}, use_orthogonal_reg={self.use_orthogonal_reg}")


class LAPContrastiveLossWithHardMining(nn.Module):
    """
    带困难样本挖掘的 LAP 逆向对比损失
    
    在 In-Batch 基础上，额外关注最困难的样本
    
    Args:
        margin_pos: 正样本避让安全线
        margin_neg: 负样本贴合目标线
        hard_ratio: 困难样本比例（0.0-1.0）
    """
    
    def __init__(
        self,
        margin_pos: float = 0.1,
        margin_neg: float = 0.8,
        hard_ratio: float = 0.3
    ):
        super().__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.hard_ratio = hard_ratio
    
    def forward(
        self,
        q_neg_proj: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor
    ) -> tuple:
        """
        计算带困难样本挖掘的逆向对比损失
        """
        batch_size = q_neg_proj.size(0)
        num_hard = max(1, int(batch_size * self.hard_ratio))
        
        # Push Away Loss
        sim_pos = torch.matmul(q_neg_proj, d_pos.T)
        diag_sim_pos = torch.diag(sim_pos)
        
        # 困难样本挖掘：选择相似度最高的（最难推开的）
        _, hard_indices_pos = torch.topk(diag_sim_pos, num_hard)
        hard_sim_pos = diag_sim_pos[hard_indices_pos]
        loss_pos = torch.mean(F.relu(hard_sim_pos - self.margin_pos))
        
        # Pull Closer Loss
        sim_neg = torch.matmul(q_neg_proj, d_neg.T)
        diag_sim_neg = torch.diag(sim_neg)
        
        # 困难样本挖掘：选择相似度最低的（最难拉近的）
        _, hard_indices_neg = torch.topk(diag_sim_neg, num_hard, largest=False)
        hard_sim_neg = diag_sim_neg[hard_indices_neg]
        loss_neg = torch.mean(F.relu(self.margin_neg - hard_sim_neg))
        
        total_loss = loss_pos + loss_neg
        
        return total_loss, loss_pos, loss_neg
