import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCLR_MLP(nn.Module):
    """
    DSCLR 动态门控网络
    
    支持多种输入维度：
    - BGE-large: 1024 维
    - E5-Mistral-7B: 4096 维
    
    隐藏层维度根据输入维度自适应调整，保证充足的表达能力
    """
    def __init__(self, input_dim=1024, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        
        # 根据输入维度自适应设置隐藏层维度
        if hidden_dim is None:
            # 4096 维输入使用更大的隐藏层 (512)
            # 1024 维输入使用标准隐藏层 (256)
            if input_dim >= 4096:
                hidden_dim = 512
            elif input_dim >= 1024:
                hidden_dim = 256
            else:
                hidden_dim = 128
        
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
        # tau 偏置初始化：-0.2 使得初始 tau ≈ 0.45
        # S_base_neg 的范围是 [0.35, 0.63]，均值 0.44
        # 初始 tau 应该略低于均值，惩罚那些过于相似的负样本
        self.net[-1].bias.data[0] = 0.0   # alpha 偏置
        self.net[-1].bias.data[1] = -0.2  # tau 偏置，初始 tau ≈ 0.45

    def forward(self, v_q_minus, encoder_type='bge'):
        """
        支持 BGE、Mistral 和 RepLLaMA 三种模式的动态参数预测
        
        Args:
            v_q_minus: Q- 查询的 embedding
            encoder_type: 'bge'、'mistral' 或 'repllama'，决定使用哪种参数约束策略
        """
        out = self.net(v_q_minus)
        
        alpha_raw = out[:, 0]
        tau_raw = out[:, 1]
        
        if encoder_type == 'repllama':
            # ========== RepLLaMA 模式：黄金中心点对齐 ==========
            # 动态 Alpha 映射：上限为 4.0，当网络初始输出为 0 时，sigmoid(0)=0.5，此时 Alpha = 4.0 * 0.5 = 2.0（完美命中黄金中心点）
            alpha = 4.0 * torch.sigmoid(alpha_raw)

            # 动态 Tau 映射：搜索空间为 [0.50, 0.90]，跨度为 0.40，当网络初始输出为 0 时，sigmoid(0)=0.5，此时 Tau = 0.50 + (0.40 * 0.5) = 0.70（完美命中黄金中心点）
            tau = 0.50 + 0.40 * torch.sigmoid(tau_raw)
        elif encoder_type == 'mistral':
            # ========== Mistral 模式：保持原有配置 ==========
            # Alpha 最大火力锁定在 1.2
            alpha = 1.2 * torch.sigmoid(alpha_raw)
            
            # Tau 底盘抬高到 0.45
            # Tau 范围: [0.45, 0.85]
            tau = 0.45 + 0.4 * torch.sigmoid(tau_raw)
        else:
            # ========== BGE 模式：保持原有配置 ==========
            # Alpha: Scaled Sigmoid 锁定在 (0, 3.0)
            alpha = 3.0 * torch.sigmoid(alpha_raw)
            
            # Tau: Sigmoid 激活，映射到 (0.3, 0.7)
            tau = torch.sigmoid(tau_raw)
            tau = torch.clamp(tau, min=0.3, max=0.7)
        
        return alpha, tau
