import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCLR_MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
        self.net[-1].bias.data[1] = 2.0

    def forward(self, v_q_minus):
        out = self.net(v_q_minus)
        
        alpha_raw = out[:, 0]
        tau_raw = out[:, 1] - 1.0
        
        alpha = F.softplus(alpha_raw + 1.0)
        alpha = torch.clamp(alpha, min=0.001, max=100.0)
        
        tau = torch.sigmoid(tau_raw)
        tau = torch.clamp(tau, min=0.3, max=0.95)
        
        return alpha, tau
