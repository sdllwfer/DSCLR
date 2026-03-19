import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 定义我们的外挂门控网络 (DSCLR MLP)
# ==========================================
class DSCLRGating(nn.Module):
    def __init__(self, hidden_dim=1024): # 注意：这里必须和 BGE 的维度对齐
        super(DSCLRGating, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 输出 alpha 和 tau
        )
        
    def forward(self, v_q_minus):
        is_empty = (v_q_minus.norm(p=2, dim=-1, keepdim=True) < 1e-6).float()
        out = self.mlp(v_q_minus)
        alpha = F.softplus(out[:, 0]) 
        tau = torch.sigmoid(out[:, 1]) 
        alpha = alpha * (1.0 - is_empty.squeeze(-1)) 
        return alpha, tau

# ==========================================
# 2. 初始化基座模型与 MLP
# ==========================================
print("📥 正在加载基座检索模型 BGE-large...")
# device='cuda' 会自动把模型加载到你的 3090 上
encoder = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')

print("📥 正在初始化 DSCLR 外挂 MLP...")
mlp_gating = DSCLRGating(hidden_dim=1024).to('cuda')
# 注意：在真实的实验中，这里应该 load_state_dict() 加载你训练好的 MLP 权重
# 这里我们先用随机初始化的权重走通前向传播

# ==========================================
# 3. 模拟一次真实的重排流水线 (Inference)
# ==========================================
def run_dsclr_reranking():
    # 假设这是前端用 GPT-4 提纯出来的数据
    q_plus = "advantages of self-driving cars, improved traffic safety"
    q_minus = "Elon Musk, Tesla crashes"
    
    # 假设这是我们要重排的 3 篇候选文档
    docs = [
        "Self-driving cars can significantly reduce traffic accidents caused by human error.", # 完美命中 Q+
        "Elon Musk stated that Tesla's autopilot is the future, despite some crashes.",     # 命中 Q+，但严重违规 (触碰 Q-)
        "The Chunnel construction was delayed due to budget issues."                        # 完全不相关
    ]
    
    # --- A. 提取特征 (Frozen Base Model) ---
    print("\n🚀 开始提取向量特征...")
    # normalize_embeddings=True 极其重要！它保证了后续的点积等于余弦相似度
    v_q_plus = encoder.encode(q_plus, convert_to_tensor=True, normalize_embeddings=True)
    v_q_minus = encoder.encode(q_minus, convert_to_tensor=True, normalize_embeddings=True)
    v_docs = encoder.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
    
    # 把向量升维以适应批量计算 (1, 1024)
    v_q_plus = v_q_plus.unsqueeze(0)
    v_q_minus = v_q_minus.unsqueeze(0)
    
    # --- B. 计算基础得分 ---
    # 矩阵乘法：(1, 1024) @ (1024, 3) -> (1, 3)
    s_base = torch.matmul(v_q_plus, v_docs.T).squeeze(0)
    s_neg = torch.matmul(v_q_minus, v_docs.T).squeeze(0)
    
    # --- C. 外挂 MLP 介入 ---
    # 根据 Q- 的语义，动态预测惩罚参数
    alpha, tau = mlp_gating(v_q_minus)
    print(f"⚖️ MLP 预测参数 -> Alpha (惩罚力度): {alpha.item():.4f}, Tau (容忍底线): {tau.item():.4f}")
    
    # --- D. DSCLR 终极打分公式 ---
    # S_final = S_base - alpha * ReLU(S_neg - tau)
    penalty = alpha * F.relu(s_neg - tau)
    s_final = s_base - penalty
    
    # --- 打印结果对比 ---
    print("\n📊 重排打分结果对比:")
    for i, doc in enumerate(docs):
        print(f"\n文档 {i+1}: {doc[:50]}...")
        print(f"  ▶ 基础正向得分 (S_base): {s_base[i].item():.4f}")
        print(f"  ▶ 触碰负向得分 (S_neg) : {s_neg[i].item():.4f}")
        print(f"  ▶ 受到扣分惩罚 (Penalty): -{penalty[i].item():.4f}")
        print(f"  🏆 DSCLR 最终得分     : {s_final[i].item():.4f}")

if __name__ == "__main__":
    with torch.no_grad(): # 推理阶段不需要算梯度
        run_dsclr_reranking()