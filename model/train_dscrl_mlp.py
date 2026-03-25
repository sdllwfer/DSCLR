#!/usr/bin/env python
"""
DSCLR MLP 训练脚本 - 通用版本
支持 RepLLaMA、Mistral、BGE 等多种编码器
"""

import os
import sys
import argparse
import json
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from dsclr_mlp import DSCLR_MLP


class DSCLRDataset(Dataset):
    """DSCLR 数据集加载器"""
    def __init__(self, cache_path, num_neg=15):
        raw_data = torch.load(cache_path, weights_only=False)
        
        # 处理不同的键名格式
        if 'q_plus_embeddings' in raw_data:
            # 新格式 (v4)
            self.data = {
                'q_plus': raw_data['q_plus_embeddings'].float(),
                'q_minus': raw_data['q_minus_embeddings'].float(),
                'pos': raw_data['pos_embeddings'].float(),
                'neg': raw_data['neg_embeddings'].float()
            }
            self.num_neg = raw_data.get('num_neg_per_query', num_neg)
        else:
            # 旧格式
            self.data = {
                'q_plus': raw_data['q_plus'].float(),
                'q_minus': raw_data['q_minus'].float(),
                'pos': raw_data['pos'].float(),
                'neg': raw_data['neg'].float()
            }
            self.num_neg = num_neg
        
        self.num_queries = len(self.data['q_plus'])
        
    def __len__(self):
        return self.num_queries
    
    def __getitem__(self, idx):
        q_plus = self.data['q_plus'][idx]
        q_minus = self.data['q_minus'][idx]
        
        pos_start = idx
        pos = self.data['pos'][pos_start]
        
        neg_start = idx * self.num_neg
        neg = self.data['neg'][neg_start:neg_start + self.num_neg]
        
        return {
            'q_plus': q_plus,
            'q_minus': q_minus,
            'pos': pos,
            'neg': neg
        }


def collate_fn(batch):
    q_plus = torch.stack([item['q_plus'] for item in batch])
    q_minus = torch.stack([item['q_minus'] for item in batch])
    pos = torch.stack([item['pos'] for item in batch])
    neg = torch.stack([item['neg'] for item in batch])
    
    return {
        'q_plus': q_plus,
        'q_minus': q_minus,
        'pos': pos,
        'neg': neg
    }


def compute_scores(q_plus, q_minus, pos, neg, mlp, neg_mask=None, encoder_type='bge'):
    """
    计算 DSCLR 得分
    """
    batch_size = q_plus.size(0)
    num_neg = neg.size(1)
    
    q_plus_expanded = q_plus.unsqueeze(1)
    q_minus_expanded = q_minus.unsqueeze(1)
    pos_expanded = pos.unsqueeze(2)
    neg_transposed = neg.transpose(1, 2)
    
    S_base_pos = torch.bmm(q_plus_expanded, pos_expanded).squeeze(-1)
    S_base_neg = torch.bmm(q_plus_expanded, neg_transposed).squeeze(1)
    
    alpha, tau = mlp(q_minus, encoder_type=encoder_type)
    
    tau_expanded = tau.unsqueeze(1)
    alpha_expanded = alpha.unsqueeze(1)
    
    penalty_pos = torch.zeros_like(S_base_pos)
    penalty_neg = torch.relu(S_base_neg - tau_expanded)
    
    if neg_mask is not None:
        penalty_neg = penalty_neg * neg_mask
    
    S_final_pos = S_base_pos - alpha_expanded * penalty_pos
    S_final_neg = S_base_neg - alpha_expanded * penalty_neg
    
    S_final = torch.cat([S_final_pos, S_final_neg], dim=1)
    
    return S_final, alpha, tau, penalty_pos, penalty_neg


def dsclr_loss(S_final, alpha, penalty_pos=None, temperature=0.1, lambda_reg=0.0, encoder_type='bge'):
    """
    DSCLR 损失函数
    """
    logits = S_final / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    nce_loss = F.cross_entropy(logits, labels)
    
    if encoder_type == 'repllama':
        reg_weight = 10.0
        pos_protection = reg_weight * torch.mean(penalty_pos) if penalty_pos is not None else torch.tensor(0.0)
        alpha_reg = torch.tensor(0.0)
    elif encoder_type == 'mistral':
        reg_weight = 10.0
        pos_protection = reg_weight * torch.mean(penalty_pos) if penalty_pos is not None else torch.tensor(0.0)
        alpha_reg = torch.tensor(0.0)
    else:
        alpha_reg = lambda_reg * torch.mean(alpha ** 2)
        pos_protection = torch.tensor(0.0)
    
    total_loss = nce_loss + alpha_reg + pos_protection
    
    return total_loss, nce_loss, alpha_reg, pos_protection


def train_epoch(mlp, dataloader, optimizer, device, encoder_type='bge'):
    """训练一个 epoch"""
    mlp.train()
    total_loss = 0
    total_nce_loss = 0
    total_pos_protection = 0
    num_batches = 0
    all_penalty_pos = []
    all_penalty_neg = []
    all_alphas = []
    
    for batch in tqdm(dataloader, desc="Training"):
        q_plus = batch['q_plus'].to(device)
        q_minus = batch['q_minus'].to(device)
        pos = batch['pos'].to(device)
        neg = batch['neg'].to(device)
        
        neg_mask = (q_minus.abs().sum(dim=1) > 0).float()
        neg_mask = neg_mask.unsqueeze(1)
        neg_mask = neg_mask.expand(-1, neg.size(1))
        
        S_final, alpha, tau, penalty_pos, penalty_neg = compute_scores(
            q_plus, q_minus, pos, neg, mlp, neg_mask, encoder_type=encoder_type
        )
        
        total_loss_batch, nce_loss, alpha_reg, pos_protection = dsclr_loss(
            S_final, alpha, penalty_pos=penalty_pos,
            temperature=0.1, lambda_reg=0.0, encoder_type=encoder_type
        )
        
        if torch.isnan(total_loss_batch):
            continue
        
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_nce_loss += nce_loss.item()
        total_pos_protection += pos_protection.item()
        all_penalty_pos.append(penalty_pos.mean().item())
        all_penalty_neg.append(penalty_neg.mean().item())
        all_alphas.append(alpha.mean().item())
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_nce_loss = total_nce_loss / num_batches if num_batches > 0 else 0
    avg_pos_protection = total_pos_protection / num_batches if num_batches > 0 else 0
    avg_penalty_pos = sum(all_penalty_pos) / len(all_penalty_pos) if all_penalty_pos else 0
    avg_penalty_neg = sum(all_penalty_neg) / len(all_penalty_neg) if all_penalty_neg else 0
    
    return avg_loss, avg_nce_loss, avg_penalty_pos, avg_penalty_neg, avg_pos_protection


@torch.no_grad()
def validate(mlp, dataloader, device, encoder_type='bge'):
    """验证 - 收集所有样本的 alpha/tau 以计算标准差"""
    mlp.eval()
    total_loss = 0
    total_nce_loss = 0
    total_pos_protection = 0
    num_batches = 0
    all_alphas = []
    all_taus = []
    all_penalty_pos = []
    all_penalty_neg = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        q_plus = batch['q_plus'].to(device)
        q_minus = batch['q_minus'].to(device)
        pos = batch['pos'].to(device)
        neg = batch['neg'].to(device)
        
        neg_mask = (q_minus.abs().sum(dim=1) > 0).float()
        neg_mask = neg_mask.unsqueeze(1)
        neg_mask = neg_mask.expand(-1, neg.size(1))
        
        S_final, alpha, tau, penalty_pos, penalty_neg = compute_scores(
            q_plus, q_minus, pos, neg, mlp, neg_mask, encoder_type=encoder_type
        )
        
        total_loss_batch, nce_loss, alpha_reg, pos_protection = dsclr_loss(
            S_final, alpha, penalty_pos=penalty_pos,
            temperature=0.1, lambda_reg=0.0, encoder_type=encoder_type
        )
        
        if not torch.isnan(total_loss_batch):
            total_loss += total_loss_batch.item()
            total_nce_loss += nce_loss.item()
            total_pos_protection += pos_protection.item()
            all_alphas.extend(alpha.cpu().numpy().tolist())
            all_taus.extend(tau.cpu().numpy().tolist())
            all_penalty_pos.append(penalty_pos.mean().item())
            all_penalty_neg.append(penalty_neg.mean().item())
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_nce_loss = total_nce_loss / num_batches if num_batches > 0 else 0
    avg_pos_protection = total_pos_protection / num_batches if num_batches > 0 else 0
    avg_alpha = sum(all_alphas) / len(all_alphas) if all_alphas else 0
    avg_tau = sum(all_taus) / len(all_taus) if all_taus else 0
    std_alpha = np.std(all_alphas) if all_alphas else 0
    std_tau = np.std(all_taus) if all_taus else 0
    avg_penalty_pos = sum(all_penalty_pos) / len(all_penalty_pos) if all_penalty_pos else 0
    avg_penalty_neg = sum(all_penalty_neg) / len(all_penalty_neg) if all_penalty_neg else 0
    
    return avg_loss, avg_nce_loss, avg_alpha, avg_tau, std_alpha, std_tau, avg_penalty_pos, avg_penalty_neg, avg_pos_protection


def plot_and_save_curves(history, save_path):
    """绘制训练曲线 - 包含 Alpha/Tau 均值和标准差阴影"""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not installed, skipping plot generation")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Loss curves
    axes[0].plot(epochs, history['train_nce_loss'], 'b-', label='Train NCE Loss', linewidth=2)
    axes[0].plot(epochs, history['val_nce_loss'], 'r-', label='Val NCE Loss', linewidth=2)
    axes[0].set_ylabel('InfoNCE Loss')
    axes[0].set_title('DSCLR Training & Validation NCE Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Alpha with std
    avg_alpha = np.array(history['val_avg_alpha'])
    std_alpha = np.array(history['val_std_alpha'])
    axes[1].plot(epochs, avg_alpha, 'g-', label='Avg Predicted Alpha', linewidth=2)
    axes[1].fill_between(epochs, avg_alpha - std_alpha, avg_alpha + std_alpha, 
                         alpha=0.3, color='green', label='±1 Std')
    axes[1].set_ylabel('Alpha Value')
    axes[1].set_title('Average Predicted Penalty Multiplier (Alpha) with Std Dev')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Tau with std
    avg_tau = np.array(history['val_avg_tau'])
    std_tau = np.array(history['val_std_tau'])
    axes[2].plot(epochs, avg_tau, 'orange', linestyle='-', label='Avg Predicted Tau', linewidth=2)
    axes[2].fill_between(epochs, avg_tau - std_tau, avg_tau + std_tau, 
                         alpha=0.3, color='orange', label='±1 Std')
    axes[2].set_ylabel('Tau Value')
    axes[2].set_title('Average Predicted Tau Threshold with Std Dev')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Penalties
    axes[3].plot(epochs, history['val_penalty_pos'], 'm-', label='Pos Penalty', linewidth=2)
    axes[3].plot(epochs, history['val_penalty_neg'], 'c-', label='Neg Penalty', linewidth=2)
    axes[3].set_xlabel('Epochs')
    axes[3].set_ylabel('Penalty Value')
    axes[3].set_title('Average Positive vs Negative Penalties')
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 训练曲线已保存: {save_path}")


def train():
    parser = argparse.ArgumentParser(description='DSCLR MLP Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='train/output', help='Output directory')
    parser.add_argument('--cache_path', type=str, default='dataset/FollowIR_train/embeddings/dsclr_train_embeddings.pt', help='Training data cache path')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_neg', type=int, default=15, help='Number of negatives')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--model_type', type=str, default='bge', choices=['bge', 'mistral', 'repllama'], help='Model type')
    args = parser.parse_args()
    
    DEVICE = f"cuda:{args.gpu}"
    CACHE_PATH = args.cache_path
    OUTPUT_DIR = args.output_dir
    NUM_NEG = args.num_neg
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    VAL_RATIO = args.val_ratio
    PATIENCE = args.patience
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    MODEL_TYPE = args.model_type
    
    # 根据模型类型设置 encoder_type
    encoder_type = MODEL_TYPE
    MODEL_SUFFIX = MODEL_TYPE
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("DSCLR MLP 训练配置")
    print("="*60)
    print(f"编码器: {MODEL_TYPE}")
    print(f"嵌入维度: {EMBED_DIM}")
    print(f"隐藏层维度: {HIDDEN_DIM}")
    print(f"GPU: {args.gpu}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"缓存路径: {CACHE_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)
    
    print("\n加载数据...")
    dataset = DSCLRDataset(CACHE_PATH, num_neg=NUM_NEG)
    total_size = len(dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"总数据量: {total_size}")
    print(f"训练集: {train_size}, 验证集: {val_size}")
    print(f"Batch size: {BATCH_SIZE}")
    
    print("\n初始化模型...")
    mlp = DSCLR_MLP(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=0.01)
    
    num_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP 参数量: {num_params / 1024:.1f} KB ({num_params / 1024 / 1024:.2f} MB)")
    
    writer = SummaryWriter(log_dir="./runs/dsclr_experiment_1") if HAS_TENSORBOARD else None
    
    history = {
        'train_loss': [],
        'train_nce_loss': [],
        'val_loss': [],
        'val_nce_loss': [],
        'val_avg_alpha': [],
        'val_std_alpha': [],
        'val_avg_tau': [],
        'val_std_tau': [],
        'val_penalty_pos': [],
        'val_penalty_neg': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n开始训练...")
    
    for epoch in range(EPOCHS):
        train_loss, train_nce_loss, train_penalty_pos, train_penalty_neg, train_pos_protection = train_epoch(
            mlp, train_loader, optimizer, DEVICE, encoder_type=encoder_type
        )
        
        val_loss, val_nce_loss, avg_alpha, avg_tau, std_alpha, std_tau, val_penalty_pos, val_penalty_neg, val_pos_protection = validate(
            mlp, val_loader, DEVICE, encoder_type=encoder_type
        )
        
        history['train_loss'].append(train_loss)
        history['train_nce_loss'].append(train_nce_loss)
        history['val_loss'].append(val_loss)
        history['val_nce_loss'].append(val_nce_loss)
        history['val_avg_alpha'].append(avg_alpha)
        history['val_std_alpha'].append(std_alpha)
        history['val_avg_tau'].append(avg_tau)
        history['val_std_tau'].append(std_tau)
        history['val_penalty_pos'].append(val_penalty_pos)
        history['val_penalty_neg'].append(val_penalty_neg)
        
        writer.add_scalar('Loss/Train', train_nce_loss, epoch) if writer else None
        writer.add_scalar('Loss/Val', val_nce_loss, epoch) if writer else None
        writer.add_scalar('Metrics/Avg_Alpha', avg_alpha, epoch) if writer else None
        writer.add_scalar('Metrics/Std_Alpha', std_alpha, epoch) if writer else None
        writer.add_scalar('Metrics/Avg_Tau', avg_tau, epoch) if writer else None
        writer.add_scalar('Metrics/Std_Tau', std_tau, epoch) if writer else None
        writer.add_scalar('Metrics/Penalty_Pos', val_penalty_pos, epoch) if writer else None
        writer.add_scalar('Metrics/Penalty_Neg', val_penalty_neg, epoch) if writer else None
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train NCE={train_nce_loss:.4f}, Val NCE={val_nce_loss:.4f}, Alpha={avg_alpha:.4f}±{std_alpha:.4f}, Tau={avg_tau:.4f}±{std_tau:.4f}, PosPen={val_penalty_pos:.4f}, NegPen={val_penalty_neg:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_filename = f"dsclr_best_mlp_{MODEL_SUFFIX}.pt"
            best_model_path = os.path.join(OUTPUT_DIR, model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_alpha': avg_alpha,
                'val_tau': avg_tau,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
            }, best_model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n🛑 触发 Early Stopping! 连续{PATIENCE}轮未提升，停止训练。")
            break
    
    if writer:
        writer.close()
    
    plot_and_save_curves(history, f"{OUTPUT_DIR}/dsclr_training_curves_{MODEL_SUFFIX}.png")
    
    # 保存训练历史到 JSON
    history_file = f"{OUTPUT_DIR}/training_history_{MODEL_SUFFIX}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"📊 训练历史已保存: {history_file}")
    
    print(f"\n训练完成! Best Val Loss: {best_val_loss:.4f}")
    print(f"最佳模型已保存: {OUTPUT_DIR}/{model_filename}")


if __name__ == "__main__":
    train()
