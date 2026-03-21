import os
import argparse
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
    def __init__(self, cache_path, num_neg=15):
        self.data = torch.load(cache_path, weights_only=False)
        self.data['q_plus_embeddings'] = self.data['q_plus_embeddings'].float()
        self.data['q_minus_embeddings'] = self.data['q_minus_embeddings'].float()
        self.data['pos_embeddings'] = self.data['pos_embeddings'].float()
        self.data['neg_embeddings'] = self.data['neg_embeddings'].float()
        self.num_neg = num_neg
        self.num_queries = len(self.data['train_data_idx'])
        
    def __len__(self):
        return self.num_queries
    
    def __getitem__(self, idx):
        q_plus = self.data['q_plus_embeddings'][idx]
        q_minus = self.data['q_minus_embeddings'][idx]
        
        pos_start = idx
        pos = self.data['pos_embeddings'][pos_start]
        
        neg_start = idx * self.num_neg
        neg = self.data['neg_embeddings'][neg_start:neg_start + self.num_neg]
        
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


def compute_scores(q_plus, q_minus, pos, neg, mlp, neg_mask=None):
    batch_size = q_plus.size(0)
    num_neg = neg.size(1)
    
    q_plus_expanded = q_plus.unsqueeze(1)
    q_minus_expanded = q_minus.unsqueeze(1)
    pos_expanded = pos.unsqueeze(2)
    neg_transposed = neg.transpose(1, 2)
    
    S_base_pos = torch.bmm(q_plus_expanded, pos_expanded).squeeze(-1)
    S_base_neg = torch.bmm(q_plus_expanded, neg_transposed).squeeze(1)
    
    S_neg_pos = torch.bmm(q_minus_expanded, pos_expanded).squeeze(-1)
    S_neg_neg = torch.bmm(q_minus_expanded, neg_transposed).squeeze(1)
    
    alpha, tau = mlp(q_minus)
    
    tau_expanded = tau.unsqueeze(1)
    alpha_expanded = alpha.unsqueeze(1)
    
    penalty_pos = torch.relu(S_neg_pos - tau_expanded)
    penalty_neg = torch.relu(S_neg_neg - tau_expanded)
    
    if neg_mask is not None:
        penalty_neg = penalty_neg * neg_mask
    
    S_final_pos = S_base_pos - alpha_expanded * penalty_pos
    S_final_neg = S_base_neg - alpha_expanded * penalty_neg
    
    S_final = torch.cat([S_final_pos, S_final_neg], dim=1)
    
    return S_final, alpha, tau, penalty_pos, penalty_neg


def dsclr_loss(S_final, alpha, temperature=0.1, lambda_reg=0.0):
    logits = S_final / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    nce_loss = F.cross_entropy(logits, labels)
    alpha_reg_loss = lambda_reg * torch.mean(alpha)
    
    total_loss = nce_loss + alpha_reg_loss
    
    return total_loss, nce_loss, alpha_reg_loss


def train_epoch(mlp, dataloader, optimizer, device):
    mlp.train()
    total_loss = 0
    total_nce_loss = 0
    num_batches = 0
    all_penalty_pos = []
    all_penalty_neg = []
    
    for batch in tqdm(dataloader, desc="Training"):
        q_plus = batch['q_plus'].to(device)
        q_minus = batch['q_minus'].to(device)
        pos = batch['pos'].to(device)
        neg = batch['neg'].to(device)
        
        neg_mask = (q_minus.abs().sum(dim=1) > 0).float()
        neg_mask = neg_mask.unsqueeze(1)
        neg_mask = neg_mask.expand(-1, neg.size(1))
        
        S_final, alpha, tau, penalty_pos, penalty_neg = compute_scores(q_plus, q_minus, pos, neg, mlp, neg_mask)
        total_loss_batch, nce_loss, alpha_reg = dsclr_loss(S_final, alpha, temperature=0.1, lambda_reg=0.0)
        
        if torch.isnan(total_loss_batch):
            continue
        
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_nce_loss += nce_loss.item()
        all_penalty_pos.append(penalty_pos.mean().item())
        all_penalty_neg.append(penalty_neg.mean().item())
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_nce_loss = total_nce_loss / num_batches if num_batches > 0 else 0
    avg_penalty_pos = sum(all_penalty_pos) / len(all_penalty_pos) if all_penalty_pos else 0
    avg_penalty_neg = sum(all_penalty_neg) / len(all_penalty_neg) if all_penalty_neg else 0
    
    return avg_loss, avg_nce_loss, avg_penalty_pos, avg_penalty_neg


@torch.no_grad()
def validate(mlp, dataloader, device):
    mlp.eval()
    total_loss = 0
    total_nce_loss = 0
    num_batches = 0
    all_alphas = []
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
        
        S_final, alpha, tau, penalty_pos, penalty_neg = compute_scores(q_plus, q_minus, pos, neg, mlp, neg_mask)
        total_loss_batch, nce_loss, alpha_reg = dsclr_loss(S_final, alpha, temperature=0.1, lambda_reg=0.0)
        
        if not torch.isnan(total_loss_batch):
            total_loss += total_loss_batch.item()
            total_nce_loss += nce_loss.item()
            all_alphas.append(alpha.mean().item())
            all_penalty_pos.append(penalty_pos.mean().item())
            all_penalty_neg.append(penalty_neg.mean().item())
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_nce_loss = total_nce_loss / num_batches if num_batches > 0 else 0
    avg_alpha = sum(all_alphas) / len(all_alphas) if all_alphas else 0
    avg_penalty_pos = sum(all_penalty_pos) / len(all_penalty_pos) if all_penalty_pos else 0
    avg_penalty_neg = sum(all_penalty_neg) / len(all_penalty_neg) if all_penalty_neg else 0
    
    return avg_loss, avg_nce_loss, avg_alpha, avg_penalty_pos, avg_penalty_neg


def plot_and_save_curves(history, save_path):
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not installed, skipping plot generation")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(epochs, history['train_nce_loss'], 'b-', label='Train NCE Loss', linewidth=2)
    axes[0].plot(epochs, history['val_nce_loss'], 'r-', label='Val NCE Loss', linewidth=2)
    axes[0].set_ylabel('InfoNCE Loss')
    axes[0].set_title('DSCLR Training & Validation NCE Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(epochs, history['val_avg_alpha'], 'g-', label='Avg Predicted Alpha', linewidth=2)
    axes[1].set_ylabel('Alpha Value')
    axes[1].set_title('Average Predicted Penalty Multiplier (Alpha)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[2].plot(epochs, history['val_penalty_pos'], 'm-', label='Pos Penalty', linewidth=2)
    axes[2].plot(epochs, history['val_penalty_neg'], 'c-', label='Neg Penalty', linewidth=2)
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Penalty Value')
    axes[2].set_title('Average Positive vs Negative Penalties')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("DSCLR MLP 训练 (带验证集 & Early Stopping)")
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
    mlp = DSCLR_MLP(input_dim=1024, hidden_dim=256).to(DEVICE)
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
        'val_penalty_pos': [],
        'val_penalty_neg': []
    }
    
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_nce_loss, train_penalty_pos, train_penalty_neg = train_epoch(mlp, train_loader, optimizer, DEVICE)
        val_loss, val_nce_loss, avg_alpha, val_penalty_pos, val_penalty_neg = validate(mlp, val_loader, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_nce_loss'].append(train_nce_loss)
        history['val_loss'].append(val_loss)
        history['val_nce_loss'].append(val_nce_loss)
        history['val_avg_alpha'].append(avg_alpha)
        history['val_penalty_pos'].append(val_penalty_pos)
        history['val_penalty_neg'].append(val_penalty_neg)
        
        writer.add_scalar('Loss/Train', train_nce_loss, epoch) if writer else None
        writer.add_scalar('Loss/Val', val_nce_loss, epoch) if writer else None
        writer.add_scalar('Metrics/Avg_Alpha', avg_alpha, epoch) if writer else None
        writer.add_scalar('Metrics/Penalty_Pos', val_penalty_pos, epoch) if writer else None
        writer.add_scalar('Metrics/Penalty_Neg', val_penalty_neg, epoch) if writer else None
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train NCE={train_nce_loss:.4f}, Val NCE={val_nce_loss:.4f}, Alpha={avg_alpha:.4f}, PosPen={train_penalty_pos:.4f}, NegPen={train_penalty_neg:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp.state_dict(), f"{OUTPUT_DIR}/dsclr_best_mlp.pt")
            print(f"  🔥 新的 Best Model 已保存! (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ patience = {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n🛑 触发 Early Stopping! 连续{PATIENCE}轮未提升，停止训练。")
            break
    
    if writer:
        writer.close()
    
    plot_and_save_curves(history, f"{OUTPUT_DIR}/dsclr_training_curves.png")
    
    print(f"\n训练完成! Best Val Loss: {best_val_loss:.4f}")
    print(f"最佳模型已保存: {OUTPUT_DIR}/dsclr_best_mlp.pt")


if __name__ == "__main__":
    train()
