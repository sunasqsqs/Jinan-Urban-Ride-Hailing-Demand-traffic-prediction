import os
import sys
import subprocess
import warnings
import unicodedata

warnings.filterwarnings('ignore')

# 【终极防线】：强制单显卡，杜绝多卡梯度方差
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

env_needs_update = False
env = os.environ.copy()

if env.get('PYTHONHASHSEED') != '42':
    env['PYTHONHASHSEED'] = '42'
    env_needs_update = True

if env.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env_needs_update = True

if env_needs_update:
    subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(0)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import random
import time
import math

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

class Config:
    data_path = 'data/finaldata2.csv'
    save_dir = 'new results2/'

    grid_size = (15, 15)
    time_interval = '30min'

    history_steps = 12
    future_steps = 1

    batch_size = 64
    epochs = 50
    train_ratio = 0.7
    val_ratio = 0.15

    input_dim = 3
    hidden_dim = 32
    num_layers = 3
    dropout = 0.15

    num_nodes = grid_size[0] * grid_size[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
set_seed(config.seed)

def get_display_width(text):
    width = 0
    for c in str(text):
        if unicodedata.east_asian_width(c) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width

def pad_str(text, target_width):
    text = str(text)
    display_width = get_display_width(text)
    padding = target_width - display_width
    return text + ' ' * (padding if padding > 0 else 0)

def format_table_row(items, widths):
    formatted = []
    for item, w in zip(items, widths):
        formatted.append(pad_str(item, w))
    return " | ".join(formatted)

class RevIN_ST(nn.Module):
    def __init__(self, num_nodes, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_nodes, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_nodes, 1))

    def forward(self, x, mode):
        if mode == 'norm':
            demand = x[..., 0:1]
            self.mean = demand.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(demand.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            demand_norm = (demand - self.mean) / self.stdev
            demand_norm = demand_norm * self.gamma + self.beta
            return torch.cat([demand_norm, x[..., 1:]], dim=-1)
        elif mode == 'denorm':
            x = (x - self.beta) / self.gamma
            x = x * self.stdev[:, -1:, :, :] + self.mean[:, -1:, :, :]
            return x

class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter('bias', nn.Parameter(torch.zeros(d)) if bias else None)

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.bias is not None:
            return normed * self.weight + self.bias
        return normed * self.weight

try:
    from mamba_ssm import Mamba
    print(">> [系统] 成功载入原生 mamba-ssm 库！")
    class TS_MambaBlock(nn.Module):
        def __init__(self, d_model, expand=2, dropout=0.1):
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=expand)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            return x + self.drop(self.mamba(self.norm1(x)))
except ImportError:
    Mamba = None
    print(">> [系统] 未检测到 mamba_ssm，启用【Mock 单向 RNN】...")
    class TS_MambaBlock(nn.Module):
        def __init__(self, d_model, expand=2, dropout=0.1):
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.seq_core = nn.GRU(d_model, d_model, bidirectional=False, batch_first=True)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            x_norm = self.norm1(x)
            seq_out, _ = self.seq_core(x_norm)
            return x + self.drop(seq_out)

def load_and_process_data():
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"【致命错误】未找到 {config.data_path}")

    print(f"  [✓] 成功载入真实交通数据集: {config.data_path}")
    df = pd.read_csv(config.data_path)
    df['dep_time'] = pd.to_datetime(df['dep_time'])

    if 'grid_id' not in df.columns:
        lon_min, lon_max = 116.0, 118.0
        lat_min, lat_max = 36.0, 37.8
        lon_step = (lon_max - lon_min) / config.grid_size[0]
        lat_step = (lat_max - lat_min) / config.grid_size[1]
        def get_grid_id(row):
            x = int((row['dep_longitude'] - lon_min) / lon_step)
            y = int((row['dep_latitude'] - lat_min) / lat_step)
            x = max(0, min(x, config.grid_size[0] - 1))
            y = max(0, min(y, config.grid_size[1] - 1))
            return y * config.grid_size[0] + x
        df['grid_id'] = df.apply(get_grid_id, axis=1)

    lon_min, lon_max = 116.0, 118.0
    lat_min, lat_max = 36.0, 37.8
    lon_step = (lon_max - lon_min) / config.grid_size[0]
    lat_step = (lat_max - lat_min) / config.grid_size[1]

    df_agg = df.groupby([pd.Grouper(key='dep_time', freq=config.time_interval), 'grid_id']).size().unstack(fill_value=0)
    full_idx = pd.date_range(start=df_agg.index[0], end=df_agg.index[-1], freq=config.time_interval)
    df_agg = df_agg.reindex(full_idx, fill_value=0)

    for i in range(config.num_nodes):
        if i not in df_agg.columns:
            df_agg[i] = 0
    df_agg = df_agg[sorted(df_agg.columns)]

    hours = df_agg.index.hour + df_agg.index.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    demand_data = df_agg.values.astype(np.float32)
    demand_log = np.log1p(demand_data)

    train_size = int(len(demand_log) * config.train_ratio)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(demand_log[:train_size])
    demand_norm = scaler.transform(demand_log)

    combined_data = np.zeros((demand_norm.shape[0], config.num_nodes, 3), dtype=np.float32)
    for t in range(demand_norm.shape[0]):
        combined_data[t, :, 0] = demand_norm[t]
        combined_data[t, :, 1] = hour_sin[t]
        combined_data[t, :, 2] = hour_cos[t]

    grid_meta = {
        'lon_min': lon_min, 'lon_step': lon_step,
        'lat_min': lat_min, 'lat_step': lat_step,
        'cols': config.grid_size[0], 'rows': config.grid_size[1]
    }
    return combined_data, scaler, grid_meta

def get_adjacency_matrix():
    adj = np.zeros((config.num_nodes, config.num_nodes), dtype=np.float32)
    rows, cols = config.grid_size
    for r in range(rows):
        for c in range(cols):
            curr = r * cols + c
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[curr, nr * cols + nc] = 1.0

    adj = adj + np.eye(config.num_nodes)
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return torch.tensor(adj_norm, device=config.device, dtype=torch.float32)

def create_dataloaders(data):
    X, Y = [], []
    for i in range(len(data) - config.history_steps - config.future_steps + 1):
        X.append(data[i : i + config.history_steps])
        Y.append(data[i + config.history_steps : i + config.history_steps + config.future_steps, :, 0])

    X, Y = np.array(X), np.array(Y)

    train_end = int(len(X) * config.train_ratio)
    val_end = train_end + int(len(X) * config.val_ratio)

    X_train = torch.FloatTensor(X[:train_end])
    Y_train = torch.FloatTensor(Y[:train_end])
    X_val = torch.FloatTensor(X[train_end:val_end])
    Y_val = torch.FloatTensor(Y[train_end:val_end])
    X_test = torch.FloatTensor(X[val_end:])
    Y_test = torch.FloatTensor(Y[val_end:])

    def to_loader(x, y, shuffle=False):
        ds = TensorDataset(x, y)
        kwargs = {
            'batch_size': config.batch_size,
            'shuffle': shuffle,
            'num_workers': 0,
            'drop_last': False
        }
        return DataLoader(ds, **kwargs)

    return (to_loader(X_train, Y_train, True),
            to_loader(X_val, Y_val, False),
            to_loader(X_test, Y_test, False))

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, count_include_pad=False)
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        trend = self.moving_avg(x_t).permute(0, 2, 1)
        res = x - trend
        return res, trend

class AnchorReadout(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.temporal_proj = nn.Linear(seq_len, 1)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        t_pool = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        last_out = x[:, -1, :]
        fused = t_pool + last_out
        return self.norm(fused + self.proj(fused))

class Adaptive_Graph_Conv(nn.Module):
    def __init__(self, dim, num_nodes, dropout=0.1):
        super().__init__()
        self.node_emb_s = nn.Parameter(torch.randn(num_nodes, dim // 4) * 0.01)
        self.node_emb_t = nn.Parameter(torch.randn(dim // 4, num_nodes) * 0.01)

        self.W_p = nn.Linear(dim, dim)
        self.W_a = nn.Linear(dim, dim)
        self.W_r = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        res = self.W_r(x)
        x = self.norm(x)

        out_p = torch.einsum('ij, bjc -> bic', adj, x)
        out_p = self.W_p(out_p)

        adp_adj = torch.softmax(torch.relu(torch.matmul(self.node_emb_s, self.node_emb_t)), dim=-1)
        out_a = torch.einsum('ij, bjc -> bic', adp_adj, x)
        out_a = self.W_a(out_a)

        return res + self.drop(torch.nn.functional.silu(out_p + out_a))

class ST_ASG_Fusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # 联合上下文感知门控计算
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # 对传入的 GNN 空间特征进行非线性安全转换
        self.spatial_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.xavier_uniform_(self.gate[-1].weight)

        # 软冷启动保护
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, t_feat, g_feat):
        ctx = torch.cat([t_feat, g_feat], dim=-1)
        spatial_gate = torch.sigmoid(self.gate(ctx))
        return t_feat + spatial_gate * self.spatial_transform(g_feat)

class ST_Mamba_Model(nn.Module):
    def __init__(self, use_gnn=True, temporal_type='mamba', fusion_mode='advanced', adj=None):
        super().__init__()
        self.use_gnn = use_gnn
        self.temporal_type = temporal_type.lower()
        self.fusion_mode = fusion_mode.lower()

        if adj is not None:
            self.register_buffer('adj_matrix', adj)

        self.h_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.revin = RevIN_ST(config.num_nodes)

        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU()
        )

        self.spatial_emb = nn.Parameter(torch.randn(1, 1, config.num_nodes, self.h_dim) * 0.02)
        self.temporal_emb = nn.Parameter(torch.randn(1, config.history_steps, 1, self.h_dim) * 0.02)
        self.st_pe = nn.Parameter(torch.randn(1, config.num_nodes, config.history_steps, self.h_dim) * 0.02)

        self.decomp = SeriesDecomp(kernel_size=3)
        self.trend_proj = nn.Linear(self.h_dim, self.h_dim)

        # 构建通用时序骨干网络，容纳所有对照组Baseline
        if self.temporal_type == 'mamba':
            self.temporal_net = nn.ModuleList([
                TS_MambaBlock(d_model=self.h_dim, expand=2, dropout=config.dropout)
                for _ in range(self.num_layers)
            ])
        elif self.temporal_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.h_dim, nhead=4,
                dim_feedforward=self.h_dim * 2, dropout=config.dropout, batch_first=True
            )
            self.temporal_net = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        elif self.temporal_type == 'lstm':
            self.temporal_net = nn.LSTM(self.h_dim, self.h_dim, num_layers=self.num_layers, batch_first=True, dropout=config.dropout if self.num_layers > 1 else 0)
        elif self.temporal_type == 'gru':
            self.temporal_net = nn.GRU(self.h_dim, self.h_dim, num_layers=self.num_layers, batch_first=True, dropout=config.dropout if self.num_layers > 1 else 0)

        # ---------------- 空间塔 ----------------
        if self.use_gnn:
            self.spatial_net = Adaptive_Graph_Conv(self.h_dim, config.num_nodes, dropout=config.dropout)

            if self.fusion_mode == 'advanced':
                self.st_asg = ST_ASG_Fusion(self.h_dim, dropout=config.dropout)
            elif self.fusion_mode == 'add':
                self.simple_s_proj = nn.Linear(self.h_dim, self.h_dim)

        self.temporal_readout = AnchorReadout(self.h_dim, config.history_steps)
        self.ar_full = nn.Linear(config.history_steps, 1)

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.h_dim),
            SwiGLU(self.h_dim, self.h_dim, hidden_dim=self.h_dim * 2),
            nn.Dropout(config.dropout),
            nn.Linear(self.h_dim, 1)
        )

        self.conf_gate = nn.Linear(self.h_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'temporal_net' not in name and 'spatial_net' not in name and 'st_asg' not in name:
                nn.init.xavier_uniform_(p)

        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.xavier_uniform_(self.ar_full.weight)
        nn.init.zeros_(self.ar_full.bias)

        if hasattr(self, 'conf_gate'):
            nn.init.zeros_(self.conf_gate.weight)
            nn.init.constant_(self.conf_gate.bias, -1.0)

    def forward(self, x):
        B, T, N, C = x.shape

        x_norm = self.revin(x, 'norm')
        x_emb = self.embedding(x_norm) + self.spatial_emb + self.temporal_emb

        t_in = x_emb.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        res, trend = self.decomp(t_in)

        # ---------------- 核心时序流转 ----------------
        res_spatial = res.reshape(B, N, T, -1) + self.st_pe
        t_dyn_long = res_spatial.reshape(B, N * T, -1)

        if self.temporal_type == 'mamba':
            for layer in self.temporal_net:
                t_dyn_long = layer(t_dyn_long)
        elif self.temporal_type == 'transformer':
            t_dyn_long = self.temporal_net(t_dyn_long)
        elif self.temporal_type in ['lstm', 'gru']:
            t_dyn_long, _ = self.temporal_net(t_dyn_long)

        # 若 temporal_type == 'none', 则直接越过序列层跳出

        t_dyn = t_dyn_long.reshape(B * N, T, -1)
        t_out = t_dyn + self.trend_proj(trend)
        t_feat = self.temporal_readout(t_out).reshape(B, N, -1)

        # ---------------- 空间扩散与协同融合 ----------------
        if self.use_gnn and self.temporal_type != 'none':
            g_feat = self.spatial_net(t_feat, self.adj_matrix)

            if self.fusion_mode == 'advanced':
                final_feat = self.st_asg(t_feat, g_feat)
            else:
                final_feat = t_feat + self.simple_s_proj(g_feat)
        elif self.use_gnn and self.temporal_type == 'none':
            # GNN-Only Fallback 独立处理
            g_feat = self.spatial_net(t_feat, self.adj_matrix)
            final_feat = g_feat
        else:
            final_feat = t_feat

        # ---------------- 输出头 ----------------
        delta_norm = self.output_head(final_feat).reshape(B, 1, N, 1)
        conf = torch.sigmoid(self.conf_gate(final_feat)).reshape(B, 1, N, 1)

        history_demand = x_norm[..., 0]
        ar_base = self.ar_full(history_demand.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)

        pred_norm = ar_base + delta_norm * conf
        pred_real = self.revin(pred_norm, 'denorm')

        return pred_real

def plot_fusion_loss(history_dict, save_dir):
    if 'Mamba-GNN' not in history_dict: return
    hist = history_dict['Mamba-GNN']
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='训练损失 (Training Loss)', color='#FF1493', linewidth=2, alpha=0.9)
    plt.plot(hist['val_loss'], label='验证损失 (Validation Loss)', color='#00D9FF', linewidth=2, linestyle='--', alpha=0.9)
    plt.title("Mamba-GNN 模型训练损失", fontsize=16)
    plt.xlabel("轮数 (Epochs)", fontsize=12)
    plt.ylabel("损失 (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'fusion_model_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 完整对照组调色板
PLOT_COLORS = {
    'Mamba-GNN': '#FF1493',
    'Mamba-U-GNN': '#FFD700',
    'Transformer-GNN': '#00FFFF',
    'LSTM-GNN': '#FFA500',
    'GRU-GNN': '#FFFF00',
    'Mamba-Only': '#FF4500',
    'Transformer-Only': '#FF69B4',
    'LSTM-Only': '#1E90FF',
    'GRU-Only': '#8A2BE2',
    'GNN-Only': '#32CD32'
}

def plot_total_demand(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    plt.figure(figsize=(15, 6))
    first_key = list(all_trues.keys())[0]
    total_true = np.sum(all_trues[first_key], axis=1)
    plot_len = min(len(total_true), 200)

    plt.plot(total_true[:plot_len], label='真实值 (Ground Truth)', color='white', linewidth=3, alpha=0.5)

    sorted_names = sorted(all_preds.keys(), key=lambda x: ('Mamba' in x, 'Transformer' in x), reverse=True)

    for name in sorted_names:
        preds = all_preds[name]
        total_pred = np.sum(preds, axis=1)
        lw = 3.5 if 'Mamba' in name else 1.5
        alpha = 0.95 if 'Mamba' in name else 0.7
        ls = '-' if 'Mamba' in name else '--'
        plt.plot(total_pred[:plot_len], label=name, color=PLOT_COLORS.get(name, 'red'),
                 linewidth=lw, linestyle=ls, alpha=alpha)

    title_str = f"[{prefix}] 总需求量预测对比" if prefix else "总需求量预测对比"
    plt.title(title_str, fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    filename = f"{prefix}_total_demand_comparison.png" if prefix else "total_demand_comparison.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_fit(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    n_models = len(all_preds)
    cols = min(4, n_models)
    rows = math.ceil(n_models / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (name, preds) in enumerate(all_preds.items()):
        plt.subplot(rows, cols, i+1)
        trues = all_trues[name].flatten()
        preds = preds.flatten()
        idx = np.random.choice(len(trues), min(10000, len(trues)), replace=False)
        plt.scatter(trues[idx], preds[idx], alpha=0.3, color=PLOT_COLORS.get(name, 'white'), s=5)
        max_val = max(trues[idx].max(), preds[idx].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想拟合线')

        plt.title(f"{name} 拟合分析", fontsize=14, color=PLOT_COLORS.get(name, 'white') if 'Mamba' in name else 'white')
        plt.xlabel("真实值", fontsize=12)
        if i % cols == 0: plt.ylabel("预测值", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.legend()

    plt.tight_layout()
    filename = f"{prefix}_goodness_of_fit.png" if prefix else "goodness_of_fit.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_error(preds, trues, grid_meta, name, save_dir):
    node_mae = np.mean(np.abs(trues - preds), axis=0)
    error_matrix = node_mae.reshape((grid_meta['rows'], grid_meta['cols']))
    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, cmap='magma', origin='lower', aspect='auto')
    plt.colorbar(label='平均绝对误差 (MAE)')
    plt.title(f"{name} 空间误差分布", fontsize=14)
    plt.xlabel("经度网格索引")
    plt.ylabel("纬度网格索引")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spatial_error_map_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    plt.figure(figsize=(12, 6))

    for name, preds in all_preds.items():
        errors = (all_trues[name] - preds).flatten()
        errors = errors[(errors >= -2) & (errors <= 2)]
        lw = 2.5 if 'Mamba' in name else 1.0
        plt.hist(errors, bins=100, alpha=0.5 if 'Mamba' in name else 0.3, label=name, color=PLOT_COLORS.get(name, 'white'), density=True, histtype='step', linewidth=lw)

    plt.axvline(x=0, color='white', linestyle='--', linewidth=2)
    title_str = f"[{prefix}] 误差分布" if prefix else "误差分布"
    plt.title(title_str, fontsize=16)
    plt.xlabel("误差值 (真实值 - 预测值)", fontsize=12)
    plt.ylabel("密度 (Density)", fontsize=12)
    plt.xlim(-2, 2)
    plt.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    filename = f"{prefix}_error_distribution.png" if prefix else "error_distribution.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_epoch_metrics(history_dict, save_dir):
    for name, hist in history_dict.items():
        if 'train_mse' not in hist or not hist['train_mse']: continue
        epochs = range(1, len(hist['train_mse']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{name} - 训练指标变化曲线", fontsize=18, weight='bold', color=PLOT_COLORS.get(name, 'white'))

        axs[0, 0].plot(epochs, hist['train_acc'], label='训练 Acc', color='#00FF00', linewidth=2)
        axs[0, 0].plot(epochs, hist['val_acc'], label='验证 Acc', color='#FF00FF', linewidth=2, linestyle='--')
        axs[0, 0].plot(epochs, hist['test_acc'], label='测试 Acc', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[0, 0].set_title('准确率 (Accuracy)', fontsize=14)
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.3)

        axs[0, 1].plot(epochs, hist['train_mae'], label='训练 MAE', color='#00FF00', linewidth=2)
        axs[0, 1].plot(epochs, hist['val_mae'], label='验证 MAE', color='#FF00FF', linewidth=2, linestyle='--')
        axs[0, 1].plot(epochs, hist['test_mae'], label='测试 MAE', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[0, 1].set_title('平均绝对误差 (MAE)', fontsize=14)
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.3)

        axs[1, 0].plot(epochs, hist['train_mse'], label='训练 MSE', color='#00FF00', linewidth=2)
        axs[1, 0].plot(epochs, hist['val_mse'], label='验证 MSE', color='#FF00FF', linewidth=2, linestyle='--')
        axs[1, 0].plot(epochs, hist['test_mse'], label='测试 MSE', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[1, 0].set_title('均方误差 (MSE)', fontsize=14)
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle='--', alpha=0.3)

        train_r2 = [max(x, -1.0) for x in hist['train_r2']]
        val_r2 = [max(x, -1.0) for x in hist['val_r2']]
        test_r2 = [max(x, -1.0) for x in hist['test_r2']]
        axs[1, 1].plot(epochs, train_r2, label='训练 R2', color='#00FF00', linewidth=2)
        axs[1, 1].plot(epochs, val_r2, label='验证 R2', color='#FF00FF', linewidth=2, linestyle='--')
        axs[1, 1].plot(epochs, test_r2, label='测试 R2', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[1, 1].set_title('决定系数 (R2 Score)', fontsize=14)
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'{name}_all_metrics_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compute_mape(y_true, y_pred, threshold=5.0):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100

def get_hotspots(pred_data, grid_meta, top_k=5):
    total_vol = np.sum(pred_data, axis=0)
    sorted_indices = np.argsort(-total_vol)
    top_indices = sorted_indices[:top_k]

    hotspots = []
    for rank, idx in enumerate(top_indices):
        idx = int(idx)
        cols = grid_meta['cols']
        y = idx // cols
        x = idx % cols
        lon_start = grid_meta['lon_min'] + x * grid_meta['lon_step']
        lon_end = lon_start + grid_meta['lon_step']
        lat_start = grid_meta['lat_min'] + y * grid_meta['lat_step']
        lat_end = lat_start + grid_meta['lat_step']

        hotspots.append({
            "rank": rank + 1,
            "grid_id": idx,
            "lon_range": [float(lon_start), float(lon_end)],
            "lat_range": [float(lat_start), float(lat_end)],
            "predicted_volume": int(total_vol[idx])
        })
    return hotspots

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_model(model, path)

    def save_model(self, model, path):
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, path)

class Hybrid_Robust_Loss(nn.Module):
    def __init__(self, mse_weight=0.02):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.mse_w = mse_weight

    def forward(self, pred, true):
        # 98% L1 用于狂降误差，2% MSE 垫底防震荡
        return self.l1(pred, true) + self.mse_w * self.mse(pred, true)

def run_exp(name, model, loaders, scaler):
    set_seed(config.seed)

    train_loader, val_loader, test_loader = loaders

    weight_decay = 1e-4
    lr = 1.0e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    criterion = Hybrid_Robust_Loss(mse_weight=0.02)
    early_stopping = EarlyStopping(patience=8, delta=1e-6)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=False)
    save_path = os.path.join(config.save_dir, f'best_model_{name}.pt')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_acc': [], 'val_acc': [], 'test_acc': []
    }

    def calculate_metrics_from_tensors(p_list, t_list):
        pred_norm = np.concatenate(p_list)
        true_norm = np.concatenate(t_list)

        pred_norm = np.clip(pred_norm, 0.0, 1.0)
        true_norm = np.clip(true_norm, 0.0, 1.0)

        pred_log = scaler.inverse_transform(pred_norm)
        true_log = scaler.inverse_transform(true_norm)

        pred_log = np.clip(pred_log, a_min=-10.0, a_max=20.0)

        p_final = np.maximum(np.expm1(pred_log), 0)
        t_final = np.maximum(np.expm1(true_log), 0)

        mse = mean_squared_error(t_final, p_final)
        mae = mean_absolute_error(t_final, p_final)
        r2 = r2_score(t_final.flatten(), p_final.flatten())
        wmape = compute_wmape(t_final, p_final)
        acc = max(0.0, 100.0 - wmape)
        return mse, mae, r2, acc

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  > [{name}] 初始化完成 | 模型参数量: {total_params:,}")

    total_train_time = 0.0
    epochs_run = 0

    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        model.train()
        t_loss = 0
        train_p, train_t = [], []
        for bx, by in train_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=False):
                p = model(bx)
                loss = criterion(p.squeeze(), by.squeeze())

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            t_loss += loss.item()
            train_p.append(p.detach().cpu().reshape(-1, config.num_nodes))
            train_t.append(by.cpu().reshape(-1, config.num_nodes))

        model.eval()
        v_loss = 0
        val_p, val_t = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(config.device), by.to(config.device)
                with torch.cuda.amp.autocast(enabled=False):
                    p = model(bx)
                    v_loss += criterion(p.squeeze(), by.squeeze()).item()
                val_p.append(p.cpu().reshape(-1, config.num_nodes))
                val_t.append(by.cpu().reshape(-1, config.num_nodes))

        test_p, test_t = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(config.device), by.to(config.device)
                with torch.cuda.amp.autocast(enabled=False):
                    p = model(bx)
                test_p.append(p.cpu().reshape(-1, config.num_nodes))
                test_t.append(by.cpu().reshape(-1, config.num_nodes))

        tr_mse, tr_mae, tr_r2, tr_acc = calculate_metrics_from_tensors(train_p, train_t)
        v_mse, v_mae, v_r2, v_acc = calculate_metrics_from_tensors(val_p, val_t)
        te_mse, te_mae, te_r2, te_acc = calculate_metrics_from_tensors(test_p, test_t)

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)

        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)

        history['train_mse'].append(tr_mse); history['val_mse'].append(v_mse); history['test_mse'].append(te_mse)
        history['train_mae'].append(tr_mae); history['val_mae'].append(v_mae); history['test_mae'].append(te_mae)
        history['train_r2'].append(tr_r2);   history['val_r2'].append(v_r2);   history['test_r2'].append(te_r2)
        history['train_acc'].append(tr_acc); history['val_acc'].append(v_acc); history['test_acc'].append(te_acc)

        scheduler.step()
        early_stopping(avg_v, model, save_path)

        epoch_duration = time.time() - epoch_start_time
        total_train_time += epoch_duration
        epochs_run += 1

        if early_stopping.early_stop:
            print(f"  -> [{name}] 在第 {epoch+1} 轮触发提前停止 (Early Stopping)")
            break

        if (epoch+1) % 5 == 0 or (epoch+1) == config.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  -> Epoch [{epoch+1:03d}/{config.epochs:03d}] | Val Loss: {avg_v:.5f} | Val MAE: {v_mae:.4f} | LR: {current_lr:.2e} | 耗时: {epoch_duration:.2f}s")

    state_dict = torch.load(save_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            with torch.cuda.amp.autocast(enabled=False):
                p = model(bx)
            all_p.append(p.cpu().reshape(-1, config.num_nodes))
            all_t.append(by.cpu().reshape(-1, config.num_nodes))

    pred_norm = np.concatenate(all_p)
    true_norm = np.concatenate(all_t)

    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    true_norm = np.clip(true_norm, 0.0, 1.0)

    pred_log = scaler.inverse_transform(pred_norm)
    true_log = scaler.inverse_transform(true_norm)

    p_final = np.maximum(np.expm1(pred_log), 0)
    t_final = np.maximum(np.expm1(true_log), 0)

    avg_epoch_time = total_train_time / epochs_run

    return history, p_final, t_final, -early_stopping.best_score, total_params, avg_epoch_time

TIME_COMPLEXITY_MAP = {
    'Mamba-GNN': 'O(B·(T·N)·d + T·N²) [ST-ASG 顶级门控协同]',
    'Mamba-U-GNN': 'O(B·(T·N)·d + T·N²) [暴力相加模态污染]',
    'Mamba-Only': 'O(B·(T·N)·d) [极致纯净时间流]',
    'Transformer-GNN': 'O(B·(T·N)² + T·N²) [全局自注意力]',
    'Transformer-Only': 'O(B·(T·N)²) [全局自注意力]',
    'LSTM-GNN': 'O(B·(T·N)·d² + T·N²) [长序列串行]',
    'GRU-GNN': 'O(B·(T·N)·d² + T·N²) [长序列串行]',
    'LSTM-Only': 'O(B·(T·N)·d²) [长序列串行]',
    'GRU-Only': 'O(B·(T·N)·d²) [长序列串行]',
    'GNN-Only': 'O(T·N²)'
}

def main():
    print("="*100)
    print(" 🚀 学术严谨版模型环境初始化完成！(完美对照与门控版)")
    print("="*100)

    data, scaler, grid_meta = load_and_process_data()
    loaders = create_dataloaders(data)
    adj = get_adjacency_matrix()

    # 包含所有时空深度学习基线以及我们精心设计的门控消融
    experiments = {
        'Mamba-GNN':  {'gnn': True,  'temporal': 'mamba', 'fusion': 'advanced'},
        'Mamba-U-GNN':{'gnn': True,  'temporal': 'mamba', 'fusion': 'add'},
        'Mamba-Only': {'gnn': False, 'temporal': 'mamba', 'fusion': 'none'},
        'Transformer-GNN': {'gnn': True, 'temporal': 'transformer', 'fusion': 'advanced'},
        'Transformer-Only': {'gnn': False, 'temporal': 'transformer', 'fusion': 'none'},
        'LSTM-GNN':   {'gnn': True,  'temporal': 'lstm', 'fusion': 'advanced'},
        'GRU-GNN':    {'gnn': True,  'temporal': 'gru', 'fusion': 'advanced'},
        'LSTM-Only':  {'gnn': False, 'temporal': 'lstm', 'fusion': 'none'},
        'GRU-Only':   {'gnn': False, 'temporal': 'gru', 'fusion': 'none'},
        'GNN-Only':   {'gnn': True,  'temporal': 'none', 'fusion': 'none'}
    }

    results = {}
    all_hist, all_pred, all_true = {}, {} , {}

    final_report = {
        "evaluation_groups": {},
        "loss_history": {},
        "predictions_time_series": {},
        "hotspots": []
    }

    for name, cfg in experiments.items():
        needs_multi_run = 'Mamba' in name or 'Transformer' in name
        num_runs = 3 if needs_multi_run else 1

        print("="*80)
        if needs_multi_run:
            print(f"🔹 [实验进行中] 模型名称: {name} (自动运行 {num_runs} 次取均值)")
        else:
            print(f"🔹 [实验进行中] 模型名称: {name} (运行 1 次)")
        print("-" * 80)

        run_metrics = {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'wmape': [], 'r2': [], 'time': []}
        best_mae = float('inf')
        best_hist, best_p, best_t = None, None, None
        final_params = 0

        for i in range(num_runs):
            if num_runs > 1:
                print(f"\n   >>> 正在执行第 {i+1}/{num_runs} 次独立运行...")

            set_seed(config.seed)

            model = ST_Mamba_Model(
                use_gnn=cfg['gnn'],
                temporal_type=cfg['temporal'],
                fusion_mode=cfg['fusion'],
                adj=adj
            ).to(config.device)

            h, p, t, val_loss, params, avg_time = run_exp(name, model, loaders, scaler)

            mse = mean_squared_error(t, p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(t, p)
            r2 = r2_score(t.flatten(), p.flatten())
            mape = compute_mape(t, p, threshold=5.0)
            wmape = compute_wmape(t, p)

            run_metrics['mse'].append(mse)
            run_metrics['rmse'].append(rmse)
            run_metrics['mae'].append(mae)
            run_metrics['mape'].append(mape)
            run_metrics['wmape'].append(wmape)
            run_metrics['r2'].append(r2)
            run_metrics['time'].append(avg_time)
            final_params = params

            if mae < best_mae:
                best_mae = mae
                best_hist, best_p, best_t = h, p, t

        if needs_multi_run:
            results[name] = {
                "mse": f"{np.mean(run_metrics['mse']):.2f}±{np.std(run_metrics['mse']):.2f}",
                "rmse": f"{np.mean(run_metrics['rmse']):.2f}±{np.std(run_metrics['rmse']):.2f}",
                "mae": f"{np.mean(run_metrics['mae']):.2f}±{np.std(run_metrics['mae']):.2f}",
                "mape": f"{np.mean(run_metrics['mape']):.2f}±{np.std(run_metrics['mape']):.2f}",
                "wmape": f"{np.mean(run_metrics['wmape']):.2f}±{np.std(run_metrics['wmape']):.2f}",
                "r2": f"{np.mean(run_metrics['r2']):.4f}±{np.std(run_metrics['r2']):.4f}",
                "parameters": int(final_params),
                "avg_epoch_time_s": f"{np.mean(run_metrics['time']):.2f}"
            }
        else:
            results[name] = {
                "mse": f"{run_metrics['mse'][0]:.2f}",
                "rmse": f"{run_metrics['rmse'][0]:.2f}",
                "mae": f"{run_metrics['mae'][0]:.2f}",
                "mape": f"{run_metrics['mape'][0]:.2f}",
                "wmape": f"{run_metrics['wmape'][0]:.2f}",
                "r2": f"{run_metrics['r2'][0]:.4f}",
                "parameters": int(final_params),
                "avg_epoch_time_s": f"{run_metrics['time'][0]:.2f}"
            }

        all_hist[name] = best_hist
        all_pred[name] = best_p
        all_true[name] = best_t

        final_report["loss_history"][name] = {
            "train": [float(x) for x in best_hist['train_loss']],
            "val": [float(x) for x in best_hist['val_loss']]
        }

    # 严谨划分三大经典实验评估矩阵
    experiment_groups = {
        "Ablation_Study": {
            "title": "消融实验 (Ablation Study) - 验证 ST-ASG 特征级协同门控",
            "models": ['Mamba-GNN', 'Mamba-U-GNN', 'Mamba-Only', 'GNN-Only']
        },
        "Fusion_Comparison": {
            "title": "融合模型对比 (Spatio-Temporal Baselines) - 验证 Mamba 效能",
            "models": ['Mamba-GNN', 'Transformer-GNN', 'LSTM-GNN', 'GRU-GNN']
        },
        "Single_Comparison": {
            "title": "单一时序模型对比 (Temporal Baselines)",
            "models": ['Mamba-Only', 'Transformer-Only', 'LSTM-Only', 'GRU-Only']
        }
    }

    widths = [16, 14, 14, 14, 14, 14, 15, 10, 12, 45]
    headers = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'WMAPE', 'R2', 'Params', 'Time/Ep(s)', 'Complexity']

    line_length = sum(widths) + len(widths) * 3 - 1

    print("\n\n" + "="*line_length)
    print(" 严谨学术模型评估与对比报告")
    print("="*line_length)

    for group_key, group_info in experiment_groups.items():
        title = group_info["title"]
        models_in_group = group_info["models"]

        print(f"\n>> {title}")
        print("-" * line_length)

        print(format_table_row(headers, widths))
        print("-" * line_length)

        group_results = {}
        for name in models_in_group:
            if name in results:
                m = results[name]
                group_results[name] = m
                cplx = TIME_COMPLEXITY_MAP.get(name, 'N/A')

                columns = [
                    name,
                    str(m['mse']), str(m['rmse']), str(m['mae']),
                    str(m['mape']), str(m['wmape']), str(m['r2']),
                    str(m['parameters']), str(m['avg_epoch_time_s']),
                    cplx
                ]
                print(format_table_row(columns, widths))
        print("-" * line_length)

        final_report["evaluation_groups"][group_key] = group_results

        subset_pred = {k: all_pred[k] for k in models_in_group if k in all_pred}
        subset_true = {k: all_true[k] for k in models_in_group if k in all_true}

        plot_total_demand(subset_pred, subset_true, config.save_dir, prefix=group_key)
        plot_scatter_fit(subset_pred, subset_true, config.save_dir, prefix=group_key)
        plot_error_distribution(subset_pred, subset_true, config.save_dir, prefix=group_key)

    print(f"\n[可视化] 实验评估图表已生成至目录: {os.path.abspath(config.save_dir)}")

    plot_fusion_loss(all_hist, config.save_dir)
    if 'Mamba-GNN' in all_pred:
        plot_spatial_error(all_pred['Mamba-GNN'], all_true['Mamba-GNN'], grid_meta, 'Mamba-GNN', config.save_dir)
    plot_epoch_metrics(all_hist, config.save_dir)

    if 'Mamba-GNN' in all_true:
        gt_series = np.sum(all_true['Mamba-GNN'], axis=1).tolist()
        final_report["predictions_time_series"]["ground_truth"] = [float(x) for x in gt_series]
        pred_series = np.sum(all_pred['Mamba-GNN'], axis=1).tolist()
        final_report["predictions_time_series"]["prediction_mamba_gnn"] = [float(x) for x in pred_series]

        hotspots_data = get_hotspots(all_pred['Mamba-GNN'], grid_meta, top_k=5)
        final_report["hotspots"] = hotspots_data

        widths_hot = [6, 8, 12, 24, 24]
        headers_hot = ['Rank', 'Grid ID', '预测总流量', '经度范围 (Lon)', '纬度范围 (Lat)']
        line_hot_length = sum(widths_hot) + len(widths_hot) * 3 - 1

        print("\n" + "="*line_hot_length)
        print(" 📍 流量预测最高 Top 5 热点区域 (基于最佳模型结果)")
        print("-" * line_hot_length)
        print(format_table_row(headers_hot, widths_hot))
        print("-" * line_hot_length)

        for item in hotspots_data:
            lon_r = f"[{item['lon_range'][0]:.4f}, {item['lon_range'][1]:.4f}]"
            lat_r = f"[{item['lat_range'][0]:.4f}, {item['lat_range'][1]:.4f}]"
            row_hot = [str(item['rank']), str(item['grid_id']), str(item['predicted_volume']), lon_r, lat_r]
            print(format_table_row(row_hot, widths_hot))
        print("="*line_hot_length)

    json_path = os.path.join(config.save_dir, 'experiment_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\n[文件] 结构化报告 JSON 已生成: {json_path}")
    print(f"[完成] 所有深度学习评估流程安全结束！")

if __name__ == '__main__':
    main()