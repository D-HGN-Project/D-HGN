import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model

def load_data_and_model(data_root="./data", checkpoint_path="checkpoints/dhgn/EMCI_vs_CN/fold_0_best.pth"):
    # 1. Load Data
    dl = DHGNDataLoader(data_root=data_root)
    print("Loading data...")
    dynamic_graphs, sc_matrices, labels, subject_ids = dl.load_all_data(groups=['EMCI', 'CN'], use_sc=True)
    
    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Load Model with correct config
    config = {
        'num_rois': 90,
        'num_windows': 71,
        'num_classes': 2,
        'spatial_hidden_dim': 48,
        'temporal_hidden_dim': 192,
        'st_output_dim': 192,
        'use_sc': True,
        'sc_hidden_dim': 96,
        'sc_output_dim': 48,
        'population_hidden_dim': 96,
        'num_gnn_layers': 3,
        'dropout': 0.4
    }
    model = create_dhgn_model(config).to(device)
    
    # Load weights
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    return model, dynamic_graphs, sc_matrices, labels, device


def compute_saliency_maps(model, dynamic_graphs, sc_matrices, labels, device):
    """
    计算基于 Input x Gradient 的显著性图
    """
    model.eval()
    
    # 我们按样本单独计算，避免爆显存，同时也为了累加组级显著性
    saliency_nc = []
    saliency_emci = []
    
    print("Computing Saliency Maps...")
    for i in range(len(labels)):
        # 提取单个样本并加上batch维
        dfc_tensor = torch.tensor(dynamic_graphs[i:i+1], dtype=torch.float32).to(device)
        sc_tensor = torch.tensor(sc_matrices[i:i+1], dtype=torch.float32).to(device)
        label = labels[i]
        
        # 激活梯度
        dfc_tensor.requires_grad = True
        sc_tensor.requires_grad = True
        
        # 前向传播
        logits = model(dfc_tensor, sc_tensor)
        
        # 我们对类别 1 (eMCI) 的 logit 求导，看哪些特征促成了病理特征
        model.zero_grad()
        score_disease = logits[0, 1] 
        score_disease.backward()
        
        # 计算 Input x Gradient
        dfc_grad_disease = dfc_tensor.grad.detach().cpu().numpy()[0].mean(axis=0)  # [90, 90]
        sc_grad_disease = sc_tensor.grad.detach().cpu().numpy()[0]    # [90, 90]
        
        # fMRI的显著性图：对时间维度求平均，然后乘上输入特征的均值
        dfc_input = dynamic_graphs[i].mean(axis=0) # [90, 90]
        dfc_sal_disease = dfc_input * dfc_grad_disease
        
        # SC的显著性图
        sc_input = sc_matrices[i]
        sc_sal_disease = sc_input * sc_grad_disease
        
        # 我们将两种模态加权或者可视化，这里以结构连接(SC)为例，或者fMRI
        # 既然 D-HGN 是用 fMRI 提取特征，SC构图，fMRI的改变可能更加强烈
        # 为了兼容性，我们把两者简单相加作为全局显著性，或者主要看 fMRI 的改变。
        # 论文中通常选取一个主路，我们选择 dfc_sal_disease (功能退化更明显)
        # 或者 sc_sal_disease 都可以。我们保存两者的融合
        combined_saliency = (dfc_sal_disease + sc_sal_disease) / 2.0
        
        # 反对称化，确保矩阵对称
        combined_saliency = (combined_saliency + combined_saliency.T) / 2.0
        
        if label == 0:
            saliency_nc.append(combined_saliency)
        else:
            saliency_emci.append(combined_saliency)
            
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(labels)} subjects")
            
    # 计算组级平均
    mean_nc = np.mean(saliency_nc, axis=0)
    mean_emci = np.mean(saliency_emci, axis=0)
    
    return mean_nc, mean_emci


def plot_group_heatmaps(mean_nc, mean_emci, output_dir="analysis_results"):
    """
    绘制图一：组级热力图 (Group-level Connectivity Heatmaps)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 标准化到 [0, 1] 区间以便使用 Parula 风格作图
    global_min = min(mean_nc.min(), mean_emci.min())
    global_max = max(mean_nc.max(), mean_emci.max())
    
    norm_nc = (mean_nc - global_min) / (global_max - global_min + 1e-8)
    norm_emci = (mean_emci - global_min) / (global_max - global_min + 1e-8)
    
    # 模拟 Parula 配色
    cmap = "viridis" 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(norm_nc, ax=axes[0], cmap=cmap, square=True, vmin=0, vmax=1,
                xticklabels=10, yticklabels=10, cbar_kws={'shrink': 0.8})
    axes[0].set_title("(a) NC", fontsize=16)
    axes[0].set_xlabel("ROI", fontsize=14)
    axes[0].set_ylabel("ROI", fontsize=14)
    axes[0].invert_yaxis()
    
    sns.heatmap(norm_emci, ax=axes[1], cmap=cmap, square=True, vmin=0, vmax=1,
                xticklabels=10, yticklabels=10, cbar_kws={'shrink': 0.8})
    axes[1].set_title("(b) eMCI", fontsize=16)
    axes[1].set_xlabel("ROI", fontsize=14)
    axes[1].set_ylabel("ROI", fontsize=14)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_group_heatmaps.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'fig4_group_heatmaps.png')}")
    plt.close()


def plot_altered_connectivity_scatter(mean_nc, mean_emci, output_dir="analysis_results"):
    """
    绘制图二：差异化连接散点图 (Altered Connectivity Scatter Plot)
    """
    # 差异矩阵: eMCI - NC
    diff_matrix = mean_emci - mean_nc
    
    # 提取上三角避免重复
    diff_matrix = np.triu(diff_matrix, k=1)
    
    # 正向差异 (Increased / Compensatory)
    increased = np.maximum(diff_matrix, 0)
    # 负向差异 (Decreased / Damage), 转换为正的绝对值以便比较阈值
    decreased = np.abs(np.minimum(diff_matrix, 0))
    
    # 计算非零阈值
    def get_threshold_mask(matrix, percentile):
        flat_nonzero = matrix[matrix > 0]
        if len(flat_nonzero) == 0:
            return np.zeros_like(matrix)
            
        threshold = np.percentile(flat_nonzero, percentile)
        mask = (matrix >= threshold).astype(float)
        # 对称补全
        return mask + mask.T

    # 绘制50%和75%阈值 (即 Top 50% / Top 25% 显著边)
    # 注意：percentile是下分位数，Top 50% 对应 percentile=50，Top 25%对应 percentile=75
    dec_50 = get_threshold_mask(decreased, 50)
    inc_50 = get_threshold_mask(increased, 50)
    dec_75 = get_threshold_mask(decreased, 75)
    inc_75 = get_threshold_mask(increased, 75)
    
    # 画图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = [
        "Decreased connections\nThreshold=50%",
        "Increased connections\nThreshold=50%",
        "Decreased connections\nThreshold=75%",
        "Increased connections\nThreshold=75%"
    ]
    matrices = [dec_50, inc_50, dec_75, inc_75]
    
    # 背景设为黄色，点设为蓝色（模拟参考图）
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['yellow', 'darkblue'])
    
    for ax, mat, title in zip(axes, matrices, titles):
        # sns.heatmap 对稀疏0,1矩阵可视化，0=黄，1=蓝
        sns.heatmap(mat, ax=ax, cmap=custom_cmap, square=True, cbar=False,
                    xticklabels=10, yticklabels=10)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("ROI", fontsize=14)
        ax.set_ylabel("ROI", fontsize=14)
        ax.invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_altered_connectivity.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'fig8_altered_connectivity.png')}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='D-HGN Saliency Visualization')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dhgn/EMCI_vs_CN/fold_0_best.pth')
    args = parser.parse_args()
    
    # 为了保证不出错，如果 fold 0 没有，找第一个存在的
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        import glob
        ckpts = glob.glob('checkpoints/dhgn/EMCI_vs_CN/*.pth')
        if ckpts:
            ckpt_path = ckpts[0]
            print(f"Default checkpoint not found, using {ckpt_path}")
        else:
            print("No checkpoints found! Make sure you have trained models.")
            exit(1)
            
    print(f"Using checkpoint: {ckpt_path}")
    
    model, dynamic_graphs, sc_matrices, labels, device = load_data_and_model(
        data_root=args.data_root, checkpoint_path=ckpt_path)
    
    mean_nc, mean_emci = compute_saliency_maps(model, dynamic_graphs, sc_matrices, labels, device)
    
    plot_group_heatmaps(mean_nc, mean_emci)
    plot_altered_connectivity_scatter(mean_nc, mean_emci)
    print("All visualizations generated successfully.")
