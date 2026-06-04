
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from dhgn_model import create_dhgn_model
from data_loader import DHGNDataLoader
from gpu_utils import setup_gpu

# Configuration
CKPT_DIR = 'checkpoints/dhgn'
TASK = 'EMCI_vs_CN'

# 模型配置 - 与 train_dhgn.py 中的 TASK_CONFIGS['EMCI_vs_CN'] 保持一致
MODEL_CONFIG = {
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

def extract_embeddings(model, dynamic_fc, sc_matrix, device, batch_size=16):
    """提取模型学到的特征表示"""
    model.eval()
    embeddings_list = []
    
    num_samples = len(dynamic_fc)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_dyn = torch.FloatTensor(dynamic_fc[i:i+batch_size]).to(device)
            batch_sc = torch.FloatTensor(sc_matrix[i:i+batch_size]).to(device)
            
            # 提取时空特征
            st_features = model.dynamic_imaging_pathway(batch_dyn)
            embeddings_list.append(st_features.cpu().numpy())
            
    return np.concatenate(embeddings_list, axis=0)

def main():
    device = setup_gpu()
    
    # 1. 加载数据
    print("📊 加载数据 (EMCI + CN)...")
    loader = DHGNDataLoader()
    dynamic_fc, sc_matrix, labels, _ = loader.load_all_data(groups=['EMCI', 'CN'])
    
    print(f"✅ 数据形状: DFC={dynamic_fc.shape}, SC={sc_matrix.shape}")
    print(f"   标签分布: EMCI={np.sum(labels==0)}, CN={np.sum(labels==1)}")
    
    # 2. 加载训练好的模型（使用 Fold 0 作为代表）
    print(f"\n🔧 加载模型...")
    model = create_dhgn_model(config=MODEL_CONFIG).to(device)
    
    # 尝试加载最佳 Fold
    import glob
    model_paths = glob.glob(os.path.join(CKPT_DIR, TASK, 'fold_*_best.pth'))
    
    if not model_paths:
        print(f"❌ 未找到训练好的模型: {os.path.join(CKPT_DIR, TASK)}")
        print("   请先运行 train_dhgn.py 训练模型")
        return
    
    # 使用 Fold 0
    model_path = sorted(model_paths)[0]
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 加载模型: {os.path.basename(model_path)}")
    
    # 3a. 可视化原始 DFC 特征（未经模型处理）
    print(f"\n📊 提取原始 DFC 特征（未经模型训练）...")
    # 将每个样本的 DFC 矩阵展平为向量：[N, T, ROI, ROI] -> [N, T*ROI*ROI]
    raw_features = dynamic_fc.reshape(len(dynamic_fc), -1)
    print(f"✅ 原始特征维度: {raw_features.shape}")
    
    # PCA 降维原始特征
    print(f"📉 对原始特征进行 PCA 降维...")
    pca_raw = PCA(n_components=2)
    pca_raw_result = pca_raw.fit_transform(raw_features)
    
    # 绘制原始特征 PCA 图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_raw, ax_raw = plt.subplots(1, 1, figsize=(10, 8))
    label_map = {0: 'EMCI', 1: 'CN'}
    colors = {0: '#ff7f0e', 1: '#1f77b4'}
    for lbl in [0, 1]:
        mask = labels == lbl
        ax_raw.scatter(pca_raw_result[mask, 0], pca_raw_result[mask, 1],
                       c=colors[lbl], label=label_map[lbl],
                       alpha=0.7, s=80, edgecolors='w', linewidth=1.5)
    ax_raw.set_title('EMCI vs CN Raw DFC Features (Before Model Training)', fontsize=18, fontweight='bold', pad=20)
    ax_raw.legend(fontsize=14, loc='best', framealpha=0.9)
    ax_raw.grid(True, linestyle='--', alpha=0.3)
    ax_raw.set_xlabel('PCA Component 1', fontsize=12)
    ax_raw.set_ylabel('PCA Component 2', fontsize=12)
    plt.tight_layout()
    
    save_dir = 'analysis_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    raw_save_path = os.path.join(save_dir, 'raw_dfc_pca.png')
    plt.savefig(raw_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存原始特征图至: {raw_save_path}")
    plt.close()
    
    # 3b. 提取模型学习后的特征
    print(f"\n🧠 提取模型学习后的特征...")
    embeddings = extract_embeddings(model, dynamic_fc, sc_matrix, device)
    print(f"✅ 模型特征维度: {embeddings.shape}")
    
    # 4. t-SNE 降维
    print(f"\n📉 运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(embeddings)
    print(f"✅ t-SNE 完成")
    
    # 5. 可视化
    print(f"\n🎨 生成可视化...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 颜色: EMCI=橙色, CN=蓝色
    label_map = {0: 'EMCI', 1: 'CN'}
    colors = {0: '#ff7f0e', 1: '#1f77b4'}
    
    for lbl in [0, 1]:
        mask = labels == lbl
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   c=colors[lbl], label=label_map[lbl], 
                   alpha=0.7, s=80, edgecolors='w', linewidth=1.5)
    
    ax.set_title('EMCI vs CN Feature Distribution (t-SNE)', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    save_dir = 'analysis_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'tsne_emci_cn.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
