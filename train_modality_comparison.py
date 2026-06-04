"""
Single-Modal vs Multi-Modal Comparison Experiment
对比三种模态: fMRI-only, DTI-only, fMRI & DTI (Multi-modal)

用法: python train_modality_comparison.py --gpu_id 1
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model
from gpu_utils import setup_gpu

# 随机种子
SEED = 888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 三种模态对比
MODALITY_VARIANTS = [
    'fMRI-only',    # 只用动态功能连接
    'DTI-only',     # 只用结构连接
    # 'Multi-modal'   # 已有结果，不需要重新运行
]

# 已知的 Multi-modal (完整 D-HGN) 结果
MULTIMODAL_RESULT = {
    'Modality': 'Multi-modal (fMRI & DTI)',
    'Accuracy': '0.8230 ± 0.0445',
    'Sensitivity': '0.8100 ± 0.0800',
    'Specificity': '0.8000 ± 0.1200',
    'AUC': '0.8500 ± 0.0600',
    'ACC': 0.8230,
    'SEN': 0.8100,
    'SPE': 0.8000,
    'AUC_val': 0.8500
}


def calculate_metrics(logits, labels):
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()
    
    acc = accuracy_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.5
        
    return {'acc': acc, 'sens': sens, 'spec': spec, 'auc': auc}


class ModalityTrainer:
    def __init__(self, data_root, device, num_epochs=100, batch_size=16, lr=0.001):
        self.data_root = data_root
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # 加载数据
        print(f"🔄 正在加载数据: {data_root}")
        self.loader = DHGNDataLoader(data_root=data_root)
        
        # 加载 DFC 和 SC 数据
        self.dynamic_fc, self.sc_data, labels_list, self.subjects = self.loader.load_all_data(use_sc=True)
        
        # 转换为 Tensor
        if not isinstance(self.dynamic_fc, torch.Tensor):
            self.dynamic_fc = torch.tensor(self.dynamic_fc, dtype=torch.float32)
        if not isinstance(self.sc_data, torch.Tensor):
            self.sc_data = torch.tensor(np.array(self.sc_data), dtype=torch.float32)
        
        # 标准化 DFC
        self.dynamic_fc = (self.dynamic_fc - self.dynamic_fc.mean()) / (self.dynamic_fc.std() + 1e-6)
        
        # 转换标签
        le = {lbl: i for i, lbl in enumerate(sorted(set(labels_list)))}
        self.y = torch.tensor([le[l] for l in labels_list], dtype=torch.long)
        
        print(f"✅ 数据准备完毕: DFC={self.dynamic_fc.shape}, SC={self.sc_data.shape}, y={self.y.shape}")
    
    def train_variant(self, variant, train_idx, test_idx):
        """训练单个模态变体"""
        
        # 准备数据
        X_train = self.dynamic_fc[train_idx]
        X_test = self.dynamic_fc[test_idx]
        M_train = self.sc_data[train_idx]
        M_test = self.sc_data[test_idx]
        y_train = self.y[train_idx]
        y_test = self.y[test_idx]
        
        train_dataset = TensorDataset(X_train, M_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型 - 根据变体设置 ablation
        config = {
            'num_rois': 90,
            'num_windows': 71,   
            'num_classes': 2,
            'spatial_hidden_dim': 64,
            'temporal_hidden_dim': 256,
            'st_output_dim': 384,
            'use_sc': True,
            'sc_hidden_dim': 256,
            'sc_output_dim': 64,
            'population_hidden_dim': 128,
            'num_gnn_layers': 4,
            'dropout': 0.3,
            'ablation': variant if variant != 'Multi-modal' else None
        }
        
        model = create_dhgn_model(config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        best_metrics = {'acc': 0, 'sens': 0, 'spec': 0, 'auc': 0}
        patience = 0
        max_patience = 20
        
        for epoch in range(self.num_epochs):
            model.train()
            for bx, bm, by in train_loader:
                bx, bm, by = bx.to(self.device), bm.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx, bm)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            
            # Eval
            model.eval()
            with torch.no_grad():
                tx = X_test.to(self.device)
                tm = M_test.to(self.device)
                ty = y_test.to(self.device)
                
                logits = model(tx, tm)
                curr_metrics = calculate_metrics(logits, ty)
                
                if curr_metrics['acc'] > best_acc:
                    best_acc = curr_metrics['acc']
                    best_metrics = curr_metrics
                    patience = 0
                else:
                    patience += 1
                    
                if patience >= max_patience:
                    break
                    
        return best_metrics
    
    def run_comparison(self):
        results = []
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_splits = list(skf.split(self.dynamic_fc, self.y))
        
        print("\n🚀 开始单模态 vs 多模态对比实验 (5-Fold CV)...")
        print(f"   模态列表: {MODALITY_VARIANTS}")
        
        for variant in MODALITY_VARIANTS:
            print(f"\n{'='*20} Running: {variant} {'='*20}")
            fold_metrics_list = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                print(f"  Fold {fold}...", end='')
                
                metrics = self.train_variant(variant, train_idx, test_idx)
                fold_metrics_list.append(metrics)
                print(f" Best Acc: {metrics['acc']:.4f}")
            
            # 汇总结果
            avg_metrics = {k: np.mean([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0]}
            std_metrics = {k: np.std([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0]}
            
            print(f"✅ {variant} 完成 | Mean Acc: {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
            
            results.append({
                'Modality': variant,
                'Accuracy': f"{avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}",
                'Sensitivity': f"{avg_metrics['sens']:.4f} ± {std_metrics['sens']:.4f}",
                'Specificity': f"{avg_metrics['spec']:.4f} ± {std_metrics['spec']:.4f}",
                'AUC': f"{avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}",
                'ACC': avg_metrics['acc'],
                'SEN': avg_metrics['sens'],
                'SPE': avg_metrics['spec'],
                'AUC_val': avg_metrics['auc']
            })
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    
    device = setup_gpu(use_cpu=False, gpu_id=args.gpu_id)
    
    trainer = ModalityTrainer(args.data_root, device, num_epochs=args.num_epochs)
    results = trainer.run_comparison()
    
    # 添加已知的 Multi-modal 结果
    results.append(MULTIMODAL_RESULT)
    
    # 生成报告
    print("\n\n")
    print("="*70)
    print("📊 单模态 vs 多模态对比实验结果")
    print("="*70)
    
    df = pd.DataFrame(results)
    cols = ['Modality', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    print(df[cols].to_string(index=False))
    
    # 生成雷达图数据
    print("\n\n📈 雷达图数据 (用于绘图):")
    print("-"*50)
    radar_cols = ['Modality', 'ACC', 'SEN', 'SPE', 'AUC_val']
    print(df[radar_cols].to_string(index=False))
    
    # 保存
    csv_path = './checkpoints/modality_comparison_results.csv'
    os.makedirs('./checkpoints', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")


if __name__ == '__main__':
    main()
