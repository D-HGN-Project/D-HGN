"""
DTI-Only Ablation Experiment (w/o fMRI)
使用 D-HGN 架构，但忽略 fMRI 动态功能连接
验证 fMRI 的贡献
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


class DTIOnlyTrainer:
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
        
        # 标准化 DFC (虽然 w/o fMRI 不用，但保持接口一致)
        self.dynamic_fc = (self.dynamic_fc - self.dynamic_fc.mean()) / (self.dynamic_fc.std() + 1e-6)
        
        # 转换标签
        le = {lbl: i for i, lbl in enumerate(sorted(set(labels_list)))}
        self.y = torch.tensor([le[l] for l in labels_list], dtype=torch.long)
        
        print(f"✅ 数据准备完毕: DFC={self.dynamic_fc.shape}, SC={self.sc_data.shape}, y={self.y.shape}")
    
    def train_fold(self, train_idx, test_idx):
        """训练单个 Fold"""
        
        # 准备数据
        X_train = self.dynamic_fc[train_idx]
        X_test = self.dynamic_fc[test_idx]
        M_train = self.sc_data[train_idx]
        M_test = self.sc_data[test_idx]
        y_train = self.y[train_idx]
        y_test = self.y[test_idx]
        
        train_dataset = TensorDataset(X_train, M_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建 D-HGN 模型，设置 ablation='w/o fMRI'
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
            'ablation': 'w/o fMRI'  # 关键：禁用 fMRI
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
            train_loss = 0
            for bx, bm, by in train_loader:
                bx, bm, by = bx.to(self.device), bm.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx, bm)  # 内部会忽略 bx (DFC)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Eval
            model.eval()
            with torch.no_grad():
                tx = X_test.to(self.device)
                tm = M_test.to(self.device)
                ty = y_test.to(self.device)
                
                logits = model(tx, tm)
                
                curr_metrics = self.calculate_metrics(logits, ty)
                
                if curr_metrics['acc'] > best_acc:
                    best_acc = curr_metrics['acc']
                    best_metrics = curr_metrics
                    patience = 0
                else:
                    patience += 1
                    
                if patience >= max_patience:
                    break
                    
        return best_metrics
    
    def calculate_metrics(self, logits, labels):
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
    
    def run_cv(self):
        print("\n🚀 开始 DTI-Only (w/o fMRI) 实验 (5-Fold CV)...")
        print("   使用 D-HGN 架构，ablation='w/o fMRI'")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_splits = list(skf.split(self.dynamic_fc, self.y))
        
        fold_metrics_list = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            print(f"  Fold {fold}...", end='')
            
            metrics = self.train_fold(train_idx, test_idx)
            fold_metrics_list.append(metrics)
            print(f" Best Acc: {metrics['acc']:.4f}")
        
        # 汇总结果
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0]}
        std_acc = np.std([m['acc'] for m in fold_metrics_list])
        std_sen = np.std([m['sens'] for m in fold_metrics_list])
        std_spec = np.std([m['spec'] for m in fold_metrics_list])
        std_auc = np.std([m['auc'] for m in fold_metrics_list])
        
        print(f"\n✅ DTI-Only (w/o fMRI) 完成 | Mean Acc: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        
        result = {
            'Variant': 'w/o fMRI (DTI-Only)',
            'Accuracy': f"{avg_metrics['acc']:.4f} ± {std_acc:.4f}",
            'Sensitivity': f"{avg_metrics['sens']:.4f} ± {std_sen:.4f}",
            'Specificity': f"{avg_metrics['spec']:.4f} ± {std_spec:.4f}",
            'AUC': f"{avg_metrics['auc']:.4f} ± {std_auc:.4f}",
            'Acc_Mean': avg_metrics['acc'],
            'Acc_Std': std_acc
        }
        
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    
    device = setup_gpu(use_cpu=False, gpu_id=args.gpu_id)
    
    trainer = DTIOnlyTrainer(args.data_root, device, num_epochs=args.num_epochs)
    result = trainer.run_cv()
    
    # 生成报告
    print("\n\n")
    print("="*60)
    print("🧪 DTI-Only (w/o fMRI) 消融实验结果")
    print("="*60)
    
    df = pd.DataFrame([result])
    
    # 与完整 D-HGN 对比
    FULL_BASELINE_ACC = 0.8230
    df['Drop'] = df['Acc_Mean'].apply(lambda x: x - FULL_BASELINE_ACC)
    df['Drop_Percent'] = df['Drop'].apply(lambda x: f"{x*100:.2f}%")
    
    cols = ['Variant', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Drop_Percent']
    print(df[cols].to_string(index=False))
    
    # 保存
    csv_path = './checkpoints/dti_only_results.csv'
    os.makedirs('./checkpoints', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")


if __name__ == '__main__':
    main()
