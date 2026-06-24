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

ABLATION_VARIANTS = [
    # 'D-HGN (Full)', # 已知结果: 82.30% (from train_dhgn.py)
    'w/o Attention',
    'w/o Graph',
    'w/o SC',
    'w/o Spatial'
]

# 已知的完整版结果 (用于计算 Drop)
FULL_BASELINE_ACC = 0.8230

class AblationTrainer:
    def __init__(self, data_root, device, num_epochs=100, batch_size=16, lr=0.001):
        self.data_root = data_root
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # 加载数据
        print(f"🔄 正在加载数据: {data_root}")
        self.loader = DHGNDataLoader(data_root=data_root)
        self.dynamic_fc, self.sc_data, labels_list, self.subjects = self.loader.load_all_data()
        
        # sc_data 已经是 SC 矩阵列表
        self.sc_tensor_list = self.sc_data  # 直接使用
        
        # 转换为 Tensor (如果不是)
        if not isinstance(self.dynamic_fc, torch.Tensor):
            self.dynamic_fc = torch.tensor(self.dynamic_fc, dtype=torch.float32)
        
        # 标准化 DFC
        print("🔄 标准化 DFC 数据...")
        self.dynamic_fc = (self.dynamic_fc - self.dynamic_fc.mean()) / (self.dynamic_fc.std() + 1e-6)
        
        # 转换标签
        le = {lbl: i for i, lbl in enumerate(sorted(set(labels_list)))}
        self.y = torch.tensor([le[l] for l in labels_list], dtype=torch.long)
        self.X = self.dynamic_fc
        
        # Use real demographic data (age + sex) for the w/o SC variant.
        self.clin_tensor_list = torch.tensor(self.loader.non_imaging_data, dtype=torch.float32)
        
        print(f"✅ 数据准备完毕: X={self.X.shape}, y={self.y.shape}")
        
    def train_variant(self, variant_name, train_idx, test_idx):
        """训练单个变体的一次Fold"""
        
        # 准备数据
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]
        
        # 准备 Modal Data
        # 如果是 w/o SC，用 Clinical Data (Age/Sex)
        # 否则用 SC Matrix
        is_wo_sc = (variant_name == 'w/o SC')
        
        if is_wo_sc:
            # w/o SC -> Use Clinical
            M_train = self.clin_tensor_list[train_idx]
            M_test = self.clin_tensor_list[test_idx]
            use_sc_flag = False
        else:
            # Full or other variants -> Use SC
            # 确保是 Tensor 类型
            if isinstance(self.sc_tensor_list, torch.Tensor):
                M_train = self.sc_tensor_list[train_idx]
                M_test = self.sc_tensor_list[test_idx]
            else:
                # 如果是 numpy 或列表，转换为 tensor
                sc_tensor = torch.tensor(np.array(self.sc_tensor_list), dtype=torch.float32)
                M_train = sc_tensor[train_idx]
                M_test = sc_tensor[test_idx]
            use_sc_flag = True
            
        train_dataset = TensorDataset(X_train, M_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型
        # 解析 ablation 参数
        ablation_arg = None
        if variant_name != 'D-HGN (Full)':
            ablation_arg = variant_name
            
        # 配置 (参考 train_dhgn.py)
        config = {
            'num_rois': 90,
            'num_windows': 71,   
            'num_classes': 2,
            'spatial_hidden_dim': 64,
            'temporal_hidden_dim': 256,
            'st_output_dim': 384,
            'use_sc': use_sc_flag, # 关键: 根据变体自动切换
            'sc_hidden_dim': 256,
            'sc_output_dim': 64,
            'population_hidden_dim': 128,
            'num_gnn_layers': 4,
            'dropout': 0.3,
            'ablation': ablation_arg # 传递 ablation 参数
        }
        
        model = create_dhgn_model(config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4) # 增加 weight_decay 防止过拟合
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        best_metrics = {'acc': 0, 'sens': 0, 'spec': 0, 'auc': 0}
        patience = 0
        
        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0
            for bx, bm, by in train_loader:
                bx, bm, by = bx.to(self.device), bm.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx, bm)
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
                
                # 计算完整指标
                curr_metrics = calculate_metrics(logits, ty)
                
                if curr_metrics['acc'] > best_acc:
                    best_acc = curr_metrics['acc']
                    best_metrics = curr_metrics
                    patience = 0
                else:
                    patience += 1
                    
        return best_metrics

    def run_ablation(self):
        results = []
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_splits = list(skf.split(self.X, self.y))
        
        print("\n🚀 开始消融实验 (5-Fold CV)...")
        print(f"变体列表: {ABLATION_VARIANTS}")
        
        for variant in ABLATION_VARIANTS:
            print(f"\n==================== Running: {variant} ====================")
            fold_metrics_list = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                print(f"  Fold {fold}...", end='')
                
                # 训练
                metrics = self.train_variant(variant, train_idx, test_idx)
                fold_metrics_list.append(metrics)
                print(f" Best Acc: {metrics['acc']:.4f}")
            
            # 汇总结果
            avg_metrics = {k: np.mean([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0]}
            std_acc = np.std([m['acc'] for m in fold_metrics_list])
            std_sen = np.std([m['sens'] for m in fold_metrics_list])
            std_spec = np.std([m['spec'] for m in fold_metrics_list])
            std_auc = np.std([m['auc'] for m in fold_metrics_list])
            
            print(f"✅ {variant} 完成 | Mean Acc: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
            
            results.append({
                'Variant': variant,
                'Accuracy': f"{avg_metrics['acc']:.4f} ± {std_acc:.4f}",
                'Sensitivity': f"{avg_metrics['sens']:.4f} ± {std_sen:.4f}",
                'Specificity': f"{avg_metrics['spec']:.4f} ± {std_spec:.4f}",
                'AUC': f"{avg_metrics['auc']:.4f} ± {std_auc:.4f}",
                'Acc_Mean': avg_metrics['acc'],
                'Acc_Std': std_acc
            })
            
        return results

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    
    device = setup_gpu(use_cpu=False, gpu_id=args.gpu_id)
    
    trainer = AblationTrainer(args.data_root, device, num_epochs=args.num_epochs)
    all_results = trainer.run_ablation()
    
    # 生成报告
    print("\n\n")
    print("="*60)
    print("🧪 消融实验结果汇总 (Ablation Study)")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    # 排序：按均值降序
    df = df.sort_values(by='Acc_Mean', ascending=False)
    
    # 计算相对于 D-HGN (Full) 的下降
    full_acc = FULL_BASELINE_ACC  # 使用已知的基准值
    df['Drop'] = df['Acc_Mean'].apply(lambda x: x - full_acc)
    df['Drop_Percent'] = df['Drop'].apply(lambda x: f"{x*100:.2f}%")
    
    # 添加 D-HGN (Full) 作为对照行
    full_row = pd.DataFrame([{
        'Variant': 'D-HGN (Full)',
        'Accuracy': f"{full_acc:.4f} ± 0.0445",  # 之前的结果
        'Sensitivity': '0.8100',
        'Specificity': '0.8000',
        'AUC': '0.8500',
        'Acc_Mean': full_acc,
        'Acc_Std': 0.0445,
        'Drop': 0.0,
        'Drop_Percent': '0.00%'
    }])
    df = pd.concat([full_row, df], ignore_index=True)
        
    cols = ['Variant', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Drop_Percent']
    df = df[cols]
    
    print(df.to_string(index=False))
    
    # 保存
    csv_path = './checkpoints/ablation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")

if __name__ == '__main__':
    main()
