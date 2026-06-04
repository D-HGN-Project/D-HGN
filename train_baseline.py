
"""
Baseline Model Training Script
批量训练所有对比模型并生成汇总报告
用法: python train_baseline.py --gpu_id 1
"""
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import pandas as pd
from datetime import datetime

# 复用现有的工具
from data_loader import DHGNDataLoader
from gpu_utils import setup_gpu
from baseline_models import create_baseline_model

# 固定随机种子
SEED = 888 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 定义要运行的基线模型列表
BASELINE_MODELS = [
    # 'mlp', 
    # 'brainnetcnn', 
    'gcn', 
    'gat',
    'stgcn',
    'diffpool',
    # 'lstm', 
    'tcn',
    'transformer',
    'graphsage',
    'gin',
    'bilstm',
    'gru'
]

def calculate_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    TP = ((preds_np == 1) & (labels_np == 1)).sum()
    TN = ((preds_np == 0) & (labels_np == 0)).sum()
    FP = ((preds_np == 1) & (labels_np == 0)).sum()
    FN = ((preds_np == 0) & (labels_np == 1)).sum()
    
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    sen = TP / (TP + FN + 1e-8)
    spe = TN / (TN + FP + 1e-8)
    
    return acc, sen, spe

class BaselineTrainer:
    def __init__(self, model_name, args, device, data_loader_obj):
        self.model_name = model_name
        self.args = args
        self.device = device
        self.loader = data_loader_obj
        self.results = []
        
        # 创建检查点目录
        self.ckpt_dir = os.path.join(args.ckpt_path, 'baselines', model_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def run_cv(self):
        print(f"\n{'='*20} 开始训练模型: {self.model_name.upper()} {'='*20}")
        
        # 1. 准备数据 (只加载一次，所有模型共用)
        dynamic_fc = self.loader.dynamic_fc
        labels = self.loader.labels
        cv_splits = self.loader.cv_splits # 复用划分以保证公平对比
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            print(f"  Fold {fold}...", end='', flush=True)
            
            # 数据准备
            # 这里简化处理，直接用Tensor切片，不重复做详细的归一化检查
            # 因为Baseline主要是看相对趋势
            train_x = dynamic_fc[train_idx].to(self.device)
            train_y = labels[train_idx].to(self.device)
            test_x = dynamic_fc[test_idx].to(self.device)
            test_y = labels[test_idx].to(self.device)
            
            # 创建模型
            model = create_baseline_model(self.model_name).to(self.device)
            
            # 优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss() # Baseline一般不用加权Loss，或者是简单的
            
            # 训练循环
            best_acc = 0
            best_fold_res = (0, 0, 0, 0) # Acc, Sen, Spe, AUC
            
            for epoch in range(self.args.num_epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward
                logits = model(train_x)
                loss = criterion(logits, train_y)
                
                loss.backward()
                optimizer.step()
                
                # Eval
                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        test_logits = model(test_x)
                        acc, sen, spe = calculate_metrics(test_logits, test_y)
                        
                        # 简单的AUC (Softmax prob)
                        probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
                        try:
                            from sklearn.metrics import roc_auc_score
                            auc = roc_auc_score(test_y.cpu().numpy(), probs)
                        except:
                            auc = 0.5
                            
                        if acc > best_acc:
                            best_acc = acc
                            best_fold_res = (acc, sen, spe, auc)
                            # 保存最佳模型
                            torch.save(model.state_dict(), os.path.join(self.ckpt_dir, f'fold_{fold}_best.pth'))
            
            print(f" Best Acc: {best_acc:.4f}")
            fold_metrics.append(best_fold_res)
            
        # 汇总当前模型结果
        avg_acc = np.mean([m[0] for m in fold_metrics])
        std_acc = np.std([m[0] for m in fold_metrics])
        avg_sen = np.mean([m[1] for m in fold_metrics])
        std_sen = np.std([m[1] for m in fold_metrics])
        avg_spe = np.mean([m[2] for m in fold_metrics])
        std_spe = np.std([m[2] for m in fold_metrics])
        avg_auc = np.mean([m[3] for m in fold_metrics])
        std_auc = np.std([m[3] for m in fold_metrics])
        
        print(f"✅ {self.model_name.upper()} 完成 | Mean Acc: {avg_acc:.4f} ± {std_acc:.4f}")
        
        return {
            'Model': self.model_name.upper(),
            'Accuracy': f"{avg_acc:.4f} ± {std_acc:.4f}",
            'Sensitivity': f"{avg_sen:.4f} ± {std_sen:.4f}",
            'Specificity': f"{avg_spe:.4f} ± {std_spe:.4f}",
            'AUC': f"{avg_auc:.4f} ± {std_auc:.4f}"
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--groups', nargs='+', default=['EMCI', 'CN'])
    parser.add_argument('--num_epochs', type=int, default=100) # Baseline一般收敛快
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    device = setup_gpu(gpu_id=args.gpu_id)
    
    print("\n📊 加载数据...")
    loader = DHGNDataLoader(data_root=args.data_root)
    # 加载数据，不使用SC (Baseline如果不特别支持SC，我们就只用DFC)
    # baseline_models.py里的模型大多只设计了处理 dynamic_fc
    raw_dyn, _, raw_labels, _ = loader.load_all_data(groups=args.groups, use_sc=False)
    
    # 简单的全局归一化
    raw_dyn = (raw_dyn - np.mean(raw_dyn)) / (np.std(raw_dyn) + 1e-8)
    
    # 封装便于传递
    class DataContainer:
        pass
    data_container = DataContainer()
    data_container.dynamic_fc = torch.FloatTensor(raw_dyn)
    data_container.labels = torch.LongTensor(raw_labels)
    data_container.cv_splits = loader.data_split(raw_labels, n_folds=5)
    
    all_results = []
    
    print(f"\n🚀 开始批量训练对比模型 ({len(BASELINE_MODELS)} models)")
    print(f"   列表: {BASELINE_MODELS}")
    
    for model_name in BASELINE_MODELS:
        trainer = BaselineTrainer(model_name, args, device, data_container)
        try:
            res = trainer.run_cv()
            all_results.append(res)
        except Exception as e:
            print(f"❌ 模型 {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
    # 生成最终报告
    print("\n\n")
    print("="*60)
    print("🏆 最终对比实验结果汇总")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    
    # 提取准确率均值用于排序 (高到低)，标准差作为第二排序键 (低到高，即更稳定优先)
    # 格式: "0.8500 ± 0.0200"
    df['Acc_Mean'] = df['Accuracy'].apply(lambda x: float(x.split(' ± ')[0]))
    df['Acc_Std'] = df['Accuracy'].apply(lambda x: float(x.split(' ± ')[1]))
    
    # 排序: 均值降序，标准差升序
    df = df.sort_values(by=['Acc_Mean', 'Acc_Std'], ascending=[False, True])
    
    # 调整列顺序 (不包含临时排序列)
    cols = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    df = df[cols]
    
    print(df.to_string(index=False))
    
    # 保存CSV
    csv_path = os.path.join(args.ckpt_path, f'baseline_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")

if __name__ == '__main__':
    main()
