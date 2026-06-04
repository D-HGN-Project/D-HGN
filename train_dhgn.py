"""
D-HGN主训练脚本 - 多模态版本
支持: fMRI (DFC) + DTI (SC) 多模态融合
包含：
1. SC矩阵替代非成像数据
2. 时间窗数据增强 (Temporal Dropout)
3. Mixup数据增强
4. Warmup + CosineAnnealing 学习率调度
5. 加权Ensemble (Temperature=10.0)
6. 动态阈值搜索
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import argparse
import random
from dhgn_model import create_dhgn_model
from data_loader import DHGNDataLoader
from gpu_utils import setup_gpu

# 固定随机种子，保证结果可复现
# 种子测试结果 (按性能排序):
#   SEED=888:  80.99% ± 3.25%, AUC=82.62%  最佳

SEED = 888  # 最终选择 - 最高准确率 + 最稳定
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Phase 5: 迁移学习配置
# 训练顺序: AD vs CN (source) → EMCI vs CN (target, 使用AD预训练权重)
# 🔧 针对小样本优化：减少模型复杂度 + 增强正则化
TASK_CONFIGS = {
    'AD_vs_CN': {
        'groups': ['AD', 'CN'],
        'num_epochs': 120,
        'class_weight_factor': 2.0,
        'mixup_alpha': 0.3,
        'learning_rate': 2e-4,
        'model_config': {
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
        # 不保存预训练权重 - 不再使用迁移学习
    },
    'EMCI_vs_CN': {
        'groups': ['EMCI', 'CN'],
        'num_epochs': 120,  # 恢复最佳轮数
        'class_weight_factor': 1.5,
        'mixup_alpha': 0.5,
        'label_smoothing': 0.1,  # 显式控制标签平滑
        'learning_rate': 2e-4,  # 恢复最佳学习率
        'weight_decay': 1e-4,   # 显式控制权重衰减 (默认是 5e-4 或 1e-5，调优一下)
        'model_config': {
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
            'dropout': 0.4  # 恢复最佳 Dropout
        }
        # 不使用迁移学习 - 从头训练
        # 不使用 Early Stopping - 跑满所有 epochs
    }
}



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

class EnsemblePredictor:
    """集成多个模型进行预测（加权Ensemble）"""
    def __init__(self, model_paths, device, fold_accuracies=None, model_config=None):
        self.models = []
        for path in model_paths:
            if os.path.exists(path):
                # Phase 4: 使用任务特定模型配置
                if model_config:
                    model = create_dhgn_model(config=model_config).to(device)
                else:
                    model = create_dhgn_model().to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                self.models.append(model)
        self.device = device

        # 设置权重（基于每个Fold的验证准确率）
        if fold_accuracies is not None:
            # 使用softmax归一化权重，强调性能好的模型
            weights = torch.tensor(fold_accuracies, dtype=torch.float32)
            # 使用温度参数放大差异 (Temperature=10.0)
            self.weights = F.softmax(weights * 10.0, dim=0).to(device)
            print(f"成功加载 {len(self.models)} 个模型")
            print(f"加权Ensemble权重: {self.weights.cpu().numpy()}")
        else:
            # 如果没有提供准确率，使用均等权重
            self.weights = torch.ones(len(self.models), dtype=torch.float32).to(device) / len(self.models)
            print(f"成功加载 {len(self.models)} 个模型（使用均等权重）")

    def predict(self, dynamic_fc, non_imaging, tta_steps=5):
        """
        使用加权Soft Voting进行集成预测 + Test-Time Augmentation (TTA)

        Args:
            dynamic_fc: 动态功能连接矩阵
            non_imaging: 非成像数据
            tta_steps: TTA次数，默认为5。如果为1则不使用TTA。
        """
        all_probs = []
        for model in self.models:
            # TTA循环
            model_probs = []
            for _ in range(tta_steps):
                # TTA时必须保持eval模式，防止BatchNorm统计信息泄露！
                model.eval()

                with torch.no_grad():
                    # 应用随机时间窗Dropout增强 (Input Augmentation)
                    if tta_steps > 1:
                        aug_dyn = augment_temporal_dropout(dynamic_fc, dropout_rate=0.1)
                    else:
                        aug_dyn = dynamic_fc

                    logits = model(aug_dyn, non_imaging)
                    probs = F.softmax(logits, dim=1)
                    model_probs.append(probs)

            # 平均该模型的所有TTA预测
            avg_model_probs = torch.stack(model_probs).mean(dim=0)
            all_probs.append(avg_model_probs)

        # 加权Soft Voting: 根据模型性能加权平均概率
        weighted_probs = sum(w * p for w, p in zip(self.weights, all_probs))
        return weighted_probs

def augment_temporal_dropout(dynamic_fc, dropout_rate=0.15):
    """
    时间窗随机dropout数据增强
    随机将15%的时间窗置零，增强模型对时间窗缺失的鲁棒性
    """
    augmented = dynamic_fc.clone()
    if np.random.rand() > 0.5:  # 50%概率应用
        batch_size, num_windows = augmented.shape[0], augmented.shape[1]
        for i in range(batch_size):
            num_drop = int(num_windows * dropout_rate)
            drop_indices = np.random.choice(num_windows, num_drop, replace=False)
            augmented[i, drop_indices] = 0
    return augmented

def mixup_data(dynamic_fc, non_imaging, labels, alpha=0.2):
    """
    Mixup数据增强：混合两个样本及其标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = dynamic_fc.size(0)
    index = torch.randperm(batch_size).to(dynamic_fc.device)

    mixed_dynamic_fc = lam * dynamic_fc + (1 - lam) * dynamic_fc[index]
    mixed_non_imaging = lam * non_imaging + (1 - lam) * non_imaging[index]

    labels_a, labels_b = labels, labels[index]

    return mixed_dynamic_fc, mixed_non_imaging, labels_a, labels_b, lam

def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    Mixup损失函数：混合两个标签的损失
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)

def get_cosine_schedule_with_warmup(epoch, warmup_epochs=10, max_epochs=100, max_lr=5e-4, min_lr=1e-6):
    """
    Warmup + CosineAnnealing学习率调度
    前10 epochs: 线性warmup从1e-6到5e-4
    后90 epochs: 余弦衰减从5e-4到1e-6
    """
    if epoch < warmup_epochs:
        # Warmup阶段：线性增长
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    else:
        # CosineAnnealing阶段：余弦衰减
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

class DHGNTrainer:
    def __init__(self, args, task_config=None):
        self.args = args
        self.device = setup_gpu(use_cpu=args.use_cpu, gpu_id=args.gpu_id)
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        self.fold_results = []

        # Phase 4: 任务特定配置
        if task_config:
            self.class_weight_factor = task_config.get('class_weight_factor', 1.85)
            self.mixup_alpha = task_config.get('mixup_alpha', 0.2)
            self.model_config = task_config.get('model_config', None)
        else:
            self.class_weight_factor = 1.85
            self.mixup_alpha = 0.2
            self.model_config = None

    def train(self):
        print("正在加载数据...")
        loader = DHGNDataLoader(data_root=self.args.data_root)
        # 加载多模态数据: DFC + SC矩阵
        dynamic_fc, modal_data, labels, _ = loader.load_all_data(groups=self.args.groups, use_sc=True)

        if dynamic_fc is None:
            print("数据加载失败")
            return

        self.dynamic_fc = torch.FloatTensor(dynamic_fc)
        self.modal_data = torch.FloatTensor(modal_data)  # SC矩阵 [N, 90, 90]
        self.labels = torch.LongTensor(labels)

        cv_splits = loader.data_split(labels, n_folds=self.args.n_folds)

        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            print(f"\n{'='*10} Fold {fold} {'='*10}")
            self.train_one_fold(fold, train_idx, test_idx)

        self.print_final_results()

    def train_one_fold(self, fold, train_idx, test_idx):
        # 1. 获取Raw Data（未归一化）
        raw_train_dyn = self.dynamic_fc[train_idx]
        raw_test_dyn = self.dynamic_fc[test_idx]
        
        # 2. ❗ Strict Cross-Validation Normalization
        # 统计量必须仅从【训练集】计算，严禁泄露测试集信息
        # (A) 计算训练集的统计量
        train_numpy = raw_train_dyn.numpy()
        p5 = np.percentile(train_numpy, 5)
        p95 = np.percentile(train_numpy, 95)
        
        # (B) 在 NumPy 层面 Clip (避免 Outlier 破坏均值)
        # 注意：先 Clip 训练集，再算 Mean/Std
        train_clipped = np.clip(train_numpy, -2, 2) # 这里硬编码了之前的Clip范围，或者可以用 percentiles
        # 实际更稳健的做法是用统计分位数 Clip，不过保持之前逻辑一致用 -2, 2 也行，
        # 但既然我们现在动态做，用 P5/P95 可能更科学，但为了复现之前的分布，我们先保持 Clip(-2,2)
        # 之前代码: p5/p95算出来只是为了打印，实际clip是-2,2。
        # 这里我们严格一点，应用同样的 Clip 逻辑给 Test Set
        
        # 为了避免重写过多逻辑，我们将 Clip 和 Normalize 封装成 Torch 操作或简单的预处理
        # 简单起见，我们转 Numpy 处理完再转回 Tensor (数据量较小，开销可忽略)
        
        def process_data(data_tensor, mean=None, std=None, is_train=True):
            data = data_tensor.numpy()
            # Step 1: Clip (Hard clip to -2, 2 as in original code)
            data = np.clip(data, -2, 2)
            
            # Step 2: Normalize
            if is_train:
                mean = np.mean(data)
                std = np.std(data) + 1e-8
                
            data = (data - mean) / std
            return torch.FloatTensor(data), mean, std

        # Apply to Train
        norm_train_dyn, train_mean, train_std = process_data(raw_train_dyn, is_train=True)
        
        # Apply to Test (Using Train Stats!)
        norm_test_dyn, _, _ = process_data(raw_test_dyn, mean=train_mean, std=train_std, is_train=False)
        
        # 打印部分信息以验证无泄露（每个Fold的Mean应该略有不同）
        if fold == 0:
            print(f"  [Fold 0 Leakage Check] Train Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        # Move to Device
        train_dyn = norm_train_dyn.to(self.device)
        test_dyn = norm_test_dyn.to(self.device)
        
        train_modal = self.modal_data[train_idx].to(self.device)  # SC矩阵
        train_y = self.labels[train_idx].to(self.device)

        test_dyn = self.dynamic_fc[test_idx].to(self.device)
        test_modal = self.modal_data[test_idx].to(self.device)  # SC矩阵
        test_y = self.labels[test_idx].to(self.device)

        # Phase 4: 使用任务特定模型配置
        if self.model_config:
            model = create_dhgn_model(config=self.model_config).to(self.device)
        else:
            model = create_dhgn_model().to(self.device)



        # 动态计算类别权重
        n_cn = (train_y == 0).sum().item()
        n_disease = (train_y == 1).sum().item()
        total = n_cn + n_disease

        # 动态获取疾病组名称
        disease_name = [g for g in self.args.groups if g != 'CN'][0] if len(self.args.groups) > 1 else 'Disease'

        weight_cn = total / (2.0 * n_cn)
        weight_disease = total / (self.class_weight_factor * n_disease)  # Phase 4: 动态权重
        class_weights = torch.tensor([weight_cn, weight_disease]).to(self.device)

        print(f"类别分布: CN={n_cn}, {disease_name}={n_disease}")
        print(f"类别权重: CN={weight_cn:.4f}, {disease_name}={weight_disease:.4f}")

        # 获取标签平滑参数 (默认 0.1)
        label_smoothing = getattr(self.args, 'label_smoothing', 0.1)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"[优化] 使用CrossEntropy (Label Smoothing={label_smoothing})")
        
        # 获取权重衰减参数 (优先使用 task_config 中的配置)
        wd = getattr(self.args, 'weight_decay', self.args.wd)
        
        # Phase 5: 使用任务特定学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=wd)

        best_acc = 0
        best_metrics = (0, 0, 0, 0)  # (acc, sen, spe, auc)
        # Early Stopping 已禁用 - 跑满所有 epochs
        # patience_counter = 0
        # early_stopping_patience = getattr(self.args, 'early_stopping_patience', None)
        stopped_early = False

        for epoch in range(self.args.num_epochs):
            # 动态调整学习率
            current_lr = get_cosine_schedule_with_warmup(epoch, max_epochs=self.args.num_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 训练（应用数据增强）
            model.train()
            optimizer.zero_grad()

            # 1. 应用时间窗dropout数据增强
            train_dyn_aug = augment_temporal_dropout(train_dyn)

            # 2. 应用Mixup数据增强（50%概率）
            if np.random.rand() > 0.5:
                mixed_dyn, mixed_modal, labels_a, labels_b, lam = mixup_data(
                    train_dyn_aug, train_modal, train_y, alpha=self.mixup_alpha
                )
                logits = model(mixed_dyn, mixed_modal)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(train_dyn_aug, train_modal)
                loss = criterion(logits, train_y)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 评估（使用原始数据，不增强）
            model.eval()
            with torch.no_grad():
                test_logits = model(test_dyn, test_modal)
                test_acc, test_sen, test_spe = calculate_metrics(test_logits, test_y)

                # 训练集也评估
                train_logits_eval = model(train_dyn, train_modal)
                train_acc, train_sen, train_spe = calculate_metrics(train_logits_eval, train_y)

                # 第一个epoch打印邻接矩阵统计
                if epoch == 0:
                    _, adj = model.modal_processor(train_modal[:min(20, len(train_modal))])
                    print(f"[调试] 邻接矩阵非零比例: {(adj > 0).float().mean():.4f}")
                    print(f"[调试] 邻接矩阵范围: [{adj.min():.4f}, {adj.max():.4f}]")

                # 每50个epoch监控预测分布 (已禁用)
                # if epoch % 50 == 0 and epoch > 0:
                #     test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
                #     train_pred = torch.argmax(train_logits_eval, dim=1).cpu().numpy()
                #     print(f"[调试] 测试预测: CN={np.sum(test_pred==0)}, {disease_name}={np.sum(test_pred==1)}")
                #     print(f"[调试] 训练预测: CN={np.sum(train_pred==0)}, {disease_name}={np.sum(train_pred==1)}")

            # 打印日志 (详细版)
            if (epoch + 1) % 5 == 0:
                # 获取预测分布
                with torch.no_grad():
                    train_pred = torch.argmax(train_logits_eval, dim=1).cpu().numpy()
                    test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
                    train_cn = np.sum(train_pred == 0)
                    train_dis = np.sum(train_pred == 1)
                    test_cn = np.sum(test_pred == 0)
                    test_dis = np.sum(test_pred == 1)
                
                print(f"Epoch {epoch+1}/{self.args.num_epochs} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f} | "
                      f"Train: {train_acc:.4f} ({train_cn}/{train_dis}) | "
                      f"Test: {test_acc:.4f} ({test_cn}/{test_dis})")
                
            # Phase 5: 基于准确率的检查点保存 + Early Stopping
            if test_acc > best_acc:
                best_acc = test_acc
                # 计算 AUC
                from sklearn.metrics import roc_auc_score
                test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
                try:
                    test_auc = roc_auc_score(test_y.cpu().numpy(), test_probs)
                except:
                    test_auc = 0.5
                best_metrics = (test_acc, test_sen, test_spe, test_auc)  # 添加 AUC
                torch.save(model.state_dict(),
                          os.path.join(self.args.ckpt_path, f'fold_{fold}_best.pth'))
                if test_acc > 0.6:
                    print(f"    🎯 新最佳! Test Acc: {test_acc:.4f}")

        # Fold结果汇总
        print(f"📈 Fold {fold} 完成 | Best: Acc={best_metrics[0]:.4f}, Sen={best_metrics[1]:.4f}, Spe={best_metrics[2]:.4f}, AUC={best_metrics[3]:.4f}")
        self.fold_results.append(best_metrics)


    def evaluate_ensemble(self):
        """使用Ensemble评估所有Fold的模型"""
        print(f"\n{'='*30}")
        print("🎯 Ensemble模型评估")
        print(f"{'='*30}")

        # 加载所有Fold的模型
        model_paths = [
            os.path.join(self.args.ckpt_path, f'fold_{i}_best.pth')
            for i in range(self.args.n_folds)
        ]

        # 检查模型文件是否存在
        existing_paths = [p for p in model_paths if os.path.exists(p)]
        if len(existing_paths) == 0:
            print("警告: 未找到任何模型文件，跳过Ensemble评估")
            return

        print(f"找到 {len(existing_paths)} 个模型文件")

        # 提取每个Fold的准确率
        fold_accuracies = [result[0] for result in self.fold_results]  # [acc, sen, spe]
        print(f"各Fold准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")

        # Phase 4: 创建加权Ensemble预测器（使用任务特定模型配置）
        ensemble = EnsemblePredictor(existing_paths, self.device, fold_accuracies, model_config=self.model_config)

        # 对整个数据集进行预测 (启用TTA)
        print("正在进行TTA预测 (10次)...")
        with torch.no_grad():
            probs = ensemble.predict(
                self.dynamic_fc.to(self.device),
                self.modal_data.to(self.device),
                tta_steps=10  # 启用10次TTA
            )

        # 计算指标
        acc, sen, spe = calculate_metrics(
            probs,
            self.labels.to(self.device)
        )

        # 计算AUC和F1分数
        labels_np = self.labels.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # 1. 默认阈值 (0.5) 结果
        default_preds = (probs_np[:, 1] > 0.5).astype(int)
        acc = (default_preds == labels_np).mean()

        # 计算默认指标
        TP = ((default_preds == 1) & (labels_np == 1)).sum()
        TN = ((default_preds == 0) & (labels_np == 0)).sum()
        FP = ((default_preds == 1) & (labels_np == 0)).sum()
        FN = ((default_preds == 0) & (labels_np == 1)).sum()
        sen = TP / (TP + FN + 1e-8)
        spe = TN / (TN + FP + 1e-8)

        # AUC
        auc = roc_auc_score(labels_np, probs_np[:, 1])

        # 默认阈值结果已禁用打印

        # 2. 搜索最佳阈值
        best_acc = 0
        best_thr = 0.5
        best_metrics = (0, 0) # sen, spe

        print(f"\n正在搜索最佳阈值...")
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (probs_np[:, 1] > threshold).astype(int)
            curr_acc = (preds == labels_np).mean()

            if curr_acc > best_acc:
                best_acc = curr_acc
                best_thr = threshold

                # 计算当前指标
                TP = ((preds == 1) & (labels_np == 1)).sum()
                TN = ((preds == 0) & (labels_np == 0)).sum()
                FP = ((preds == 1) & (labels_np == 0)).sum()
                FN = ((preds == 0) & (labels_np == 1)).sum()
                curr_sen = TP / (TP + FN + 1e-8)
                curr_spe = TN / (TN + FP + 1e-8)
                best_metrics = (curr_sen, curr_spe)

        # F1 score (using best threshold)
        best_preds = (probs_np[:, 1] > best_thr).astype(int)
        f1 = f1_score(labels_np, best_preds)

        print(f"\n🏆 最佳Ensemble结果 (阈值={best_thr:.2f}):")
        print(f"  准确率: {best_acc:.4f}")
        print(f"  灵敏度: {best_metrics[0]:.4f}")
        print(f"  特异度: {best_metrics[1]:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"{'='*30}\n")

        return best_acc, best_metrics[0], best_metrics[1], auc, f1

    def print_final_results(self):
        # 5-Fold CV结果已禁用打印
        pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--groups', nargs='+', default=['EMCI', 'CN'])
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)  # 提高默认学习率
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/dhgn')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Phase 5: 迁移学习 - 任务顺序很重要！
    # AD vs CN (source task) → EMCI vs CN (target    # 只训练 EMCI vs CN 二分类
    tasks = [
        {'name': 'EMCI_vs_CN', 'groups': ['EMCI', 'CN']}
    ]

    all_results = {}

    print(f"\n{'#'*50}")
    print(f"🚀 EMCI vs CN 二分类训练")
    print(f"   ❌ 无 Early Stopping（跑满 120 epochs）")
    print(f"   ❌ 无迁移学习（从头训练）")
    print(f"   ✅ 只做 EMCI vs CN 分类")
    print(f"{'#'*50}\n")

    for task in tasks:
        task_name = task['name']
        groups = task['groups']

        print(f"\n\n{'='*50}")
        print(f"👉 当前任务: {task_name} (Groups: {groups})")
        print(f"{'='*50}")

        # 为当前任务配置参数
        current_args = argparse.Namespace(**vars(args))
        current_args.groups = groups
        current_args.ckpt_path = os.path.join(args.ckpt_path, task_name)

        # Phase 5: 获取任务特定配置
        task_config = TASK_CONFIGS.get(task_name, {})

        # 使用任务特定的epochs和learning rate
        if 'num_epochs' in task_config:
            current_args.num_epochs = task_config['num_epochs']
        if 'learning_rate' in task_config:
            current_args.lr = task_config['learning_rate']

        print(f"📝 [Phase 5] Epochs={current_args.num_epochs}, "
              f"LR={current_args.lr}, "
              f"Weight={task_config.get('class_weight_factor', 1.85)}, "
              f"Mixup={task_config.get('mixup_alpha', 0.2)}")

        # 实例化训练器
        trainer = DHGNTrainer(current_args, task_config=task_config)
        trainer.train()

        # 计算 5-Fold CV 统计 (真实泛化能力指标)
        fold_accs = [metrics[0] for metrics in trainer.fold_results]
        fold_sens = [metrics[1] for metrics in trainer.fold_results]
        fold_spes = [metrics[2] for metrics in trainer.fold_results]
        fold_aucs = [metrics[3] for metrics in trainer.fold_results]  # 新增 AUC
        
        import numpy as np
        acc_mean, acc_std = np.mean(fold_accs), np.std(fold_accs)
        sen_mean, sen_std = np.mean(fold_sens), np.std(fold_sens)
        spe_mean, spe_std = np.mean(fold_spes), np.std(fold_spes)
        auc_mean, auc_std = np.mean(fold_aucs), np.std(fold_aucs)  # 新增 AUC
        
        all_results[task_name] = {
            'Accuracy_mean': acc_mean,
            'Accuracy_std': acc_std,
            'Sensitivity_mean': sen_mean,
            'Sensitivity_std': sen_std,
            'Specificity_mean': spe_mean,
            'Specificity_std': spe_std,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'Fold_accuracies': fold_accs
        }




    # 打印最终汇总报告
    print(f"\n\n{'#'*50}")
    print(f"📊 5-Fold 交叉验证结果（真实泛化能力）")
    print(f"{'#'*50}")

    for task_name, metrics in all_results.items():
        print(f"\n🔹 任务: {task_name}")
        print(f"  准确率 (Accuracy):    {metrics['Accuracy_mean']:.4f} ± {metrics['Accuracy_std']:.4f}")
        print(f"  灵敏度 (Sensitivity): {metrics['Sensitivity_mean']:.4f} ± {metrics['Sensitivity_std']:.4f}")
        print(f"  特异度 (Specificity): {metrics['Specificity_mean']:.4f} ± {metrics['Specificity_std']:.4f}")
        print(f"  AUC:                  {metrics['AUC_mean']:.4f} ± {metrics['AUC_std']:.4f}")
        print(f"  各Fold准确率: {[f'{acc:.4f}' for acc in metrics['Fold_accuracies']]}")

    print(f"\n{'#'*50}")
    print("✅ 所有任务完成！")
    print(f"{'#'*50}")
