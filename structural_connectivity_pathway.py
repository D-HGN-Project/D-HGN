"""
多模态通路 (Multimodal Pathway)
支持: SC矩阵 (DTI结构连接) 或 非成像数据 (年龄+性别)
用于构建群体图的邻接矩阵
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCProcessor(nn.Module):
    """SC矩阵处理器 - 基于DTI结构连接矩阵计算被试相似度"""
    
    def __init__(self, num_rois=90, hidden_dim=256, output_dim=64, dropout=0.2):
        """
        Args:
            num_rois: ROI数量 (AAL90=90)
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            dropout: dropout比率
        """
        super(SCProcessor, self).__init__()
        
        self.num_rois = num_rois
        self.input_dim = num_rois * num_rois  # 90*90 = 8100
        self.output_dim = output_dim
        
        # SC矩阵展平后通过MLP降维
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, sc_batch):
        """
        Args:
            sc_batch: [n_subjects, 90, 90] SC矩阵
        Returns:
            features: [n_subjects, output_dim] 编码后的特征
            adj_matrix: [n_subjects, n_subjects] 基于SC相似度的邻接矩阵
        """
        batch_size = sc_batch.size(0)
        
        # 1. 展平SC矩阵
        sc_flat = sc_batch.view(batch_size, -1)  # [n, 8100]
        
        # 2. MLP编码
        features = self.encoder(sc_flat)  # [n, output_dim]
        
        # 3. 计算被试间相似度作为群体图的边
        adj_matrix = self.compute_topk_similarity(features, k=10)
        
        return features, adj_matrix
    
    def compute_topk_similarity(self, features, k=10):
        """Top-K稀疏化：每个节点只连接最相似的K个节点"""
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 余弦相似度
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 移除对角线（自己）
        similarity_matrix.fill_diagonal_(-1e9)
        
        # Top-K选择
        batch_size = similarity_matrix.size(0)
        k = min(k, batch_size - 1)
        
        topk_values, topk_indices = torch.topk(similarity_matrix, k, dim=1)
        
        # 构建稀疏邻接矩阵
        adj_matrix = torch.zeros_like(similarity_matrix)
        for i in range(batch_size):
            adj_matrix[i, topk_indices[i]] = topk_values[i]
        
        # 对称化（无向图）
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2.0
        
        # 归一化到[0,1]
        adj_matrix = torch.clamp(adj_matrix, min=0)
        if adj_matrix.max() > 0:
            adj_matrix = adj_matrix / adj_matrix.max()
        
        return adj_matrix


class NonImagingProcessor(nn.Module):
    """非成像数据处理器 (备用) - 基于年龄+性别"""

    def __init__(self,
                 input_dim=2,  # 年龄 + 性别
                 hidden_dims=[64, 128, 64],
                 dropout=0.2):
        super(NonImagingProcessor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, non_imaging_data):
        """
        Returns:
            features: [n_subjects, output_dim]
            adj_matrix: [n_subjects, n_subjects] 稀疏邻接矩阵
        """
        features = self.mlp(non_imaging_data)
        adj_matrix = self.compute_topk_similarity(features, k=10)
        return features, adj_matrix
    
    def compute_topk_similarity(self, features, k=10):
        """Top-K稀疏化"""
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        similarity_matrix.fill_diagonal_(-1e9)
        
        batch_size = similarity_matrix.size(0)
        k = min(k, batch_size - 1)
        
        topk_values, topk_indices = torch.topk(similarity_matrix, k, dim=1)
        
        adj_matrix = torch.zeros_like(similarity_matrix)
        for i in range(batch_size):
            adj_matrix[i, topk_indices[i]] = topk_values[i]
        
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2.0
        adj_matrix = torch.clamp(adj_matrix, min=0)
        if adj_matrix.max() > 0:
            adj_matrix = adj_matrix / adj_matrix.max()
        
        return adj_matrix


# 统一接口
def create_modal_processor(use_sc=True, **kwargs):
    """
    创建模态处理器
    Args:
        use_sc: 是否使用SC矩阵模式
    Returns:
        processor: SCProcessor 或 NonImagingProcessor
    """
    if use_sc:
        return SCProcessor(**kwargs)
    else:
        return NonImagingProcessor(**kwargs)