"""
D-HGN (Dynamic Hierarchical Graph Network) 完整模型
整合动态成像通路、SC结构连接通路和群体图分类器
支持: fMRI (DFC) + DTI (SC) 多模态融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_imaging_pathway import SpatioTemporalExtractor
from structural_connectivity_pathway import SCProcessor, NonImagingProcessor
from population_graph_classifier import SimplifiedPopulationClassifier


class SimpleMLP(nn.Module):
    """Simple MLP classifier for w/o Graph ablation"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x, adj=None):
        # Ignore adjacency matrix
        return self.fc(x)


class DHGN(nn.Module):
    """
    D-HGN完整模型
    
    架构:
    1. 动态成像通路 (Dynamic Imaging Pathway)
       - 时空特征提取器 (Mean Pooling + CNN)
    2. 静态非成像通路 (Static Non-Imaging Pathway)
       - 非成像数据处理器 (Normalization + MLP + Similarity)
    3. 群体图分类器 (Population Graph Classifier)
       - 残差GNN + 多尺度融合
    """
    
    def __init__(self,
                 # 数据参数
                 num_rois=90,
                 num_windows=71,
                 num_classes=2,
                 # 时空提取器参数
                 spatial_hidden_dim=64,
                 temporal_hidden_dim=256,
                 st_output_dim=384,
                 # 模态处理参数
                 use_sc=True,  # 是否使用SC矩阵模式
                 sc_hidden_dim=256,
                 sc_output_dim=64,
                 non_imaging_dim=2,  # 备用: 年龄 + 性别
                 non_imaging_hidden=[64, 128, 64],
                 # 群体分类器参数
                 population_hidden_dim=128,

                 num_gnn_layers=4,
                 # 通用参数
                 dropout=0.3,
                 ablation=None): # Add ablation parameter
        """
        Args:
            num_rois: ROI数量 (AAL90=90)
            num_windows: 动态图窗口数 (71)
            num_classes: 分类类别数 (2: EMCI vs CN)
            spatial_hidden_dim: 空间GNN隐藏维度
            temporal_hidden_dim: 时间GRU隐藏维度
            st_output_dim: 时空提取器输出维度
            non_imaging_dim: 非成像数据维度
            non_imaging_hidden: 非成像MLP隐藏层维度列表
            population_hidden_dim: 群体分类器隐藏维度
            num_gnn_layers: 群体分类器GNN层数
            dropout: dropout比率
            ablation: 消融实验变体名称
        """
        super(DHGN, self).__init__()
        
        self.ablation = ablation
        
        # 为 DTI-only 类消融添加 SC 特征投影层 (小型 MLP)
        if ablation in ['w/o fMRI', 'DTI-only']:
            self.sc_feature_proj = nn.Sequential(
                nn.Linear(num_rois * num_rois, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, st_output_dim),
                nn.ReLU()
            )
        
        # 1. 动态成像通路：时空特征提取器
        self.dynamic_imaging_pathway = SpatioTemporalExtractor(
            num_rois=num_rois,
            spatial_hidden_dim=spatial_hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            output_dim=st_output_dim,
            dropout=0.1,
            ablation=ablation
        )
        
        # 2. 模态处理通路：SC矩阵 或 非成像数据
        # Ablation: w/o SC force use_sc=False
        if ablation == 'w/o SC':
            use_sc = False
            
        self.use_sc = use_sc
        if use_sc:
            self.modal_processor = SCProcessor(
                num_rois=num_rois,
                hidden_dim=sc_hidden_dim,
                output_dim=sc_output_dim,
                dropout=0.2
            )
        else:
            self.modal_processor = NonImagingProcessor(
                input_dim=non_imaging_dim,
                hidden_dims=non_imaging_hidden,
                dropout=0.2
            )
        
        
        # 3. 群体图分类器 (较高dropout=0.3，防止过拟合)
        if ablation == 'w/o Graph':
            self.population_graph_classifier = SimpleMLP(
                input_dim=st_output_dim,
                hidden_dim=population_hidden_dim,
                num_classes=num_classes,
                dropout=0.3
            )
        else:
            self.population_graph_classifier = SimplifiedPopulationClassifier(
                input_dim=st_output_dim,
                hidden_dim=population_hidden_dim,
                num_layers=num_gnn_layers,
                num_classes=num_classes,
                dropout=0.3  # 较高dropout防止过拟合
            )
        
    def forward(self, dynamic_fc_batch, modal_batch, return_attention=False):
        """
        前向传播
        
        Args:
            dynamic_fc_batch: [batch_size, num_windows, num_rois, num_rois]
                             动态功能连接序列
            modal_batch: SC模式: [batch_size, num_rois, num_rois] SC矩阵
                        非SC模式: [batch_size, 2] 年龄+性别
            return_attention: 是否返回注意力权重
            
        Returns:
            logits: [batch_size, num_classes] 分类logits
            weights: [batch_size, num_windows, 1] 注意力权重 (if return_attention=True)
        """
        batch_size = dynamic_fc_batch.size(0)
        attention_weights = None
        
        # 根据消融类型选择不同路径
        if self.ablation == 'fMRI-only':
            # fMRI-only: 只用动态FC，不用SC构建群体图
            if return_attention:
                node_features, attention_weights = self.dynamic_imaging_pathway(dynamic_fc_batch, return_attention=True)
            else:
                node_features = self.dynamic_imaging_pathway(dynamic_fc_batch)
            # 使用简单邻接矩阵（全连接）
            adj_matrix = torch.ones(batch_size, batch_size, device=dynamic_fc_batch.device) / batch_size
            
        elif self.ablation == 'DTI-only':
            # DTI-only: 只用SC，不用动态FC
            sc_flat = modal_batch.view(batch_size, -1)
            node_features = self.sc_feature_proj(sc_flat)
            # 用SC构建邻接矩阵
            _, adj_matrix = self.modal_processor(modal_batch)
            
        elif self.ablation == 'w/o fMRI':
            # 同 DTI-only
            sc_flat = modal_batch.view(batch_size, -1)
            node_features = self.sc_feature_proj(sc_flat)
            _, adj_matrix = self.modal_processor(modal_batch)
            
        else:
            # 正常多模态路径
            if return_attention:
                node_features, attention_weights = self.dynamic_imaging_pathway(dynamic_fc_batch, return_attention=True)
            else:
                node_features = self.dynamic_imaging_pathway(dynamic_fc_batch)
            _, adj_matrix = self.modal_processor(modal_batch)
        
        # 群体图分类器
        logits = self.population_graph_classifier(node_features, adj_matrix)
        
        if return_attention and attention_weights is not None:
            return logits, attention_weights
            
        return logits
    
    def get_embeddings(self, dynamic_fc_batch, modal_batch):
        """
        获取节点嵌入（用于可视化或分析）
        
        Returns:
            node_features: [batch_size, st_output_dim] 时空特征
            adj_matrix: [batch_size, batch_size] 邻接矩阵
        """
        batch_size = dynamic_fc_batch.size(0)
        
        # 提取时空特征
        node_features = []
        for i in range(batch_size):
            dynamic_fc = dynamic_fc_batch[i]
            node_feat = self.dynamic_imaging_pathway(dynamic_fc)
            node_features.append(node_feat)
        
        node_features = torch.stack(node_features, dim=0)
        
        # 构建邻接矩阵
        _, adj_matrix = self.modal_processor(modal_batch)
        
        return node_features, adj_matrix


class DHGNWithAuxiliaryLoss(DHGN):
    """
    带辅助损失的D-HGN模型
    可以添加额外的正则化或对比学习损失
    """
    
    def __init__(self, *args, **kwargs):
        super(DHGNWithAuxiliaryLoss, self).__init__(*args, **kwargs)
        
        # 对比学习投影头（可选）
        self.projection_head = nn.Sequential(
            nn.Linear(kwargs.get('st_output_dim', 256), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, dynamic_fc_batch, modal_batch, return_aux=False):
        """
        Args:
            return_aux: 是否返回辅助信息用于计算额外损失
        """
        batch_size = dynamic_fc_batch.size(0)
        
        # 提取时空特征
        node_features = []
        for i in range(batch_size):
            dynamic_fc = dynamic_fc_batch[i]
            node_feat = self.dynamic_imaging_pathway(dynamic_fc)
            node_features.append(node_feat)
        
        node_features = torch.stack(node_features, dim=0)
        
        # 构建邻接矩阵
        modal_features, adj_matrix = self.modal_processor(modal_batch)
        
        # 分类
        logits = self.population_graph_classifier(node_features, adj_matrix)
        
        if return_aux:
            # 返回辅助信息
            projected_features = self.projection_head(node_features)
            return logits, {
                'node_features': node_features,
                'modal_features': modal_features,
                'adj_matrix': adj_matrix,
                'projected_features': projected_features
            }
        else:
            return logits
    
    def compute_contrastive_loss(self, features, labels, temperature=0.5):
        """
        计算对比学习损失（同类样本拉近，异类样本推远）
        
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]
            temperature: 温度参数
            
        Returns:
            loss: 对比损失
        """
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(features, features.t()) / temperature
        
        # 构建正样本mask（同类别）
        labels = labels.unsqueeze(1)
        mask_positive = (labels == labels.t()).float()
        mask_positive.fill_diagonal_(0)  # 排除自己
        
        # 构建负样本mask（不同类别）
        mask_negative = 1 - mask_positive
        mask_negative.fill_diagonal_(0)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 对于每个样本，计算其与所有正样本的相似度 vs 所有负样本的相似度
        positive_sim = (exp_sim * mask_positive).sum(dim=1)
        negative_sim = (exp_sim * mask_negative).sum(dim=1)
        
        # 避免除零
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        loss = loss.mean()
        
        return loss


def create_dhgn_model(config=None):
    """
    工厂函数：创建D-HGN模型
    
    Args:
        config: 配置字典，如果为None则使用默认配置
        
    Returns:
        model: D-HGN模型实例
    """
    if config is None:
        config = {
            'num_rois': 90,
            'num_windows': 71,
            'num_classes': 2,
            'spatial_hidden_dim': 64,
            'temporal_hidden_dim': 256,  # 增加容量 128 -> 256
            'st_output_dim': 384,        # 增加容量 256 -> 384
            'non_imaging_dim': 2,
            'non_imaging_hidden': [64, 128, 64],
            'population_hidden_dim': 128,
            'num_gnn_layers': 4,
            'dropout': 0.3
        }
    
    model = DHGN(**config)
    return model


if __name__ == "__main__":
    # 测试
    print("测试D-HGN模型...")
    
    # 创建模型
    model = create_dhgn_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟数据
    batch_size = 8
    dynamic_fc = torch.randn(batch_size, 71, 90, 90)  # [B, T, N, N]
    non_imaging = torch.randn(batch_size, 2)  # [B, 2]
    
    # 前向传播
    logits = model(dynamic_fc, non_imaging)
    print(f"输入动态FC形状: {dynamic_fc.shape}")
    print(f"输入非成像数据形状: {non_imaging.shape}")
    print(f"输出logits形状: {logits.shape}")  # [8, 2]
    print(f"预测类别: {torch.argmax(logits, dim=1)}")
    
    # 测试带辅助损失的模型
    print("\n测试带辅助损失的D-HGN模型...")
    model_aux = DHGNWithAuxiliaryLoss()
    logits, aux_info = model_aux(dynamic_fc, non_imaging, return_aux=True)
    print(f"辅助信息keys: {aux_info.keys()}")
    
    # 计算对比损失
    labels = torch.randint(0, 2, (batch_size,))
    contrastive_loss = model_aux.compute_contrastive_loss(
        aux_info['projected_features'], labels
    )
    print(f"对比损失: {contrastive_loss.item():.4f}")
