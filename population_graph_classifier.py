"""
群体图分类器 (Population Graph Classifier)
D-HGN框架的核心决策组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv


class ResidualGNNLayer(nn.Module):
    """
    残差图神经网络层
    防止过平滑问题
    """
    
    def __init__(self, in_channels, out_channels, K=3, dropout=0.3):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            K: 切比雪夫多项式阶数
            dropout: dropout比率
        """
        super(ResidualGNNLayer, self).__init__()
        
        self.conv = ChebConv(in_channels, out_channels, K=K, normalization='sym')
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = dropout
        
        # 如果输入输出维度不同，需要投影层
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: 节点特征 [n_nodes, in_channels]
            edge_index: 边索引 [2, n_edges]
            edge_weight: 边权重 [n_edges]
            
        Returns:
            out: 输出特征 [n_nodes, out_channels]
        """
        # 图卷积
        out = self.conv(x, edge_index, edge_weight)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        # 残差连接
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        
        out = out + 0.7 * residual  # 加权残差
        
        return out


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块
    通过可学习的权重融合不同层的特征
    """
    
    def __init__(self, num_layers):
        """
        Args:
            num_layers: GNN层数
        """
        super(MultiScaleFeatureFusion, self).__init__()
        
        # 可学习的聚合权重
        self.weights = nn.Parameter(torch.randn(num_layers))
        
    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: 列表，包含每一层的输出 [n_nodes, hidden_dim]
            
        Returns:
            fused_features: 融合后的特征 [n_nodes, hidden_dim]
        """
        # Softmax归一化权重
        weights = F.softmax(self.weights, dim=0)
        
        # 加权求和
        fused_features = sum(w * feat for w, feat in zip(weights, layer_outputs))
        
        return fused_features


class PopulationGraphClassifier(nn.Module):
    """
    群体图分类器
    
    架构:
    1. 残差GNN层 (4层)
    2. 多尺度特征融合
    3. 分类器
    """
    
    def __init__(self,
                 input_dim=256,  # 来自时空特征提取器的输出
                 hidden_dim=128,
                 num_layers=4,
                 num_classes=2,
                 K=3,  # 切比雪夫多项式阶数
                 dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_classes: 分类类别数
            K: 切比雪夫多项式阶数
            dropout: dropout比率
        """
        super(PopulationGraphClassifier, self).__init__()
        
        self.num_layers = num_layers
        
        # 残差GNN层
        self.gnn_layers = nn.ModuleList()
        
        # 第一层
        self.gnn_layers.append(
            ResidualGNNLayer(input_dim, hidden_dim, K=K, dropout=dropout)
        )
        
        # 中间层
        for _ in range(num_layers - 1):
            self.gnn_layers.append(
                ResidualGNNLayer(hidden_dim, hidden_dim, K=K, dropout=dropout)
            )
        
        # 多尺度特征融合
        self.fusion = MultiScaleFeatureFusion(num_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, node_features, edge_index, edge_weight=None):
        """
        Args:
            node_features: [n_subjects, input_dim] 节点特征
            edge_index: [2, n_edges] 边索引
            edge_weight: [n_edges] 边权重
            
        Returns:
            logits: [n_subjects, num_classes] 分类logits
        """
        layer_outputs = []
        x = node_features
        
        # 逐层前向传播
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, edge_weight)
            layer_outputs.append(x)
        
        # 多尺度特征融合
        fused_features = self.fusion(layer_outputs)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_embeddings(self, node_features, edge_index, edge_weight=None):
        """
        获取节点嵌入（用于可视化或其他分析）
        
        Returns:
            embeddings: [n_subjects, hidden_dim]
        """
        layer_outputs = []
        x = node_features
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_weight)
            layer_outputs.append(x)
        
        embeddings = self.fusion(layer_outputs)
        
        return embeddings


class SimplifiedPopulationClassifier(nn.Module):
    """
    简化版群体图分类器
    直接使用邻接矩阵而不需要edge_index格式
    """
    
    def __init__(self,
                 input_dim=256,
                 hidden_dim=128,
                 num_layers=4,
                 num_classes=2,
                 dropout=0.3):
        super(SimplifiedPopulationClassifier, self).__init__()
        
        self.num_layers = num_layers
        
        # GNN层（使用简单的图卷积）
        self.gnn_layers = nn.ModuleList()
        
        self.gnn_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # 多尺度融合权重
        self.fusion_weights = nn.Parameter(torch.randn(num_layers))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: [n_subjects, input_dim]
            adj_matrix: [n_subjects, n_subjects] 邻接矩阵
            
        Returns:
            logits: [n_subjects, num_classes]
        """
        # 归一化邻接矩阵（添加自环）
        adj_norm = self._normalize_adjacency(adj_matrix)
        
        layer_outputs = []
        x = node_features
        
        # 逐层传播
        for i in range(self.num_layers):
            # 图卷积: A * X * W
            x = torch.mm(adj_norm, x)  # 聚合邻居特征
            x = self.gnn_layers[i](x)  # 线性变换
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            
            # 残差连接
            if i > 0:
                x = x + 0.7 * layer_outputs[-1]
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # 多尺度融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, layer_outputs))
        
        # 分类
        logits = self.classifier(fused)
        
        return logits
    
    def _normalize_adjacency(self, adj):
        """
        对称归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)
        
        Args:
            adj: [n, n] 邻接矩阵
            
        Returns:
            adj_norm: [n, n] 归一化邻接矩阵
        """
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # 计算度矩阵
        degree = adj.sum(dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        
        # D^(-1/2) * A * D^(-1/2)
        adj_norm = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        
        return adj_norm


if __name__ == "__main__":
    # 测试
    print("测试群体图分类器...")
    
    # 模拟数据
    n_subjects = 10
    input_dim = 256
    
    node_features = torch.randn(n_subjects, input_dim)
    adj_matrix = torch.rand(n_subjects, n_subjects)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2  # 对称化
    
    # 简化版分类器
    classifier = SimplifiedPopulationClassifier(input_dim=input_dim)
    logits = classifier(node_features, adj_matrix)
    
    print(f"输入特征形状: {node_features.shape}")
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"输出logits形状: {logits.shape}")  # [10, 2]
    print(f"预测类别: {torch.argmax(logits, dim=1)}")
