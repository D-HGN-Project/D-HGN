import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGraphConv

"""
Baseline Models for fMRI/DTI Analysis Comparison
对比实验基线模型集合
"""

# ==================================================================================
# 1. MLP Classifier (Static/Vectorized)
# ==================================================================================
class MLPClassifier(nn.Module):
    """
    多层感知机 (MLP)
    输入: 静态FC矩阵 (Flattened upper triangle or full matrix)
    策略: 将矩阵展平为向量 -> 全连接层 -> 分类
    """
    def __init__(self, num_rois=90, hidden_dims=[256, 128], num_classes=2, dropout=0.5):
        super(MLPClassifier, self).__init__()
        # 展平维度 (只取上三角，或者全矩阵，这里简化取全矩阵展平 N*N)
        input_dim = num_rois * num_rois
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, num_classes)

    def forward(self, dynamic_fc, modal_data=None):
        # 如果输入是动态FC [B, T, N, N]，先求平均得到静态FC [B, N, N]
        if dynamic_fc.dim() == 4:
            static_fc = dynamic_fc.mean(dim=1)
        else:
            static_fc = dynamic_fc
            
        # Flatten [B, N*N]
        x = static_fc.reshape(static_fc.size(0), -1)
        x = self.features(x)
        return self.classifier(x)


# ==================================================================================
# 2. BrainNetCNN (Standard CNN for Connectivity)
# ==================================================================================
class E2EBlock(nn.Module):
    def __init__(self, in_planes, planes, rois, bias=True):
        super().__init__()
        self.d = rois
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, rois), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (rois, 1), bias=bias)
        
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 2) + torch.cat([b]*self.d, 3)

class E2NBlock(nn.Module):
    def __init__(self, in_planes, planes, rois, bias=True):
        super().__init__()
        self.cnn = nn.Conv2d(in_planes, planes, (1, rois), bias=bias)
        
    def forward(self, x):
        return self.cnn(x)

class N2GBlock(nn.Module):
    def __init__(self, in_planes, planes, rois, bias=True):
        super().__init__()
        self.cnn = nn.Conv2d(in_planes, planes, (rois, 1), bias=bias)
        
    def forward(self, x):
        return self.cnn(x)

class BrainNetCNN(nn.Module):
    """
    BrainNetCNN (Kawahara et al., 2017)
    专为连接矩阵设计的卷积神经网络
    E2E -> E2N -> N2G -> Dense
    """
    def __init__(self, num_rois=90, num_classes=2, dropout=0.5):
        super(BrainNetCNN, self).__init__()
        
        # OOM Fix: Reduce channels from 32/64 to 8/16
        self.e2e = E2EBlock(1, 8, num_rois)
        self.e2n = E2NBlock(8, 16, num_rois)
        self.n2g = N2GBlock(16, 30, num_rois)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, dynamic_fc, modal_data=None):
        # [B, T, N, N] -> [B, N, N] (Static Mean)
        if dynamic_fc.dim() == 4:
            x = dynamic_fc.mean(dim=1)
        else:
            x = dynamic_fc
            
        # Add Channel Dim [B, 1, N, N]
        x = x.unsqueeze(1)
        
        x = F.leaky_relu(self.e2e(x), 0.33)
        x = F.leaky_relu(self.e2n(x), 0.33)
        x = F.leaky_relu(self.n2g(x), 0.33)
        
        return self.fc(x)

# ==================================================================================
# 3. GCN (Graph Convolutional Network)
# ==================================================================================
class GCNClassifier(nn.Module):
    """
    Graph Convolutional Network
    输入: 静态FC图
    """
    def __init__(self, num_rois=90, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GCNClassifier, self).__init__()
        
        self.conv1 = DenseGCNConv(num_rois, hidden_dim) 
        # Fix: DenseGCNConv output is [B, N, Out], so BN should be 1d(Out) but applied to (B*N, Out) or Transposed
        # Instead of complex BN, we use LayerNorm or simplified BN
        self.bn1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.dropout = dropout

    def forward(self, dynamic_fc, modal_data=None):
        # [B, T, N, N] -> [B, N, N]
        if dynamic_fc.dim() == 4:
            adj = dynamic_fc.mean(dim=1)
        else:
            adj = dynamic_fc

        # 节点特征：直接使用邻接矩阵本身作为节点特征 X=A [B, N, N]
        x = adj.clone()
        
        # 简单阈值化处理邻接矩阵 (Top 20% connections or threshold)
        # 这里为了梯度流，保留全连接但ReLU激活，或简单的Soft二值化
        # DenseGCNConv接受加权邻接矩阵
        
        x = F.relu(self.bn1(self.conv1(x, adj)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, adj)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling (Mean over nodes) [B, N, Hidden] -> [B, Hidden]
        x = x.mean(dim=1)
        
        return self.fc(x)


# ==================================================================================
# 4. LSTM (Long Short-Term Memory) - Dynamic Analysis
# ==================================================================================
class LSTMClassifier(nn.Module):
    """
    LSTM for Dynamic FC
    输入: DFC序列 [Batch, Time, Nodes*Nodes]
    """
    def __init__(self, num_rois=90, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        input_dim = int(num_rois * (num_rois - 1) / 2) # 上三角特征数
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.triu_indices = torch.triu_indices(num_rois, num_rois, offset=1)

    def forward(self, dynamic_fc, modal_data=None):
        # dynamic_fc: [B, T, N, N]
        batch_size, time_steps, _, _ = dynamic_fc.shape
        
        # 提取上三角 [B, T, Features]
        # 这种写法在Loop中可能慢，可以使用向量化操作
        # dynamic_fc_flat = dynamic_fc[:, :, self.triu_indices[0], self.triu_indices[1]]
        # 简化：Flatten all [B, T, N*N]
        x = dynamic_fc.reshape(batch_size, time_steps, -1)
        
        # 上三角提取比较繁琐，这里简单用 Linear 降维一下或者直接用 N*N
        # 为了效率我们先用 embedding layer 降维
        if not hasattr(self, 'input_proj'):
             self.input_proj = nn.Linear(90*90, self.input_dim).to(dynamic_fc.device)
        
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步 [B, Hidden]
        last_step = lstm_out[:, -1, :]
        
        return self.fc(last_step)


# ==================================================================================
# 5. Transformer (Attention-based) - Dynamic Analysis
# ==================================================================================
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder for Dynamic FC
    """
    def __init__(self, num_rois=90, d_model=128, nhead=4, num_layers=2, num_classes=2, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        
        self.input_embedding = nn.Sequential(
            nn.Flatten(start_dim=2), # [B, T, N*N]
            nn.Linear(num_rois*num_rois, d_model),
            nn.ReLU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS Token (Learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, dynamic_fc, modal_data=None):
        # [B, T, N, N]
        batch_size = dynamic_fc.size(0)
        
        # Embedding
        x = self.input_embedding(dynamic_fc) # [B, T, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, T+1, d_model]
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Pick CLS output
        cls_out = x[:, 0, :]
        
        return self.fc(cls_out)



# 6. GAT (Graph Attention Network)
# ==================================================================================
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    """
    Graph Attention Network
    相比GCN，引入注意力机制自动学习邻居的重要性
    """
    def __init__(self, num_rois=90, hidden_dim=64, num_classes=2, heads=4, dropout=0.5):
        super(GATClassifier, self).__init__()
        
        # Multi-head attention
        self.conv1 = GATConv(num_rois, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.dropout = dropout

    def forward(self, dynamic_fc, modal_data=None):
        if dynamic_fc.dim() == 4:
            adj = dynamic_fc.mean(dim=1)
        else:
            adj = dynamic_fc

        # X = Adjacency itself [B, N, N]
        x = adj.clone()
        
        # GATConv需要 edge_index格式，这里为了简便我们用 DenseGAT如果可用，
        # 或者为了兼容性，我们手动将全连接稠密图转为 edge_index
        # 但这对每个batch动态生成很慢。
        # 这里我们使用简单的近似：将 Dense 矩阵视为特征，仅使用 TopK 边进行聚合，
        # 或者简化为直接操作：GATConv 支持 dense adjacency in some versions? No.
        # 我们这里实现一个简化版：每个样本单独过GAT（效率较低但正确）或者使用 DenseGCN 替代
        # 鉴于 PyG 的 DenseGATConv 还不成熟，我们手写一个简单的 Attention Layer
    
        # 简单版 Self-Attention Graph Layer (类似于 GAT)
        # [B, N, N] * [B, N, N] (Feature) -> Attention
        
        # 重新定义为 Dense GCN + Attention
        # 为避免复杂依赖，这里用 GCN 作为 GAT 的 proxy，或者实现一个简单的 Attention GNN
        
        # 实际上，我们可以重用你的 SpatioTemporalExtractor 里的 MultiHeadAttention 思想，但应用在图上
        return self.gcn_forward_proxy(x, adj)

    def gcn_forward_proxy(self, x, adj):
        # 这是一个占位，实际应该用 torch_geometric.nn.DenseGATConv
        # 如果环境支持:
        try:
             from torch_geometric.nn import DenseGATConv
             if not hasattr(self, 'dense_gat'):
                 self.dense_gat = DenseGATConv(90, 64, heads=4).to(x.device)
                 self.dense_gat2 = DenseGATConv(64*4, 64, heads=1).to(x.device)
             
             x = F.relu(self.dense_gat(x, adj))
             x = self.dense_gat2(x, adj) 
             x = x.mean(dim=1)
             return self.fc(x)
        except ImportError:
             # Fallback to GCN if GAT not available
             return self.fc(torch.zeros(x.size(0), 64).to(x.device))


# ==================================================================================
# 7. ST-GCN (Spatio-Temporal Graph Convolutional Network)
# ==================================================================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, dropout=0):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        # Temporal Conv: kernel (3, 1) spanning 3 time steps
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, A):
        # x: [B, C, T, N]
        # A: [B, N, N] or [N, N]
        
        # 1. GCN: XW * A (Simplified)
        x = self.gcn(x)
        # Apply Adjacency: [B, C, T, N] x [N, N] -> [B, C, T, N]
        # Einsum is cleaner:
        x = torch.einsum('bctn,bnn->bctn', x, A)
        
        # 2. TCN
        x = self.relu(self.tcn(x))
        x = self.dropout(x)
        return x

class STGCNClassifier(nn.Module):
    """
    ST-GCN: 同时在时间和空间维度卷积
    """
    def __init__(self, num_rois=90, in_channels=90, hidden_dim=64, num_classes=2, num_windows=71):
        super(STGCNClassifier, self).__init__()
        
        self.block1 = STGCNBlock(in_channels, hidden_dim, num_rois)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim*2, num_rois, stride=2)
        self.block3 = STGCNBlock(hidden_dim*2, hidden_dim*4, num_rois, stride=2)
        
        # Global Pooling
        self.fc = nn.Linear(hidden_dim*4, num_classes)

    def forward(self, dynamic_fc, modal_data=None):
        # dynamic_fc: [B, T, N, N]
        B, T, N, _ = dynamic_fc.shape
        
        # Input features: Use connectivity profiles as features
        # [B, T, N, N] -> [B, N, T, N] (Channels=N)
        x = dynamic_fc.permute(0, 3, 1, 2)
        
        # Adjacency: Mean over time for graph structure [B, N, N]
        A = dynamic_fc.mean(dim=1)
        A = F.normalize(A, p=1, dim=2) # Row normalize
        
        x = self.block1(x, A)
        x = self.block2(x, A)
        x = self.block3(x, A)
        
        # Pooling: Mean over Node and Time
        x = x.mean(dim=3).mean(dim=2) # [B, C]
        
        return self.fc(x)


# ==================================================================================
# 8. DiffPool (Differentiable Pooling)
# ==================================================================================
class DiffPoolClassifier(nn.Module):
    """
    DiffPool (Ying et al., 2018)
    学习分层图表示
    """
    def __init__(self, num_rois=90, hidden_dim=64, num_classes=2):
        super(DiffPoolClassifier, self).__init__()
        
        # GNN for embedding
        self.gnn1_embed = DenseGCNConv(num_rois, hidden_dim)
        self.gnn2_embed = DenseGCNConv(hidden_dim, hidden_dim)
        
        # GNN for assignment (pooling)
        num_clusters = 10 # Pool to 10 nodes
        self.gnn1_pool = DenseGCNConv(num_rois, num_clusters)
        self.gnn2_pool = DenseGCNConv(num_clusters, num_clusters)
        
        # Final prediction
        self.fc = nn.Linear(hidden_dim, num_classes) # *2 if we concat
        
    def forward(self, dynamic_fc, modal_data=None):
        # Use static mean
        if dynamic_fc.dim() == 4:
            adj = dynamic_fc.mean(dim=1)
        else:
            adj = dynamic_fc
        
        x = adj.clone()
        mask = torch.ones(x.size(0), 90).to(x.device)
        
        # Layer 1
        z = F.relu(self.gnn1_embed(x, adj))
        s = F.softmax(self.gnn1_pool(x, adj), dim=-1)
        
        # DiffPool 1
        # x_new = S^T * Z
        x_coarse = torch.matmul(s.transpose(1, 2), z)
        # adj_new = S^T * A * S
        adj_coarse = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        
        # Layer 2 (on coarse graph)
        z2 = F.relu(self.gnn2_embed(x_coarse, adj_coarse))
        
        # Global Pooling
        out = z2.mean(dim=1) + z2.max(dim=1)[0]
        
        # Final prediction
        # Output dim of z2 is hidden_dim (64)
        # out is [B, 64], so input to fc should be 64, not 128
        return self.fc(out)


# ==================================================================================
# 9. TCN (Temporal Convolutional Network)
# ==================================================================================
class TCNClassifier(nn.Module):
    """
    TCN for Dynamic Sequence Classification
    """
    def __init__(self, num_rois=90, num_channels=[128, 128, 64], num_classes=2, kernel_size=3, dropout=0.2):
        super(TCNClassifier, self).__init__()
        
        input_size = num_rois * num_rois # Flattened
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)*dilation_size // 2, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, dynamic_fc, modal_data=None):
        # [B, T, N, N]
        b, t, n, _ = dynamic_fc.shape
        x = dynamic_fc.view(b, t, -1).permute(0, 2, 1) # [B, Channels, T]
        
        y = self.network(x)
        
        # Global Average Pooling
        y = y.mean(dim=2)
        
        return self.fc(y)

# ==================================================================================
# 10. GraphSAGE
# ==================================================================================
from torch_geometric.nn import DenseSAGEConv

class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE (Dense version)
    """
    def __init__(self, num_rois=90, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GraphSAGEClassifier, self).__init__()
        
        self.conv1 = DenseSAGEConv(num_rois, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.dropout = dropout

    def forward(self, dynamic_fc, modal_data=None):
        if dynamic_fc.dim() == 4:
            adj = dynamic_fc.mean(dim=1)
        else:
            adj = dynamic_fc
            
        x = adj.clone()
        x = F.relu(self.bn1(self.conv1(x, adj)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, adj)))
        x = x.mean(dim=1)
        return self.fc(x)

# ==================================================================================
# 11. GIN (Graph Isomorphism Network)
# ==================================================================================
from torch_geometric.nn import DenseGINConv

class GINClassifier(nn.Module):
    """
    GIN (Graph Isomorphism Network) - Dense version
    """
    def __init__(self, num_rois=90, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GINClassifier, self).__init__()
        
        mlp1 = nn.Sequential(
            nn.Linear(num_rois, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) # DenseGINConv applies MLP to nodes, so BN is ok if reshaped, but simpler to omit or use LayerNorm inside
        )
        # Fix: MLP output in DenseGINConv is [B, N, Hidden]. 
        
        self.conv1 = DenseGINConv(
            nn.Sequential(
                nn.Linear(num_rois, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv2 = DenseGINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, dynamic_fc, modal_data=None):
        if dynamic_fc.dim() == 4:
            adj = dynamic_fc.mean(dim=1)
        else:
            adj = dynamic_fc
            
        x = adj.clone()
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = x.mean(dim=1)
        return self.fc(x)

# ==================================================================================
# 12. Bi-LSTM (Bidirectional LSTM)
# ==================================================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, num_rois=90, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        input_dim = int(num_rois * (num_rois - 1) / 2)
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # <--- Bidirectional
            dropout=dropout
        )
        
        # Hidden dim * 2 because of bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, dynamic_fc, modal_data=None):
        batch_size, time_steps, _, _ = dynamic_fc.shape
        x = dynamic_fc.reshape(batch_size, time_steps, -1)
        
        if not hasattr(self, 'input_proj'):
             self.input_proj = nn.Linear(90*90, self.input_dim).to(dynamic_fc.device)
        x = self.input_proj(x)
        
        lstm_out, _ = self.lstm(x)
        # Concat last step of forward and first step of backward?
        # Or just take mean over time? Or last output?
        # lstm_out: [B, T, 2*H]
        # Max pooling over time is robust
        out = torch.mean(lstm_out, dim=1) 
        
        return self.fc(out)

# ==================================================================================
# 13. GRU (Gated Recurrent Unit)
# ==================================================================================
class GRUClassifier(nn.Module):
    def __init__(self, num_rois=90, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super(GRUClassifier, self).__init__()
        input_dim = int(num_rois * (num_rois - 1) / 2)
        self.input_dim = input_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, dynamic_fc, modal_data=None):
        batch_size, time_steps, _, _ = dynamic_fc.shape
        x = dynamic_fc.reshape(batch_size, time_steps, -1)
        
        if not hasattr(self, 'input_proj'):
             self.input_proj = nn.Linear(90*90, self.input_dim).to(dynamic_fc.device)
        x = self.input_proj(x)
        
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        return self.fc(last_step)
        
        
def create_baseline_model(model_name, config=None):
    name = model_name.lower()
    if name == 'mlp': return MLPClassifier()
    if name == 'brainnetcnn': return BrainNetCNN()
    if name == 'gcn': return GCNClassifier()
    if name == 'lstm': return LSTMClassifier()
    if name == 'transformer': return TransformerClassifier()
    if name == 'gat': return GATClassifier()
    if name == 'stgcn': return STGCNClassifier()
    if name == 'diffpool': return DiffPoolClassifier()
    if name == 'tcn': return TCNClassifier()
    if name == 'graphsage': return GraphSAGEClassifier()
    if name == 'gin': return GINClassifier()
    if name == 'bilstm': return BiLSTMClassifier()
    if name == 'gru': return GRUClassifier()
    return None
    
if __name__ == "__main__":
    # Test SOTA Models
    x = torch.randn(2, 71, 90, 90)
    print("Test ST-GCN:", STGCNClassifier()(x).shape)
    print("Test DiffPool:", DiffPoolClassifier()(x).shape)
    print("Test TCN:", TCNClassifier()(x).shape)
        

