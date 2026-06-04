"""
时空特征提取器 (Spatio-Temporal Feature Extractor)
D-HGN框架的动态成像通路核心模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Attention for temporal aggregation
    每个头关注不同的时间依赖模式
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadTemporalAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_windows, hidden_dim]
        Returns:
            output: [batch_size, hidden_dim]
            weights: attention weights for visualization (optional)
        """
        batch_size, num_windows, _ = x.shape
        
        # Linear projections and reshape to [batch, heads, time, head_dim]
        Q = self.q_linear(x).view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [batch, heads, time, time]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [batch, heads, time, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # [batch, time, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_windows, self.hidden_dim)
        
        # Global pooling: average over time
        output = attn_output.mean(dim=1)  # [batch, hidden_dim]
        
        # Final projection
        output = self.out_linear(output)
        
        # Return weights for compatibility (average across heads)
        avg_weights = attn_weights.mean(dim=1).mean(dim=-1, keepdim=True)  # [batch, time, 1]
        
        return output, avg_weights

class SpatioTemporalExtractor(nn.Module):
    """
    基于注意力的时空特征提取器
    策略：Spatial CNN (提取单帧特征) -> Temporal Attention (加权聚合)
    """
    
    def __init__(self, 
                 num_rois=90,
                 roi_feature_dim=90,
                 spatial_hidden_dim=64,
                 temporal_hidden_dim=128,
                 output_dim=256,
                 gat_heads=4,

                 gru_layers=2,
                 dropout=0.1,
                 ablation=None):  # Add ablation parameter
        super(SpatioTemporalExtractor, self).__init__()
        
        self.num_rois = num_rois
        self.output_dim = output_dim
        self.ablation = ablation
        
        # 1. 空间特征提取 (处理单个FC矩阵)
        self.use_spatial = True
        if ablation == 'w/o Spatial':
             self.use_spatial = False
             # 原始特征维度: dim=2 (N*N flattened later)
             # But here we treat N as channels first?
             # Logic below: x = [B*T, N, N]
             # If skip spatial: flatten to [B*T, N*N]
             self.temporal_dim = num_rois * num_rois
        else:
             self.spatial_extractor = nn.Sequential(
                nn.Conv1d(num_rois, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
             # Hidden = 128 * num_rois (Flatten后)
             self.temporal_dim = 128 * num_rois
        
        # 2. 时间聚合 (Attention)
        self.use_attention = True
        if ablation == 'w/o Attention':
             self.use_attention = False
             # If no attention, simple mean pooling, no parameters needed here
             # But we might need a projection to match output dim if not using Attention class
             pass
        else:
            self.temporal_attention = MultiHeadTemporalAttention(
                hidden_dim=self.temporal_dim,
                num_heads=4,
                dropout=dropout
            )
        
        # 3. 最终投影
        self.projection = nn.Sequential(
            nn.Linear(self.temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, dynamic_fc, return_attention=False):
        """
        Args:
            dynamic_fc: [Batch, Time, N, N]
            return_attention: Whether to return attention weights
            
        Returns:
            feature: [Batch, Output]
            weights: [Batch, Time, 1] (if return_attention=True)
        """
        # 处理输入维度
        if dynamic_fc.dim() == 3:
            dynamic_fc = dynamic_fc.unsqueeze(0)
            
        batch_size, num_windows, num_nodes, _ = dynamic_fc.shape
        
        # 1. 重塑以进行批量空间提取
        # [B, T, N, N] -> [B*T, N, N]
        x = dynamic_fc.view(-1, num_nodes, num_nodes)
        
        # 2. 空间特征提取
        if self.use_spatial:
            # CNN期望输入: [Batch, Channels, Length] -> 这里把N看作Channel
            spatial_feat = self.spatial_extractor(x)  # [B*T, 128, N]
            # 展平空间特征
            spatial_feat = spatial_feat.flatten(start_dim=1) # [B*T, 128*N]
        else:
            # Ablation: w/o Spatial
            # Flatten directly: [B*T, N*N]
            spatial_feat = x.flatten(start_dim=1)
        
        # 3. 重塑回时序结构
        # [B*T, Hidden] -> [B, T, Hidden]
        spatial_feat = spatial_feat.view(batch_size, num_windows, -1)
        
        weights = None
        # 4. 时间注意力聚合
        if self.use_attention:
             temporal_feat, weights = self.temporal_attention(spatial_feat) # [B, Hidden]
        else:
             # Ablation: w/o Attention
             # Mean Pooling
             temporal_feat = spatial_feat.mean(dim=1) # [B, Hidden]
             weights = None 
        
        # 5. 投影
        output = self.projection(temporal_feat) # [B, Output]
        
        if return_attention:
            return output, weights
            
        return output

if __name__ == "__main__":
    # 测试
    print("测试Attention版特征提取器...")
    model = SpatioTemporalExtractor()
    x = torch.randn(2, 71, 90, 90) # Batch=2
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
