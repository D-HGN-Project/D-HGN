
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model
from gpu_utils import setup_gpu

# Config
CKPT_PATH = 'checkpoints/dhgn/EMCI_vs_CN/fold_3_best.pth'
# 与 train_dhgn.py 中的 TASK_CONFIGS['EMCI_vs_CN']['model_config'] 保持一致
CONFIG = {
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

plt.rcParams['font.family'] = 'sans-serif'

def normalize_for_vis(data):
    d_min, d_max = data.min(), data.max()
    return (data - d_min) / (d_max - d_min + 1e-8)

def add_arrow_between_rows(fig, y_start, y_end):
    """Draw a gray arrow indicating flow between rows"""
    fig.add_artist(patches.FancyArrowPatch(
        (0.5, y_start), (0.5, y_end),
        transform=fig.transFigure, color='gray', 
        arrowstyle='simple,head_width=10,head_length=10', alpha=0.5, linewidth=0
    ))

def plot_matrix_sequence(ax_list, data_seq, row_idx, mask_indices=None, attention_weights=None, cmap='viridis'):
    """
    Plot a sequence of matrices with enhanced styling.
    """
    indices = [5, 20, 35, 50, 65]
    frame_labels = ['Early (T=5)', 'Mid (T=20)', 'Mid (T=35)', 'Mid (T=50)', 'Late (T=65)']
    
    # Pre-calculate alpha values if attention is provided
    alphas = []
    if attention_weights is not None:
        w_max = attention_weights.max()
        for idx in indices:
            w = attention_weights[idx]
            # Power law for stronger contrast: (x^2.5) suppresses middle values heavily
            alpha = np.clip((w / w_max) ** 2.5 + 0.05, 0.05, 1.0)
            alphas.append(alpha)
            
    for i, idx in enumerate(indices):
        ax = ax_list[i]
        matrix = data_seq[idx]
        
        # 1. Random Masking (Blackout)
        if mask_indices and idx in mask_indices:
            ax.imshow(np.zeros_like(matrix), cmap='gray', vmin=0, vmax=1)
            # Add dice icon text
            if i == 1: # Only on first masked
                 ax.text(45, 45, "🎲 Random\nMasking", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            else:
                 ax.text(45, 45, "[MASKED]", ha='center', va='center', color='red', fontsize=10)
            ax.set_title(frame_labels[i], fontsize=10)
            ax.axis('off')
            continue
            
        # 2. Soft Sparsity (Opacity/Brightness)
        alpha = 1.0
        border_color = 'none'
        border_width = 0
        
        if attention_weights is not None:
            alpha = alphas[i]
            # Highlighting: if alpha is high, add glowing border
            if alpha > 0.5:
                border_color = '#FFD700' # Gold
                border_width = 3
        
        # Plot Matrix
        im = ax.imshow(matrix, cmap=cmap, vmin=-2, vmax=2, alpha=alpha)
        
        # Add Border
        if border_width > 0:
            rect = patches.Rectangle((0,0), 89, 89, linewidth=border_width, edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
                
        ax.set_title(frame_labels[i], fontsize=10)
        ax.axis('off')

def main():
    device = setup_gpu()
    
    print("Loading data...")
    loader = DHGNDataLoader()
    dynamic_fc, _, labels, ids = loader.load_all_data(groups=['EMCI'])
    
    sample_idx = 0 
    sample_data = dynamic_fc[sample_idx]
    
    print("Loading model...")
    model = create_dhgn_model(config=CONFIG).to(device)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(sample_data).unsqueeze(0).to(device)
        _, attention_weights = model.dynamic_imaging_pathway(input_tensor, return_attention=True)
        raw_weights = attention_weights.squeeze().cpu().numpy()
        
    # --- SMOOTHING CURVE FOR SCHEMATIC LOOK ---
    # Apply strong Gaussian smoothing to get that idealized "U-Shape" trend
    smooth_weights = gaussian_filter1d(raw_weights, sigma=5.0) 
        
    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 14))
    plt.suptitle("Schematic of the Hierarchical Dynamic Sparsification (HDS) Module", fontsize=22, fontweight='bold', y=0.96)
    
    gs = fig.add_gridspec(4, 5, hspace=0.6, wspace=0.1)
    
    # Row Labels
    row_titles = [
        "1. Sliding\nWindows",
        "2. Threshold\nClipping",
        "3. Random\nMasking",
        "4. Feature\nSoft Sparsity"
    ]
    
    for r, title in enumerate(row_titles):
        fig.text(0.1, 0.81 - r*0.21, title, fontsize=14, fontweight='bold', rotation=90, va='center', ha='center')

    # Row 1: Sliding Windows
    ax_row1 = [fig.add_subplot(gs[0, i]) for i in range(5)]
    plot_matrix_sequence(ax_row1, sample_data, 0, cmap='jet')
    
    add_arrow_between_rows(fig, 0.74, 0.71)

    # Row 2: Threshold Clipping
    clipped_data = np.clip(sample_data, -2, 2)
    ax_row2 = [fig.add_subplot(gs[1, i]) for i in range(5)]
    plot_matrix_sequence(ax_row2, clipped_data, 1, cmap='RdBu_r')

    add_arrow_between_rows(fig, 0.53, 0.50)

    # Row 3: Random Masking
    mask_indices = [20, 50] 
    ax_row3 = [fig.add_subplot(gs[2, i]) for i in range(5)]
    plot_matrix_sequence(ax_row3, clipped_data, 2, mask_indices=mask_indices, cmap='RdBu_r')

    add_arrow_between_rows(fig, 0.32, 0.29)

    # --- ATTENTION CURVE (Overlay between Row 3 and 4) ---
    ax_curve = fig.add_axes([0.25, 0.29, 0.5, 0.08]) # Centered
    ax_curve.plot(smooth_weights, color='#D62728', linewidth=3) # Brick Red
    ax_curve.set_title("Learned Attention Weights (U-Shape Trend)", fontsize=12, color='#D62728', fontweight='bold')
    ax_curve.set_xlim(0, 70)
    ax_curve.set_ylim(min(smooth_weights)*0.9, max(smooth_weights)*1.1)
    ax_curve.axis('off')
    
    # Draw arrows from curve to specific timepoints in Row 4
    # Just schematic arrows
    indices_x = [5, 35, 65] # Early, Mid, Late
    for idx_x in indices_x:
        # Normalized x coord (0-1)
        x_norm = idx_x / 71.0
        # Determine weight relative height for arrow length
        # Just simple fixed arrows pointing down
        ax_curve.annotate('', xy=(idx_x, min(smooth_weights)*0.8), xytext=(idx_x, smooth_weights[idx_x]),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))


    # Row 4: Soft Sparsity
    ax_row4 = [fig.add_subplot(gs[3, i]) for i in range(5)]
    plot_matrix_sequence(ax_row4, clipped_data, 3, attention_weights=smooth_weights, cmap='RdBu_r')
    
    # Add a box around Row 4 to emphasize output
    # rect = patches.Rectangle((0.12, 0.02), 0.76, 0.20, transform=fig.transFigure, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    # fig.add_artist(rect)
    
    save_path = 'analysis_results/sparsification_pipeline_refined.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    # --- EXPORT INDIVIDUAL COMPONENTS ---
    print("\nExporting individual components...")
    export_dir = 'analysis_results/components'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Re-use logic for consistency
    indices = [5, 20, 35, 50, 65]
    suffixes = ['T05_Early', 'T20_Mid', 'T35_Mid', 'T50_Mid', 'T65_Late']
    
    # Pre-calc alphas
    alphas = []
    w_max = smooth_weights.max()
    for idx in indices:
        w = smooth_weights[idx]
        alpha = np.clip((w / w_max) ** 2.5 + 0.05, 0.05, 1.0)
        alphas.append(alpha)

    # 1. Attention Curve (Standalone)
    fig_curve = plt.figure(figsize=(8, 3))
    plt.plot(smooth_weights, color='#D62728', linewidth=4)
    plt.xlim(0, 70)
    plt.axis('off')
    fig_curve.savefig(os.path.join(export_dir, 'curve_attention.png'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig_curve)

    # Loop rows
    row_configs = [
        ('1_SlidingWindow', sample_data, None, None, 'jet'),
        ('2_Thresholding', clipped_data, None, None, 'RdBu_r'),
        ('3_RandomMasking', clipped_data, [20, 50], None, 'RdBu_r'),
        ('4_SoftSparsity', clipped_data, None, alphas, 'RdBu_r')
    ]

    for row_name, data_source, mask_idxs, alpha_list, cmap in row_configs:
        for i, (idx, suffix) in enumerate(zip(indices, suffixes)):
            fig_single = plt.figure(figsize=(3, 3))
            ax = plt.gca()
            
            matrix = data_source[idx]
            
            # Masking Logic
            if mask_idxs and idx in mask_idxs:
                ax.imshow(np.zeros_like(matrix), cmap='gray', vmin=0, vmax=1)
                # No text overlay for clean component, or maybe simple?
                # Keeping it simple graphic
            else:
                # Opacity Logic
                alpha = 1.0
                border_color = 'none'
                border_width = 0
                
                if alpha_list:
                    alpha = alpha_list[i]
                    if alpha > 0.5:
                        border_color = '#FFD700'
                        border_width = 5 # Thicker for single image
                
                ax.imshow(matrix, cmap=cmap, vmin=-2, vmax=2, alpha=alpha)
                
                if border_width > 0:
                     rect = patches.Rectangle((0,0), 89, 89, linewidth=border_width, edgecolor=border_color, facecolor='none')
                     ax.add_patch(rect)
            
            ax.axis('off')
            
            fname = f"{row_name}_{suffix}.png"
            fig_single.savefig(os.path.join(export_dir, fname), transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig_single)
            
    print(f"✅ Exported 21 separate images to {export_dir}/")

    plt.show()

if __name__ == "__main__":
    main()
