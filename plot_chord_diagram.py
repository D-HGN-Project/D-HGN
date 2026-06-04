import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from visualize_biomarkers import load_data_and_model, compute_saliency_maps

# Define the regions to match the reference plot exactly
# We extract only the regions present in the reference plot for top connections
REGIONS = {
    'Frontal': ['SFGdor', 'ORBsup', 'MFG', 'ORBmid', 'IFGoperc', 'IFGtriang', 'ORBinf', 'SMA', 'OLF', 'SFGmed', 'ORBsupmed', 'REC'],
    'Temporal': ['HES', 'STG', 'TPOsup', 'MTG', 'TPOmid', 'ITG'],
    'Parietal': ['PoCG', 'SPG', 'IPL', 'SMG', 'ANG', 'PCUN', 'PCL'],
    'Occipital': ['CAL', 'CUN', 'LING', 'SOG', 'MOG', 'IOG', 'FFG'],
    'Limbic': ['INS', 'ACG', 'DCG', 'PCG', 'HIP', 'PHG', 'AMYG'],
    'Subcortical': ['CAU', 'PUT', 'PAL', 'THA']
}

# The AAL90 index mapping
AAL90_LABELS = [
    'PreCG.L', 'PreCG.R', 'SFGdor.L', 'SFGdor.R', 'ORBsup.L', 'ORBsup.R', 
    'MFG.L', 'MFG.R', 'ORBmid.L', 'ORBmid.R', 'IFGoperc.L', 'IFGoperc.R', 
    'IFGtriang.L', 'IFGtriang.R', 'ORBinf.L', 'ORBinf.R', 'ROL.L', 'ROL.R', 
    'SMA.L', 'SMA.R', 'OLF.L', 'OLF.R', 'SFGmed.L', 'SFGmed.R', 
    'ORBsupmed.L', 'ORBsupmed.R', 'REC.L', 'REC.R', 'INS.L', 'INS.R', 
    'ACG.L', 'ACG.R', 'DCG.L', 'DCG.R', 'PCG.L', 'PCG.R', 
    'HIP.L', 'HIP.R', 'PHG.L', 'PHG.R', 'AMYG.L', 'AMYG.R', 
    'CAL.L', 'CAL.R', 'CUN.L', 'CUN.R', 'LING.L', 'LING.R', 
    'SOG.L', 'SOG.R', 'MOG.L', 'MOG.R', 'IOG.L', 'IOG.R', 
    'FFG.L', 'FFG.R', 'PoCG.L', 'PoCG.R', 'SPG.L', 'SPG.R', 
    'IPL.L', 'IPL.R', 'SMG.L', 'SMG.R', 'ANG.L', 'ANG.R', 
    'PCUN.L', 'PCUN.R', 'PCL.L', 'PCL.R', 'CAU.L', 'CAU.R', 
    'PUT.L', 'PUT.R', 'PAL.L', 'PAL.R', 'THA.L', 'THA.R', 
    'HES.L', 'HES.R', 'STG.L', 'STG.R', 'TPOsup.L', 'TPOsup.R', 
    'MTG.L', 'MTG.R', 'TPOmid.L', 'TPOmid.R', 'ITG.L', 'ITG.R'
]

# Map exact regions to AAL indices
AAL_INDICES = {label: i for i, label in enumerate(AAL90_LABELS)}

REGION_COLORS = {
    'Frontal': '#D98880',     # Dark Red/Pink
    'Temporal': '#F5B041',    # Orange
    'Parietal': '#5DADE2',    # Light Blue 
    'Occipital': '#A569BD',   # Purple
    'Limbic': '#48C9B0',      # Teal
    'Subcortical': '#EB984E'  # Brown/Orange
}

# The reference uses specific distinct colors for ribbons
RIBBON_COLORS = [
    '#D9534F', # Red
    '#5BC0DE', # Light Blue
    '#F0AD4E', # Orange
    '#5CB85C', # Green
    '#428BCA', # Dark Blue
    '#7D6608', # Brown
    '#8E44AD', # Purple
    '#1ABC9C', # Teal
    '#E74C3C', # Orange Red
    '#34495E'  # Dark Gray
]

def get_top_k_connections(diff_matrix, is_increased=True, k=10):
    mat = np.triu(diff_matrix, k=1)
    
    if is_increased:
        mat = np.where(mat > 0, mat, 0)
        flat_indices = np.argsort(mat.flatten())[-k:]
    else:
        mat = np.where(mat < 0, mat, 0)
        flat_indices = np.argsort(mat.flatten())[:k]
        
    conns = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, diff_matrix.shape)
        if mat[i, j] != 0:
            val = abs(diff_matrix[i, j])
            conns.append((i, j, val))
    
    # Sort by strength for coloring
    conns = sorted(conns, key=lambda x: x[2], reverse=True)
    return conns

def bezier_curve(p0, p1, p2, p3, n_points=100):
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    curve[:, 0] = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    curve[:, 1] = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return curve

def plot_beautiful_circos(conns, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.set_aspect('equal')
    
    # 1. Identify which nodes are involved
    active_nodes = set()
    for i, j, v in conns:
        active_nodes.add(i)
        active_nodes.add(j)
        
    active_nodes = list(active_nodes)
    
    # Group them roughly by lobe to make it look like the reference
    def get_lobe(node_idx):
        label = AAL90_LABELS[node_idx].split('.')[0]
        for lobe, regions in REGIONS.items():
            if label in regions:
                return lobe
        return 'Other'
        
    node_lobes = {n: get_lobe(n) for n in active_nodes}
    # Sort nodes by lobe
    active_nodes.sort(key=lambda n: (list(REGIONS.keys()).index(node_lobes[n]) if node_lobes[n] in REGIONS else 99, AAL90_LABELS[n]))
    
    num_nodes = len(active_nodes)
    if num_nodes == 0:
        return
        
    # Standard parameters
    radius = 1.0
    inner_radius = 0.95
    label_radius_multiplier = 1.08
    
    # Compute angles dynamically based on spacing (gap between lobes)
    angles = {}
    node_widths = {}
    
    # Calculate connection strength per node to determine the width of the node segment
    node_strengths = {n: 0 for n in active_nodes}
    for i, j, v in conns:
        node_strengths[i] += v
        node_strengths[j] += v
        
    total_strength = sum(node_strengths.values())
    total_gap_degrees = 40  # Total gap in degrees
    lobe_gap_degrees = 5
    
    usable_degrees = 360 - total_gap_degrees
    
    current_angle = 0
    node_angles = {}
    prev_lobe = None
    
    for n in active_nodes:
        lobe = node_lobes[n]
        if prev_lobe is not None and lobe != prev_lobe:
            current_angle += lobe_gap_degrees
            
        span = (node_strengths[n] / total_strength) * usable_degrees
        # Start and end angles for the node patch
        start_rad = np.deg2rad(current_angle)
        end_rad = np.deg2rad(current_angle + span)
        
        node_angles[n] = (start_rad, end_rad)
        
        # Draw node arc
        theta = np.linspace(start_rad, end_rad, 50)
        x_outer = radius * np.cos(theta)
        y_outer = radius * np.sin(theta)
        x_inner = inner_radius * np.cos(theta)
        y_inner = inner_radius * np.sin(theta)
        
        verts = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
        poly = patches.Polygon(verts, facecolor=REGION_COLORS.get(lobe, '#333333'), edgecolor='white', linewidth=1)
        ax.add_patch(poly)
        
        # Add Label
        mid_rad = (start_rad + end_rad) / 2
        lbl_x = radius * label_radius_multiplier * np.cos(mid_rad)
        lbl_y = radius * label_radius_multiplier * np.sin(mid_rad)
        
        rot = np.rad2deg(mid_rad)
        if rot > 90 and rot < 270:
            rot -= 180
            ha = 'right'
        else:
            ha = 'left'
            
        ax.text(lbl_x, lbl_y, AAL90_LABELS[n], rotation=rot, ha=ha, va='center', fontsize=9, fontweight='normal', fontfamily='sans-serif')
        
        # Tick marks
        tick_x1 = radius * 1.01 * np.cos(mid_rad)
        tick_y1 = radius * 1.01 * np.sin(mid_rad)
        tick_x2 = radius * 1.04 * np.cos(mid_rad)
        tick_y2 = radius * 1.04 * np.sin(mid_rad)
        ax.plot([tick_x1, tick_x2], [tick_y1, tick_y2], color='black', lw=0.8)
        
        current_angle += span + 1  # Add small gap between nodes in same lobe
        prev_lobe = lobe
    
    # 2. Draw connections (Ribbons)
    # Track current offset within each node for stacking ribbons
    node_offsets = {n: node_angles[n][0] for n in active_nodes}
    
    # Draw thicker connections first
    for idx, (i, j, v) in enumerate(conns):
        span_i = (v / total_strength) * np.deg2rad(usable_degrees)
        span_j = (v / total_strength) * np.deg2rad(usable_degrees)
        
        start_i = node_offsets[i]
        end_i = start_i + span_i
        node_offsets[i] = end_i
        
        start_j = node_offsets[j]
        end_j = start_j + span_j
        node_offsets[j] = end_j
        
        color = RIBBON_COLORS[idx % len(RIBBON_COLORS)]
        
        # Ribbon path
        r = inner_radius
        
        # Source arc
        theta_i = np.linspace(start_i, end_i, 20)
        p_i = np.column_stack((r * np.cos(theta_i), r * np.sin(theta_i)))
        
        # Target arc
        theta_j = np.linspace(end_j, start_j, 20)
        p_j = np.column_stack((r * np.cos(theta_j), r * np.sin(theta_j)))
        
        # Connect p_i end to p_j start with bezier
        p_i_end = p_i[-1]
        p_j_start = p_j[0]
        # Control points in the center
        cp1, cp2 = [0, 0], [0, 0]
        curve1 = bezier_curve(p_i_end, cp1, cp2, p_j_start, 50)
        
        # Connect p_j end to p_i start with bezier
        p_j_end = p_j[-1]
        p_i_start = p_i[0]
        curve2 = bezier_curve(p_j_end, cp1, cp2, p_i_start, 50)
        
        # Combine
        verts = np.vstack([p_i, curve1, p_j, curve2])
        poly = patches.Polygon(verts, facecolor=color, alpha=0.5, edgecolor=color, lw=0.5)
        ax.add_patch(poly)
        
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    
    # Add title label vertically on the far left side, similar to reference
    ax.text(-1.45, 0, title, rotation=90, va='center', ha='center', fontsize=22, fontfamily='serif')

def plot_top_circos(mean_nc, mean_emci, output_dir="analysis_results"):
    os.makedirs(output_dir, exist_ok=True)
    diff_matrix = mean_emci - mean_nc
    
    inc_conns = get_top_k_connections(diff_matrix, is_increased=True, k=10)
    dec_conns = get_top_k_connections(diff_matrix, is_increased=False, k=10)
    
    # Combine both plots into a single figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    plot_beautiful_circos(dec_conns, "Decreased", ax=axes[0])
    plot_beautiful_circos(inc_conns, "Increased", ax=axes[1])
    
    # Add group label at bottom, similar to reference
    fig.text(0.5, 0.05, "CN -> EMCI", ha='center', fontsize=28, fontfamily='serif')
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    filename = "fig11_circos_combined.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {os.path.join(output_dir, filename)}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='c:/PycharmProjects/D-HGN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dhgn/EMCI_vs_CN/fold_0_best.pth')
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        import glob
        ckpts = glob.glob('checkpoints/dhgn/EMCI_vs_CN/*.pth')
        ckpt_path = ckpts[0]
            
    print(f"Using checkpoint: {ckpt_path}")
    
    model, dynamic_graphs, sc_matrices, labels, device = load_data_and_model(
        data_root=args.data_root, checkpoint_path=ckpt_path)
    
    mean_nc, mean_emci = compute_saliency_maps(model, dynamic_graphs, sc_matrices, labels, device)
    plot_top_circos(mean_nc, mean_emci)
    print("Circos plots generated successfully.")
