import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from visualize_biomarkers import load_data_and_model, compute_saliency_maps

# AAL90 region labels
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

# Beautified region names for display
def format_label(raw_label):
    """
    Convert 'IFGtriang.L' -> 'Frontal_Inf_Tri_L' style labels
    to match the reference image
    """
    NAME_MAP = {
        'PreCG': 'Precentral', 'SFGdor': 'Frontal_Sup', 'ORBsup': 'Frontal_Sup_Orb',
        'MFG': 'Frontal_Mid', 'ORBmid': 'Frontal_Mid_Orb', 'IFGoperc': 'Rolandic_Oper',
        'IFGtriang': 'Frontal_Inf_Tri', 'ORBinf': 'Frontal_Inf_Orb', 'ROL': 'Rolandic_Oper',
        'SMA': 'Supp_Motor_Area', 'OLF': 'Olfactory', 'SFGmed': 'Frontal_Sup_Med',
        'ORBsupmed': 'Frontal_Med_Orb', 'REC': 'Rectus', 'INS': 'Insula',
        'ACG': 'Cingulum_Ant', 'DCG': 'Cingulum_Mid', 'PCG': 'Cingulum_Post',
        'HIP': 'Hippocampus', 'PHG': 'ParaHippocampal', 'AMYG': 'Amygdala',
        'CAL': 'Calcarine', 'CUN': 'Cuneus', 'LING': 'Lingual',
        'SOG': 'Occipital_Sup', 'MOG': 'Occipital_Mid', 'IOG': 'Occipital_Inf',
        'FFG': 'Fusiform', 'PoCG': 'Postcentral', 'SPG': 'Parietal_Sup',
        'IPL': 'Parietal_Inf', 'SMG': 'SupraMarginal', 'ANG': 'Angular',
        'PCUN': 'Precuneus', 'PCL': 'Paracentral_Lobule', 'CAU': 'Caudate',
        'PUT': 'Putamen', 'PAL': 'Pallidum', 'THA': 'Thalamus',
        'HES': 'Heschl', 'STG': 'Temporal_Sup', 'TPOsup': 'Temporal_Pole_Sup',
        'MTG': 'Temporal_Mid', 'TPOmid': 'Temporal_Pole_Mid', 'ITG': 'Temporal_Inf'
    }
    parts = raw_label.split('.')
    name = parts[0]
    side = parts[1] if len(parts) > 1 else ''
    display_name = NAME_MAP.get(name, name)
    return f'{display_name}_{side}'


def compute_nodal_strength(saliency_map):
    """Compute nodal strength (sum of connection weights per node) from saliency map."""
    strength = np.abs(saliency_map).sum(axis=1)
    return strength


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='c:/PycharmProjects/D-HGN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dhgn/EMCI_vs_CN/fold_0_best.pth')
    parser.add_argument('--top_k', type=int, default=15)
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        ckpts = glob.glob('checkpoints/dhgn/EMCI_vs_CN/*.pth')
        ckpt_path = ckpts[0]
        
    print(f"Using checkpoint: {ckpt_path}")
    
    model, dynamic_graphs, sc_matrices, labels, device = load_data_and_model(
        data_root=args.data_root, checkpoint_path=ckpt_path)
    
    mean_nc, mean_emci = compute_saliency_maps(model, dynamic_graphs, sc_matrices, labels, device)

    # Compute nodal strength for CN and EMCI
    strength_nc = compute_nodal_strength(mean_nc)
    strength_emci = compute_nodal_strength(mean_emci)

    # Normalize per group
    strength_nc_norm = strength_nc / strength_nc.mean()
    strength_emci_norm = strength_emci / strength_emci.mean()

    # Identify top K regions by the EMCI to CN difference
    diff = np.abs(strength_emci_norm - strength_nc_norm)
    top_indices = np.argsort(diff)[-args.top_k:][::-1]   # descending

    # Collect values
    labels_display = [format_label(AAL90_LABELS[i]) for i in top_indices]
    nc_vals = [strength_nc_norm[i] for i in top_indices]
    emci_vals = [strength_emci_norm[i] for i in top_indices]

    # Reverse for vertical top-to-bottom ordering
    labels_display = labels_display[::-1]
    nc_vals = nc_vals[::-1]
    emci_vals = emci_vals[::-1]

    # ─── Plot ────────────────────────────────────────────────────────────────
    CN_COLOR = '#2E86AB'     # Deep blue
    EMCI_COLOR = '#E84855'   # Brick red

    fig, ax = plt.subplots(figsize=(11, 8))

    y_pos = np.arange(len(labels_display))

    # Horizontal connecting line (dumbbell bar)
    for i, (nc, emci, y) in enumerate(zip(nc_vals, emci_vals, y_pos)):
        ax.plot([nc, emci], [y, y], color='#CCCCCC', linewidth=1.5, zorder=1)

    # CN dots
    ax.scatter(nc_vals, y_pos, color=CN_COLOR, s=100, zorder=2, label='CN')
    # EMCI dots
    ax.scatter(emci_vals, y_pos, color=EMCI_COLOR, s=100, zorder=2, label='EMCI')

    # Y-axis labels (brain region names)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_display, fontsize=10)

    # Title only (no subtitle, no x-label, no bottom annotation)
    ax.set_title('Top 15 Discriminative Brain Regions (CN vs EMCI)',
                 fontsize=14, fontweight='bold', pad=12)

    # Suppress x-axis grid (vertical dashed lines) and x-axis label
    ax.set_xlabel('')                             # no x-axis label
    ax.xaxis.grid(False)                          # no vertical grid lines
    ax.yaxis.grid(True, color='#EEEEEE', linewidth=0.6)
    ax.set_axisbelow(True)

    # Spine cleanup
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend — anchored just outside the upper-right corner
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        bbox_transform=ax.transAxes,
        frameon=True,
        framealpha=0.9,
        edgecolor='#CCCCCC',
        fontsize=10
    )

    plt.tight_layout()
    output_path = os.path.join('analysis_results', 'fig5_biomarker_regions.png')
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
