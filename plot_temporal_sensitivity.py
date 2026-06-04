"""
Generate a premium Temporal Importance Profile (U-Shape) plot
comparing CN vs eMCI groups using saliency map temporal attention weights.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter1d
import torch
from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT  = 'c:/PycharmProjects/D-HGN'
CKPT_GLOB  = 'checkpoints/dhgn/EMCI_vs_CN/*.pth'
OUTPUT     = 'analysis_results/fig_temporal_ushape.png'
NUM_WINDOWS = 71      # number of sliding dFC windows

MODEL_CONFIG = {
    'num_rois': 90, 'num_windows': NUM_WINDOWS, 'num_classes': 2,
    'spatial_hidden_dim': 48, 'temporal_hidden_dim': 192,
    'st_output_dim': 192, 'use_sc': True,
    'sc_hidden_dim': 96, 'sc_output_dim': 48,
    'population_hidden_dim': 96, 'num_gnn_layers': 3, 'dropout': 0.4
}

# ─── Premium Color Palette ───────────────────────────────────────────────────
BG_COLOR      = '#FAFAFA'
GRID_COLOR    = '#E8E8E8'
CN_LINE       = '#2E86AB'      # steel blue
EMCI_LINE     = '#E84855'      # brick red
CN_FILL       = '#2E86AB'
EMCI_FILL     = '#E84855'

# ─── Load data & model ───────────────────────────────────────────────────────
def load():
    dl = DHGNDataLoader(data_root=DATA_ROOT)
    dynamic_graphs, sc_matrices, labels, _ = dl.load_all_data(
        groups=['EMCI', 'CN'], use_sc=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_dhgn_model(MODEL_CONFIG).to(device)

    ckpts = glob.glob(CKPT_GLOB)
    assert ckpts, "No checkpoint found!"
    ckpt_path = sorted(ckpts)[0]
    print(f"Loading weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, dynamic_graphs, sc_matrices, labels, device


# ─── Extract per-sample temporal sensitivity ─────────────────────────────────
def temporal_sensitivity(model, dynamic_graphs, sc_matrices, labels, device):
    """
    For each sample → backprop w.r.t. dFC → average gradient magnitude over
    ROI pairs → yields a [num_windows] temporal importance curve per sample.
    """
    curves_cn   = []
    curves_emci = []

    for i in range(len(labels)):
        dfc = torch.tensor(dynamic_graphs[i:i+1], dtype=torch.float32, device=device)
        sc  = torch.tensor(sc_matrices[i:i+1],  dtype=torch.float32, device=device)
        dfc.requires_grad_(True)
        sc.requires_grad_(True)

        logits = model(dfc, sc)
        model.zero_grad()
        logits[0, 1].backward()   # gradient w.r.t. eMCI class score

        # dfc shape: [1, T, N, N] or [1, T, N] depending on loader
        grad = dfc.grad.detach().abs()   # [1, T, N, N] or [1, T, N]
        # Collapse all spatial dims to get per-window importance
        while grad.dim() > 2:
            grad = grad.mean(-1)
        curve = grad[0].cpu().numpy()    # [T]

        if labels[i] == 0:
            curves_cn.append(curve)
        else:
            curves_emci.append(curve)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(labels)} done")

    return np.array(curves_cn), np.array(curves_emci)


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot(curves_cn, curves_emci, out_path):
    T = curves_cn.shape[1]
    x = np.arange(1, T + 1)

    # Smooth & stats
    sigma = 2.5
    mu_cn   = gaussian_filter1d(curves_cn.mean(0),   sigma)
    std_cn  = gaussian_filter1d(curves_cn.std(0),    sigma)
    mu_emci = gaussian_filter1d(curves_emci.mean(0), sigma)
    std_emci= gaussian_filter1d(curves_emci.std(0),  sigma)

    # Normalise to [0, 1] range for readability
    global_max = max(mu_cn.max() + std_cn.max(), mu_emci.max() + std_emci.max())
    global_min = min(mu_cn.min() - std_cn.min(), mu_emci.min() - std_emci.min())
    def norm(v): return (v - global_min) / (global_max - global_min + 1e-9)

    mu_cn_n   = norm(mu_cn);   std_cn_n  = norm(std_cn)
    mu_emci_n = norm(mu_emci); std_emci_n= norm(std_emci)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Light horizontal grid only
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle='--', alpha=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Confidence bands
    ax.fill_between(x,
                    mu_cn_n - std_cn_n,
                    mu_cn_n + std_cn_n,
                    color=CN_FILL, alpha=0.18, linewidth=0)
    ax.fill_between(x,
                    mu_emci_n - std_emci_n,
                    mu_emci_n + std_emci_n,
                    color=EMCI_FILL, alpha=0.18, linewidth=0)

    # Main lines
    ax.plot(x, mu_cn_n,   color=CN_LINE,   linewidth=2.8, label='CN',   zorder=3)
    ax.plot(x, mu_emci_n, color=EMCI_LINE, linewidth=2.8, label='eMCI', zorder=3)

    # ── Phase annotations (arrows + labels) ─────────────────────────────────
    arrowprops = dict(arrowstyle='->', color='#888888', lw=1.4,
                      connectionstyle='arc3,rad=0.0')
    ax.annotate('State Transition\n(Sensitivity)',
                xy=(4, mu_emci_n[3]), xytext=(9, mu_emci_n[3] + 0.14),
                fontsize=9.5, color='#555555', ha='center',
                arrowprops=arrowprops)
    ax.annotate('State Maintenance\n(Fatigue)',
                xy=(T-3, mu_emci_n[-4]),
                xytext=(T-12, mu_emci_n[-4] + 0.12),
                fontsize=9.5, color='#555555', ha='center',
                arrowprops=arrowprops)



    # ── Spines ──────────────────────────────────────────────────────────────
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#CCCCCC')

    # ── Labels ──────────────────────────────────────────────────────────────
    ax.set_xlabel('Time Window (fMRI Frames)', fontsize=12, labelpad=8, color='#444444')
    ax.set_ylabel('Relative Importance (Normalised)', fontsize=12, labelpad=8, color='#444444')
    ax.set_title('Temporal Importance Profile: The "U-Shape" Dynamics',
                 fontsize=14, fontweight='bold', pad=14, color='#222222')

    ax.tick_params(colors='#666666', labelsize=10)
    ax.set_xlim(1, T)
    ax.set_ylim(-0.05, 1.15)

    # ── Legend — outside upper right ─────────────────────────────────────────
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        bbox_transform=ax.transAxes,
        frameon=True,
        framealpha=0.9,
        edgecolor='#DDDDDD',
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    model, dynamic_graphs, sc_matrices, labels, device = load()
    print("Extracting temporal sensitivity ...")
    curves_cn, curves_emci = temporal_sensitivity(
        model, dynamic_graphs, sc_matrices, labels, device)
    print(f"  CN curves: {curves_cn.shape}, eMCI curves: {curves_emci.shape}")
    plot(curves_cn, curves_emci, OUTPUT)
    print("Done.")
