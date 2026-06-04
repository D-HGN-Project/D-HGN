"""
Attentive Aggregation Visualization for D-HGN Dynamic Imaging Pathway.

Collapses the full [heads, T, T] attention matrix along the query axis
to produce a [heads, T] per-time-window importance vector (= column sum of
the softmax attention matrix). Plots CN vs eMCI for each head.

Output: analysis_results/fig_attentive_aggregation.png
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT  = 'c:/PycharmProjects/D-HGN'
CKPT_GLOB  = 'checkpoints/dhgn/EMCI_vs_CN/*.pth'
OUTPUT     = 'analysis_results/fig_attentive_aggregation.png'
NUM_WINDOWS = 71
NUM_HEADS   = 4

MODEL_CONFIG = {
    'num_rois': 90, 'num_windows': NUM_WINDOWS, 'num_classes': 2,
    'spatial_hidden_dim': 48, 'temporal_hidden_dim': 192,
    'st_output_dim': 192, 'use_sc': True,
    'sc_hidden_dim': 96, 'sc_output_dim': 48,
    'population_hidden_dim': 96, 'num_gnn_layers': 3, 'dropout': 0.4
}

CN_COLOR   = '#2E86AB'
EMCI_COLOR = '#E84855'
HEAD_COLORS = ['#5E4FA2', '#3288BD', '#D53E4F', '#F46D43']

# ─── Patch attention to capture full per-head weights ────────────────────────
def patch_attention(model):
    mha = model.dynamic_imaging_pathway.temporal_attention

    def new_forward(x):
        batch_size, num_windows, _ = x.shape
        Q = mha.q_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)
        K = mha.k_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)
        V = mha.v_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * mha.scale
        attn_weights = torch.softmax(scores, dim=-1)  # [B, heads, T, T]

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_windows, mha.hidden_dim)
        output = attn_output.mean(dim=1)
        output = mha.out_linear(output)

        avg_weights = attn_weights.mean(dim=1).mean(dim=-1, keepdim=True)
        return output, avg_weights, attn_weights  # extra: full weights

    mha.forward = new_forward
    return mha


def extract_aggregation(model, dynamic_graphs, sc_matrices, labels, device):
    """
    For each sample: sum attention over the query axis → per-window importance.
    Returns:
        agg_cn   : [subjects_cn, heads, T]
        agg_emci : [subjects_emci, heads, T]
    """
    model.eval()
    mha = patch_attention(model)

    accum_cn   = []
    accum_emci = []

    with torch.no_grad():
        for i in range(len(labels)):
            dfc = torch.tensor(dynamic_graphs[i:i+1], dtype=torch.float32, device=device)
            sc  = torch.tensor(sc_matrices[i:i+1],   dtype=torch.float32, device=device)

            st = model.dynamic_imaging_pathway
            batch_size, num_w, num_n, _ = dfc.shape
            x = dfc.view(-1, num_n, num_n)
            spatial_feat = st.spatial_extractor(x).flatten(start_dim=1)
            spatial_feat = spatial_feat.view(batch_size, num_w, -1)

            _, _, full_weights = mha(spatial_feat)   # [1, heads, T, T]
            # Column sum = how much each key time-window is attended to overall
            # shape: [heads, T]
            agg = full_weights[0].sum(dim=1).cpu().numpy()  # [heads, T]

            if labels[i] == 0:
                accum_cn.append(agg)
            else:
                accum_emci.append(agg)

    return np.stack(accum_cn), np.stack(accum_emci)


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot(agg_cn, agg_emci, out_dir):
    """
    agg_cn   : [N_cn, heads, T]
    agg_emci : [N_emci, heads, T]

    For each head, save a 2-row matrix heatmap:
      row 0 = mean aggregation weights for CN
      row 1 = mean aggregation weights for eMCI
    """
    from matplotlib.colors import LinearSegmentedColormap

    heads = agg_cn.shape[1]
    T     = agg_cn.shape[2]
    sigma = 1.5

    os.makedirs(out_dir, exist_ok=True)

    # Shared color scale across all heads for comparability
    global_max = max(
        agg_cn.mean(0).max(),
        agg_emci.mean(0).max()
    )
    global_min = min(
        agg_cn.mean(0).min(),
        agg_emci.mean(0).min()
    )

    # Blue-to-red diverging colormap: low -> neutral gray, high -> vibrant
    cmap = LinearSegmentedColormap.from_list(
        'agg_cmap',
        ['#D6EAF8', '#2E86AB', '#1B4F72']
    )

    tick_step = 10
    x_ticks = list(range(0, T, tick_step))

    for h in range(heads):
        mu_cn   = gaussian_filter1d(agg_cn[:, h, :].mean(0),   sigma)
        mu_emci = gaussian_filter1d(agg_emci[:, h, :].mean(0), sigma)

        # Matrix: [2, T]  row0=CN, row1=eMCI
        matrix = np.stack([mu_cn, mu_emci], axis=0)

        fig, ax = plt.subplots(figsize=(11, 2.0))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        im = ax.imshow(
            matrix,
            aspect='auto',
            cmap=cmap,
            vmin=global_min,
            vmax=global_max,
            interpolation='bilinear'
        )

        # Y-axis labels
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CN', 'eMCI'], fontsize=10, fontweight='bold')

        # X-axis ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks], fontsize=8.5)
        ax.set_xlabel('Time Window (fMRI Frames)', fontsize=10,
                      labelpad=4, color='#444444')

        # Spines
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(axis='y', length=0, colors='#333333')
        ax.tick_params(axis='x', colors='#666666', length=3)

        # Title
        ax.set_title(f'Head {h+1} — Attentive Aggregation Weights',
                     fontsize=11, fontweight='bold',
                     color='#222222', pad=7)

        # Colorbar on right
        cb = fig.colorbar(im, ax=ax, orientation='vertical',
                          fraction=0.018, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Weight', fontsize=8.5)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'head{h+1}_agg_matrix.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    OUTPUT_DIR = 'analysis_results/agg_matrices'

    print("Loading data ...")
    dl = DHGNDataLoader(data_root=DATA_ROOT)
    dynamic_graphs, sc_matrices, labels, _ = dl.load_all_data(
        groups=['EMCI', 'CN'], use_sc=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = create_dhgn_model(MODEL_CONFIG).to(device)

    ckpts = glob.glob(CKPT_GLOB)
    assert ckpts, "No checkpoint found!"
    ckpt_path = sorted(ckpts)[0]
    print(f"Loading weights: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)

    print("Extracting attentive aggregation weights ...")
    agg_cn, agg_emci = extract_aggregation(
        model, dynamic_graphs, sc_matrices, labels, device)
    print(f"  CN shape:   {agg_cn.shape}")
    print(f"  eMCI shape: {agg_emci.shape}")

    print("Plotting ...")
    plot(agg_cn, agg_emci, OUTPUT_DIR)
    print("Done.")
