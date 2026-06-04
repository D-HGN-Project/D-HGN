"""
Visualize Multi-Head Temporal Attention weights from D-HGN's Dynamic Imaging Pathway.

Output: analysis_results/fig_temporal_attention.png
- 4 rows (one per attention head) × 2 cols (CN | eMCI)
- Each cell = mean attention matrix [time × time] across all subjects in that group
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
from data_loader import DHGNDataLoader
from dhgn_model import create_dhgn_model
from dynamic_imaging_pathway import MultiHeadTemporalAttention

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = 'c:/PycharmProjects/D-HGN'
CKPT_GLOB   = 'checkpoints/dhgn/EMCI_vs_CN/*.pth'
OUTPUT_DIR  = 'analysis_results/attention_heads'
NUM_WINDOWS = 71
NUM_HEADS   = 4

MODEL_CONFIG = {
    'num_rois': 90, 'num_windows': NUM_WINDOWS, 'num_classes': 2,
    'spatial_hidden_dim': 48, 'temporal_hidden_dim': 192,
    'st_output_dim': 192, 'use_sc': True,
    'sc_hidden_dim': 96, 'sc_output_dim': 48,
    'population_hidden_dim': 96, 'num_gnn_layers': 3, 'dropout': 0.4
}

# Time-axis tick labels (every 10 windows)
TICK_STEP = 10
TICK_POSITIONS = list(range(0, NUM_WINDOWS, TICK_STEP))
TICK_LABELS    = [str(i) for i in TICK_POSITIONS]

# ─── Load model with hook to capture raw per-head attention ─────────────────
class AttentionHook:
    """Captures the full [batch, heads, time, time] attention weights."""
    def __init__(self):
        self.weights = None

    def hook_fn(self, module, input, output):
        # output is (attn_output, avg_weights); we need raw scores before avg
        pass  # we'll override forward instead


def patch_attention(model):
    """Monkey-patch MultiHeadTemporalAttention.forward to return full weights."""
    mha = model.dynamic_imaging_pathway.temporal_attention

    def new_forward(x):
        batch_size, num_windows, _ = x.shape
        Q = mha.q_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)
        K = mha.k_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)
        V = mha.v_linear(x).view(batch_size, num_windows, mha.num_heads, mha.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * mha.scale  # [B, heads, T, T]
        attn_weights = torch.softmax(scores, dim=-1)                # [B, heads, T, T]

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_windows, mha.hidden_dim)
        output = attn_output.mean(dim=1)
        output = mha.out_linear(output)

        avg_weights = attn_weights.mean(dim=1).mean(dim=-1, keepdim=True)
        return output, avg_weights, attn_weights   # extra return

    mha.forward = new_forward
    return mha


def extract_full_attention(model, dynamic_graphs, sc_matrices, labels, device):
    """
    Returns:
        attn_cn_mean   : [heads, T, T]  mean over CN subjects
        attn_emci_mean : [heads, T, T]  mean over eMCI subjects
    """
    model.eval()
    mha = patch_attention(model)

    accum_cn   = []
    accum_emci = []

    with torch.no_grad():
        for i in range(len(labels)):
            dfc = torch.tensor(dynamic_graphs[i:i+1], dtype=torch.float32, device=device)
            sc  = torch.tensor(sc_matrices[i:i+1],   dtype=torch.float32, device=device)

            # Run spatial extractor first (same as SpatioTemporalExtractor.forward)
            st = model.dynamic_imaging_pathway
            batch_size, num_w, num_n, _ = dfc.shape
            x = dfc.view(-1, num_n, num_n)
            spatial_feat = st.spatial_extractor(x).flatten(start_dim=1)
            spatial_feat = spatial_feat.view(batch_size, num_w, -1)

            # Call patched attention
            _, _, full_weights = mha(spatial_feat)   # [1, heads, T, T]

            w = full_weights[0].cpu().numpy()  # [heads, T, T]
            if labels[i] == 0:   # CN
                accum_cn.append(w)
            else:                # eMCI
                accum_emci.append(w)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(labels)}")

    cn_mean   = np.mean(accum_cn,   axis=0)  # [heads, T, T]
    emci_mean = np.mean(accum_emci, axis=0)  # [heads, T, T]
    return cn_mean, emci_mean


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot(cn_mean, emci_mean, out_dir):
    heads = cn_mean.shape[0]  # 4

    BG = '#FAFAFA'
    global_max = max(cn_mean.max(), emci_mean.max())

    from matplotlib.colors import LinearSegmentedColormap
    os.makedirs(out_dir, exist_ok=True)

    groups = [('CN',   cn_mean,   '#2E86AB', 'Blues'),
              ('eMCI', emci_mean, '#E84855', 'Reds')]

    for h in range(heads):
        for grp_name, data, clr, _ in groups:
            if grp_name == 'CN':
                cmap = LinearSegmentedColormap.from_list(
                    'cn_cmap', ['#FFFFFF', '#AED6F1', '#2E86AB'])
            else:
                cmap = LinearSegmentedColormap.from_list(
                    'emci_cmap', ['#FFFFFF', '#F5B7B1', '#E84855'])

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            im = ax.imshow(data[h], aspect='auto', cmap=cmap,
                           vmin=0, vmax=global_max, interpolation='nearest')

            ax.set_xticks(TICK_POSITIONS)
            ax.set_xticklabels(TICK_LABELS, fontsize=8)
            ax.set_yticks(TICK_POSITIONS)
            ax.set_yticklabels(TICK_LABELS, fontsize=8)

            ax.set_xlabel('Key Time Window', fontsize=10, labelpad=4)
            ax.set_ylabel('Query Time Window', fontsize=10, labelpad=4)

            ax.set_title(
                f'Head {h+1}  —  {grp_name}',
                fontsize=12, fontweight='bold', color='#222222', pad=8)

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)
            cb.set_label('Attention Weight', fontsize=9)

            # Spine cleanup
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#CCCCCC')

            fname = os.path.join(out_dir, f'head{h+1}_{grp_name}.png')
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved: {fname}")
            plt.close()



# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

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

    print("Extracting per-head attention ...")
    cn_mean, emci_mean = extract_full_attention(
        model, dynamic_graphs, sc_matrices, labels, device)
    print(f"  CN:   {cn_mean.shape}")
    print(f"  eMCI: {emci_mean.shape}")

    print("Plotting ...")
    plot(cn_mean, emci_mean, OUTPUT_DIR)
    print("Done.")
