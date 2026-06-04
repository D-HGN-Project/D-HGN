# D-HGN: Dynamic Hierarchical Graph Network

This repository contains the implementation of **D-HGN**, a structure-constrained dynamic hierarchical graph network for early mild cognitive impairment (eMCI) classification using rs-fMRI dynamic functional connectivity and DTI structural connectivity.

## Overview

D-HGN integrates two complementary neuroimaging modalities:

- **Dynamic imaging pathway:** extracts subject-level dynamic functional representations from sliding-window rs-fMRI dynamic functional connectivity sequences.
- **Structural connectivity pathway:** encodes DTI-derived white matter structural connectivity and constructs an anatomically constrained population graph.
- **Population graph classifier:** propagates dynamic functional embeddings across structurally similar subjects for eMCI/CN classification.

The code also includes baseline models, ablation experiments, unimodal comparison scripts, and visualization utilities for model interpretability.

## Repository Structure

```text
D-HGN_HBM_GitHub/
├── baseline_models.py              # Traditional sequence/GNN baseline models
├── data_loader.py                  # rs-fMRI dFC and DTI SC data loader
├── dhgn_model.py                   # Main D-HGN model
├── gpu_utils.py                    # GPU setup helpers
├── dynamic_imaging_pathway.py      # Dynamic imaging pathway for dFC feature extraction
├── population_graph_classifier.py  # Structure-constrained population graph classifier
├── structural_connectivity_pathway.py # DTI structural connectivity pathway
├── train_dhgn.py                   # Main D-HGN training script
├── train_baseline.py               # Baseline model training script
├── train_ablation.py               # Ablation study script
├── train_dti_only.py               # DTI-only comparison script
├── train_modality_comparison.py    # fMRI/DTI/multimodal comparison script
├── visualize_biomarkers.py         # Biomarker and connectivity visualization
├── visualize_sparsification.py     # Population graph sparsification visualization
├── visualize_tsne.py               # Embedding visualization
├── plot_*.py                       # Figure-generation utilities
├── data/                           # Dataset placeholder; ADNI data are not included
├── checkpoints/                    # Model checkpoint output directory
└── analysis_results/               # Visualization output directory
```

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n dhgn python=3.10 -y
conda activate dhgn
pip install -r requirements.txt
```

Install the PyTorch and PyTorch Geometric versions that match your CUDA environment. For example, please follow the official installation instructions for `torch`, `torchvision`, and `torch-geometric`.

## Data Preparation

The experiments use ADNI-derived rs-fMRI and DTI features. Raw ADNI data and processed subject-level files are not included in this repository because of data-use restrictions.

By default, `DHGNDataLoader` expects the following structure:

```text
data/
├── CN.csv
├── EMCI.csv
├── AD.csv                         # Optional, used for AD vs CN experiments
├── CN/
│   ├── GretnaDFCMatrixZ/
│   │   └── zsub071.mat
│   └── Final_SC_Matrices/
│       └── *number_of_tracts.connectivity.mat
├── EMCI/
│   ├── GretnaDFCMatrixZ/
│   │   └── zsub100.mat
│   └── Final_SC_Matrices/
│       └── *number_of_tracts.connectivity.mat
└── AD/
    ├── GretnaDFCMatrixZ/
    └── Final_SC_Matrices/
```

The CSV files should contain subject identifiers and demographic columns used by the loader, including `subject_id` or `Subject_ID`, `Subject`, `Age`, and `Sex`.

If your data use a different naming convention, update the subject-index ranges and file matching logic in `data_loader.py`.

## Training D-HGN

Run the main eMCI vs CN classification experiment:

```bash
python train_dhgn.py \
  --data_root ./data \
  --groups EMCI CN \
  --n_folds 5 \
  --gpu_id 0 \
  --ckpt_path ./checkpoints/dhgn
```

To run on CPU:

```bash
python train_dhgn.py --data_root ./data --groups EMCI CN --n_folds 5 --use_cpu
```

## Baseline Experiments

Available baseline model names include:

```text
mlp, brainnetcnn, gcn, lstm, transformer, gat, stgcn, diffpool, tcn, graphsage, gin, bilstm, gru
```

Example:

```bash
python train_baseline.py \
  --data_root ./data \
  --groups EMCI CN \
  --gpu_id 0 \
  --ckpt_path ./checkpoints
```

## Ablation and Modality Experiments

Run ablation experiments:

```bash
python train_ablation.py --data_root ./data --gpu_id 0 --num_epochs 100
```

Run DTI-only comparison:

```bash
python train_dti_only.py --data_root ./data --gpu_id 0 --num_epochs 100
```

Run modality comparison:

```bash
python train_modality_comparison.py --data_root ./data --gpu_id 0 --num_epochs 100
```

## Visualization

After training, generate biomarker and connectivity visualizations:

```bash
python visualize_biomarkers.py \
  --data_root ./data \
  --checkpoint ./checkpoints/dhgn/EMCI_vs_CN/fold_0_best.pth
```

Other figure-generation scripts include:

```bash
python visualize_tsne.py
python visualize_sparsification.py
python plot_attention_weights.py
python plot_attention_aggregation.py
python plot_temporal_sensitivity.py
python plot_dumbbell.py
python plot_chord_diagram.py
```

Generated figures are saved to `analysis_results/`.

## Reproducibility Notes

- The main training script fixes the random seed for reproducibility.
- Five-fold stratified cross-validation is used by default for the eMCI/CN task.
- The default dynamic functional connectivity setting assumes 90 ROIs, 130 time points, a 60-TR sliding window, and 71 dynamic windows.
- Checkpoint files and intermediate analysis outputs are intentionally excluded from version control.

## Citation

If you use this code, please cite the associated manuscript once it becomes available.

## License

This project is released under the license included in `LICENSE`.
