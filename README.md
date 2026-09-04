# ICARus — Code Repository

This repository contains all code accompanying the ICARus publication. The analysis is organised into five self-contained modules that together implement a complete pipeline for identifying VACV (Vaccinia virus) host factors from a pooled RNAi screen using multimodal machine learning.

## Overview

```
code_for_ICARus_repo/
├── VACV_screen_preprocessing/      # Control-based Z-scoring and QC of raw screen data
├── xCAPT5_fine-tuning/             # Fine-tuning of xCAPT5 for human–VACV PPI prediction
├── PU_model_training/              # Multimodal PU-learning model (phenotype + PPI)
├── screen_refinement/              # Refining screen intensities with PPI-based probabilities
└── overrepresentation_analysis/    # Gene-set ORA via DAVID and downstream visualisation
```

---

## Module 1 — VACV Screen Preprocessing (`VACV_screen_preprocessing/`)

**Purpose:** Quality-control and normalise the raw Dharmacon pooled RNAi screen data.

### Contents

| File | Description |
|---|---|
| `control_based_Z-scoring.ipynb` | Computes control-based Z-scores for each well across all plates |
| `controls_UMAP_investigation.ipynb` | UMAP visualisation of plate controls for stability/outlier detection |
| `investigating_intensity_stability.ipynb` | Investigates intra-plate intensity stability across replicates |
| `Dharmacon_pooled_G1_G2_screening_plates_subset_with.tsv` | Raw screen data subset |
| `Dharmacon_pooled_G1_G2_screening_plates_subset_control-based_Z-scored.tsv` | Z-scored screen data |

### Key concepts

Control-based Z-scoring is used instead of whole-plate Z-scoring to remove plate-composition bias. All plates contain the same set of controls at identical frequencies, allowing a consistent baseline to be established per plate. This makes cross-plate comparisons of Z-scores valid.

---

## Module 2 — xCAPT5 Fine-tuning (`xCAPT5_fine-tuning/`)

**Purpose:** Fine-tune the pre-trained [xCAPT5](https://github.com/Erastova-group/xCAPT5) protein–protein interaction (PPI) predictor — built on ProtT5-XL-U50 embeddings and a convolutional architecture (MCAPS) — to a human–VACV PPI dataset. The resulting model is used to generate PPI probability vectors for each human gene in the screen.

### Contents

```
xCAPT5_fine-tuning/
├── PPI_dataset/                                 # Human–VACV PPI data (FASTA + TSV splits)
├── MCAPST5-X_checkpoints/                       # Pre-trained MCAPS-T5 backbone checkpoints
├── MLP_head_checkpoints/                        # Saved fine-tuned MLP head weights
├── xCAPT5_with_MLP_head_fine-tuning_on_training_set/
│   ├── xCAPT5_fitting_MLP.py                    # Main fine-tuning script
│   └── xCAPT5_utils.py                          # Utility functions (embeddings, padding, CNN)
├── xCAPT5_with_MLP_head_evaluation_on_test_set/
│   ├── xCAPT5_with_MLP_inference.py             # Inference script for the test set
│   └── xCAPT5_utils.py                          # Utility functions
└── requirements_xCAPT5_with_MLP_head.txt        # Python environment specification
```

### PPI dataset splits

The `PPI_dataset/` directory contains a human–VACV PPI dataset in FASTA and TSV formats, split into training, validation, and test sets.

| File | Description |
|---|---|
| `entire_human-VACV_PPI_data_set.fasta` | All protein sequences (human + VACV) |
| `entire_human-VACV_PPI_data_set.tsv` | All interaction pairs with labels |
| `human-VACV_PPI_training_set.tsv` | Training split |
| `human-VACV_PPI_validation_set.tsv` | Validation split |
| `human-VACV_PPI_test_set.tsv` | Test split |

TSV files without the `_without_header` suffix include a header row; those with it do not (required by certain tools).

### Environment setup

This module requires a **separate Python 3.10 environment** due to TensorFlow/PyTorch co-dependency constraints:

```bash
pip install -r requirements_xCAPT5_with_MLP_head.txt
```

### Fine-tuning the MLP head

```bash
python xCAPT5_fitting_MLP.py \
    <train_pairs_file> \
    <val_pairs_file> \
    <fasta_file> \
    <path_to_MCAPST5_checkpoint> \
    <seed> \
    <output_name> \
    --lr 1e-4 \
    --hid_dim 16 \
    --weight_decay 1e-2 \
    --dropout 0.2 \
    --epochs 20
```

The script first generates per-residue ProtT5 embeddings (cached to `protT5/output/`), runs the frozen MCAPS-T5 backbone to obtain pair-level representations, and then trains a two-layer MLP head with BCE loss. Training and validation metrics (loss, ROC-AUC, PR-AUC) are logged to [Weights & Biases](https://wandb.ai). Checkpoints are saved every `--ckpt_interval` epochs and the best model (by validation loss) is saved as `best_model.pt`.

Two checkpoint variants are provided, trained on the **Pan** and **Sled** PPI datasets respectively.

---

## Module 3 — PU-Learning Model (`PU_model_training/`)

**Purpose:** Train a multimodal Positive-Unlabeled (PU) learning classifier that integrates two data modalities — phenotypic screen features and xCAPT5-derived PPI probability vectors — to prioritise candidate VACV host factors.

### Contents

```
PU_model_training/
├── PU_learning_dataset/
│   ├── data_formatting.ipynb              # Assembles and formats the training/validation TSVs
│   ├── train_validation_test_split.ipynb  # Splits data into train/val/test
│   ├── dataset_prior_to_formatting/       # Raw input data
│   └── formatted_dataset/                 # Model-ready TSV files
├── PU_model_training/
│   ├── train.py                           # Main training script
│   ├── multimodal_mlp.py                  # Model architecture
│   └── json_config_files/                 # Per-run JSON hyperparameter configs
├── PU_model_evaluation/
│   └── infer.py                           # Inference script
├── PU_model_checkpoints/                  # Saved model checkpoints (.ckpt)
└── requirements_pu_learning.txt           # Python environment specification
```

### Model architecture

`MultiModalPUPredictor` is a dual-branch MLP implemented in PyTorch Lightning:

- **Phenotype branch** — a two-layer MLP with LayerNorm and ReLU that encodes the 3-dimensional phenotypic feature vector (e.g., Z-scored intensities)
- **PPI branch** — a two-layer MLP (or self-attention encoder) that encodes the 440-dimensional PPI probability vector produced by xCAPT5
- **Adaptive modality gating** — a learned softmax gate dynamically weights the contribution of each modality per sample before fusion
- **Fusion** — supports `concat` (default), `matmul`, and cross-`attention` fusion strategies
- **Head** — a linear layer producing a single logit for binary classification

### PU-learning loss

Two loss functions are available:

| `--loss_type` | Description |
|---|---|
| `nnpu` (default) | Non-negative PU loss ([Kiryo et al., 2017](https://arxiv.org/abs/1703.00593)); requires a class-prior estimate `--prior` |
| `wbce` | Weighted binary cross-entropy; tune via `--pos_weight` and `--unl_weight` |

### Environment setup

```bash
# Python 3.12.4, CUDA 12.8
pip install -r requirements_pu_learning.txt
```

### Training

Training is configured via JSON files in `json_config_files/` (one per ablation / seed). Run with:

```bash
cd PU_model_training/PU_model_training

python train.py \
    --config json_config_files/both_modalities/config_nnPU_prior_0.05_fusion_type_concat_probs_both_modalities_seed_42.json
```

Alternatively, pass all arguments on the command line:

```bash
python train.py \
    --train_data <path_to_train.tsv> \
    --val_data   <path_to_val.tsv>   \
    --use_ppi true \
    --use_phenotypic true \
    --fusion_type concat \
    --loss_type nnpu \
    --prior 0.05 \
    --max_epochs 1000 \
    --seed 42
```

Training progress and metrics (loss, AUROC, PR-AUC, recall/precision/enrichment @50/100/200, modality gate weights) are logged to Weights & Biases. Checkpoints are saved periodically and the best checkpoint by validation AUROC is retained.

#### Ablation configurations

Pre-defined config directories cover the following ablation conditions:

| Directory | Condition |
|---|---|
| `both_modalities/` | Phenotype + PPI (full model) |
| `only_phenotype/` | Phenotype only |
| `only_PPIs/` | PPI only |
| `col_permutation_PPI_vec_without_phenotype/` | Column-permuted PPI vector (control) |
| `row_permutation_PPI_vec_without_phenotype/` | Row-permuted PPI vector (control) |

### Inference

```bash
cd PU_model_training/PU_model_evaluation

python infer.py \
    --ckpt_path  <path_to_checkpoint.ckpt> \
    --data_file  <path_to_data.tsv> \
    --output_file predictions.tsv
```

The output TSV contains `gene`, `logit`, and `probability` columns.

### Dataset format

Training/validation TSVs must contain the following tab-separated columns:

| Column | Type | Description |
|---|---|---|
| `phenotype_vec` | `list[float]` (stringified) | Phenotypic feature vector |
| `ppi_vec` | `list[float]` (stringified) | xCAPT5 PPI probability vector |
| `label` | `int` (0 or 1) | 1 = known positive; 0 = unlabeled |

---

## Module 4 — Screen Refinement (`screen_refinement/`)

**Purpose:** Refine raw phenotypic screen intensities by multiplying them with the host-factor probabilities predicted by the PU-learning model. This attenuates signal from genes unlikely to be true host factors and amplifies signal from high-confidence candidates.

### Contents

| File | Description |
|---|---|
| `refining_screen_intensities.ipynb` | Applies the refinement: intensity × PU probability |
| `data_for_refinement_of_entire_screen/` | Input data and the deployed model checkpoint |
| `unrefined_and_refined_early_intensities.tsv` | Early-infection intensities, before and after refinement |
| `unrefined_and_refined_late_intensities.tsv` | Late-infection intensities, before and after refinement |

The deployed checkpoint (`seed_45_multimodal-mlp-best-epoch=908-val_auroc=0.8966.ckpt`) is the best-performing model selected from the multi-seed sweep.

---

## Module 5 — Overrepresentation Analysis (`overrepresentation_analysis/`)

**Purpose:** Identify biological pathways and gene ontology terms enriched in the top-ranked candidate host factors, both before and after screen refinement.

### Contents

| File / Directory | Description |
|---|---|
| `overrepresentation_analysis_preparation.ipynb` | Selects top-250 gene lists per condition for DAVID input |
| `ORA_analysis.ipynb` | Parses DAVID output and computes summary statistics |
| `ORA_plot_generation.ipynb` | Generates publication-quality ORA figures |
| `top_250_gene_lists_unrefined_ints/` | Gene lists from unrefined intensities |
| `top_250_gene_lists_refined_ints/` | Gene lists from refined intensities |
| `DAVID_Func_Annot_Chart_ORA_results_unrefined_ints/` | Raw DAVID ORA results (unrefined) |
| `DAVID_Func_Annot_Chart_ORA_results_refined_ints/` | Raw DAVID ORA results (refined) |
| `unrefined_and_refined_intensities/` | Combined intensity tables used as ORA input |
| `ORA_comparison_average_-log10(Benjamini).tsv` | Aggregated ORA significance scores (wide format) |
| `ORA_comparison_average_-log10(Benjamini)_long_format.tsv` | Aggregated ORA significance scores (long format) |

### Workflow

1. Run `overrepresentation_analysis_preparation.ipynb` to derive the top-250 gene lists from unrefined and refined intensities (early and late; high and low) — eight lists in total.
2. Upload each gene list to the [DAVID Bioinformatics Resource](https://david.ncifcrf.gov) Functional Annotation Chart tool and download the results.
3. Place downloaded results in the corresponding `DAVID_Func_Annot_Chart_ORA_results_*/` directories.
4. Run `ORA_analysis.ipynb` to parse results and compute −log₁₀(Benjamini-corrected *p*-values).
5. Run `ORA_plot_generation.ipynb` to reproduce the publication figures.

---

## Requirements

Each major module ships with its own `requirements_*.txt` file specifying the exact Python and CUDA versions used. Two independent environments are needed:

| Environment | Python | CUDA | Requirements file |
|---|---|---|---|
| xCAPT5 fine-tuning | 3.10.4 | 12.8 | `xCAPT5_fine-tuning/requirements_xCAPT5_with_MLP_head.txt` |
| PU model + screen analysis | 3.12.4 | 12.8 | `PU_model_training/requirements_pu_learning.txt` |

The Jupyter notebooks (`*.ipynb`) in `VACV_screen_preprocessing/`, `overrepresentation_analysis/`, and elsewhere can be run in either environment, as they depend only on standard scientific Python packages (`numpy`, `pandas`, `matplotlib`).

---

## Experiment Tracking

All training runs are logged to [Weights & Biases](https://wandb.ai). Set your W&B entity in the JSON config files or via the `--wandb_entity` flag before running training.

---

## Citation

If you use this code, please cite the associated publication (details to be added upon acceptance).

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
