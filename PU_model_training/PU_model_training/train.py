from wandb.util import WandBHistoryJSONEncoder
import argparse
import ast
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import re
import numpy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure we can import the model
from multimodal_mlp import MultiModalPUPredictor


class TSVPUDataset(Dataset):
    """
    Dataset loaded from a TSV file.
    Expects columns: 'phenotype_vec', 'ppi_vec', 'label'
    """
    def __init__(self, tsv_file):
        self.df = pd.read_csv(
            tsv_file, 
            sep='\t', 
            converters={
                "phenotype_vec": ast.literal_eval, 
                "ppi_vec": ast.literal_eval
            }
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "pheno": torch.tensor(row["phenotype_vec"], dtype=torch.float32),
            "ppi": torch.tensor(row["ppi_vec"], dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal MLP")
    parser.add_argument('--config', type=str, help='Path to an optional JSON configuration file')
    
    # Core architectural arguments
    parser.add_argument('--use_ppi', type=lambda x: str(x).lower() == 'true', default=True, help='Use PPI features. If False, only pheno is used (ablation study).')
    parser.add_argument('--use_phenotypic', type=lambda x: str(x).lower() == 'true', default=True, help='Use phenotypic features. If False, only PPI is used (ablation study).')
    parser.add_argument('--fusion_type', type=str, choices=['concat', 'matmul', 'attention'], default='concat', help='Fusion technique for the two modalities')
    parser.add_argument('--ppi_net_type', type=str, choices=['mlp', 'attention'], default='mlp', help='Architecture type for the PPI branch')
    parser.add_argument('--ppi_vec_type', type=str, choices=['probs', 'logits'], default='probs', help='Whether ppi_vec contains probabilities or logits')
    
    # Dimensionality and hyperparameters
    parser.add_argument('--pheno_dim', type=int, default=3)
    parser.add_argument('--ppi_dim', type=int, default=440)
    parser.add_argument('--hidden_pheno', type=int, default=16)
    parser.add_argument('--hidden_ppi', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for Adam optimizer')
    parser.add_argument('--prior', type=float, default=0.1, help="Prior probability of positive class for PU learning")
    parser.add_argument('--loss_type', type=str, choices=['nnpu', 'wbce'], default='nnpu', help='Loss function to use: nnpu or wbce')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Weight for positive samples in WeightedBCELoss')
    parser.add_argument('--unl_weight', type=float, default=1.0, help='Weight for unlabeled samples in WeightedBCELoss')

    # Data arguments
    parser.add_argument('--train_data', type=str, help="Path to training data TSV")
    parser.add_argument('--val_data', type=str, help="Path to validation data TSV")

    # Training-specific arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help="Save a checkpoint every N epochs")
    parser.add_argument('--wandb_project', type=str, default="PU_learning")
    parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity/team name. Usually defaults to your username.")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    args, unknown = parser.parse_known_args()

    # Load config from JSON if provided, overriding defaults
    config = vars(args)
    if args.config:
        try:
            with open(args.config, 'r') as f:
                json_config = json.load(f)
                config.update(json_config)
        except Exception as e:
            print(f"Warning - failed to load config file {args.config}: {e}")

    # Infer run name based on configuration
    use_ppi_str = "yes" if config.get('use_ppi') else "no"
    use_pheno_str = "yes" if config.get('use_phenotypic', True) else "no"
    ppi_encoder_str = config.get('ppi_net_type') if config.get('use_ppi') else "None"
    ppi_vec_str = config.get('ppi_vec_type', 'probs')
    
    train_data_path = config.get('train_data') or ""
    train_data_name = os.path.basename(train_data_path)
    pu_ratio_match = re.search(r'(pu_ratio_1_[^.]+)', train_data_name)
    pu_ratio_str = pu_ratio_match.group(1) if pu_ratio_match else "pu_ratio_unknown"

    hidden_pheno = config.get('hidden_pheno')
    hidden_ppi = config.get('hidden_ppi')
    fusion_dim = config.get('fusion_dim')
    batch_size = config.get('batch_size')

    seed = config.get('seed')
    
    base_run_name = f"multimodal_PU_learning_pheno_{use_pheno_str}_ppi_{use_ppi_str}_type_{ppi_vec_str}_feature_fusion_{config.get('fusion_type')}_PPI_encoder_{ppi_encoder_str}_hidden_pheno_{hidden_pheno}_hidden_ppi_{hidden_ppi}_fusion_dim_{fusion_dim}_batch_size_{batch_size}_{pu_ratio_str}_seed_{seed}"
    
    if config.get('loss_type') == 'wbce':
        inferred_run_name = f"{base_run_name}_loss_wbce_pos_{config.get('pos_weight')}_unl_{config.get('unl_weight')}"
    else:
        inferred_run_name = f"{base_run_name}_loss_{config.get('loss_type')}_prior_{config.get('prior')}"
    
    if not config.get('wandb_run_name'):
        config['wandb_run_name'] = inferred_run_name

    print("\n" + "="*50)
    print("Running Training with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    if seed is not None:
        seed_everything(seed, workers=True)

    # 1. Setup Data
    if not config.get('train_data') or not config.get('val_data'):
        raise ValueError("Must provide train_data and val_data paths (via CLI or config json)")

    train_dataset = TSVPUDataset(config['train_data'])
    val_dataset = TSVPUDataset(config['val_data'])

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker
    )

    # 2. Setup Model
    model = MultiModalPUPredictor(
        pheno_dim=config.get('pheno_dim', 3),
        ppi_dim=config.get('ppi_dim', 440),
        hidden_pheno=config.get('hidden_pheno', 16),
        hidden_ppi=config.get('hidden_ppi', 64),
        fusion_dim=config.get('fusion_dim', 64),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        prior=config.get('prior', 0.1),
        loss_type=config.get('loss_type', 'nnpu'),
        pos_weight=config.get('pos_weight', 1.0),
        unl_weight=config.get('unl_weight', 1.0),
        l1_lambda=config.get('l1_lambda', 0.0),
        use_ppi=config.get('use_ppi', True),
        use_phenotypic=config.get('use_phenotypic', True),
        fusion_type=config.get('fusion_type', 'concat'),
        ppi_net_type=config.get('ppi_net_type', 'mlp')
    )

    # 3. Setup Logger
    # Wandb integrates seamlessly with PyTorch Lightning
    wandb_logger = WandbLogger(
        project=config.get('wandb_project'),
        name=config.get('wandb_run_name'),
        entity=config.get('wandb_entity'),
        log_model="all"
    )
    wandb_logger.experiment.config.update(config)

    # 4. Setup Callbacks
    # Save the model every N epochs
    checkpoint_dir = f"checkpoints/{config['wandb_run_name']}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='multimodal-mlp-{epoch:02d}',
        every_n_epochs=config.get('save_every_n_epochs', 5),
        save_top_k=-1,  # Save all checkpoints created at the specified intervals
    )

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='multimodal-mlp-best-{epoch:02d}-{val_auroc:.4f}',
        monitor='val_auroc',
        mode='max',
        save_top_k=1,
    )

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, best_checkpoint_callback],
        log_every_n_steps=10
    )

    # 6. Fit Model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training finished!")


if __name__ == '__main__':
    main()
