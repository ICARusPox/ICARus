import argparse
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure we can import the model
from multimodal_mlp import MultiModalPUPredictor

class InferenceDataset(Dataset):
    def __init__(self, df):
        self.genes = df['gene'].values
        self.pheno = torch.tensor(df['phenotype_vec'].tolist(), dtype=torch.float32)
        if 'ppi_vec' in df.columns:
            self.ppi = torch.tensor(df['ppi_vec'].tolist(), dtype=torch.float32)
        else:
            self.ppi = None

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        item = {
            "gene": self.genes[idx],
            "pheno": self.pheno[idx],
        }
        if self.ppi is not None:
            item["ppi"] = self.ppi[idx]
        return item

def main():
    parser = argparse.ArgumentParser(description="Run Inference on Multimodal MLP")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the trained .ckpt model file')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the TSV data file for inference')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_file', type=str, default='predictions.tsv', help='Where to save the predictions')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from checkpoint: {args.ckpt_path}...")
    
    # PyTorch Lightning handles recreating the architecture seamlessly via load_from_checkpoint
    try:
        model = MultiModalPUPredictor.load_from_checkpoint(args.ckpt_path)
        model = model.to(device)
    except FileNotFoundError:
        print(f"Error: The checkpoint file {args.ckpt_path} does not exist.")
        return

    model.eval()

    # Get dimensions dynamically from the loaded model to match correctly constraints
    pheno_dim = model.hparams.pheno_dim
    ppi_dim = model.hparams.ppi_dim

    # Setup Inference Data
    print(f"Loading inference data from {args.data_file}...")
    df = pd.read_csv(
        args.data_file, 
        sep='\t', 
        converters={"phenotype_vec": ast.literal_eval, "ppi_vec": ast.literal_eval}
    )
    infer_dataset = InferenceDataset(df)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    num_predictions = 0

    print("Running inference...")
    with torch.no_grad(), open(args.output_file, 'w') as f:
        f.write("gene\tlogit\tprobability\n")
        for batch in infer_loader:
            genes = batch["gene"]
            pheno = batch["pheno"].to(device)
            ppi = batch.get("ppi")

            if ppi is not None:
                ppi = ppi.to(device)
            
            # Predict
            logits = model(pheno, ppi)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            logits_np = logits.view(-1).cpu().numpy()
            
            for gene, logit, prob in zip(genes, logits_np, probs):
                f.write(f"{gene}\t{logit:.6f}\t{prob:.6f}\n")
                num_predictions += 1

    print(f"Inference complete! Saved {num_predictions} predictions to {args.output_file}")


if __name__ == '__main__':
    main()
