import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, unl_weight=1.0):
        """
        pos_weight: weight for positive samples (label=1)
        unl_weight: weight for unlabeled samples (label=0)
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.unl_weight = unl_weight

    def forward(self, logits, labels):
        """
        logits: (B,)
        labels: (B,) with 1 (positive) or 0 (unlabeled)
        """

        # element-wise BCE (no reduction yet)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none'
        )

        # apply weights
        weights = torch.where(labels == 1,
                              torch.full_like(labels, self.pos_weight, dtype=torch.float32),
                              torch.full_like(labels, self.unl_weight, dtype=torch.float32))

        weighted_loss = loss * weights

        return weighted_loss.mean()


class nnPULoss(nn.Module):
    def __init__(self, prior, beta=0.0, gamma=1.0):
        super().__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)

        pos_mask = labels == 1
        unl_mask = labels == 0

        # Handle edge cases safely
        if pos_mask.sum() == 0:
            return logits.sum() * 0.0
        if unl_mask.sum() == 0:
            return logits.sum() * 0.0

        # Positive risk
        pos_loss = F.binary_cross_entropy(
            probs[pos_mask],
            torch.ones_like(probs[pos_mask]),
            reduction='mean'
        )

        # Negative risk (from unlabeled)
        neg_loss = F.binary_cross_entropy(
            probs[unl_mask],
            torch.zeros_like(probs[unl_mask]),
            reduction='mean'
        )

        # Correction term
        neg_risk = neg_loss - self.prior * F.binary_cross_entropy(
            probs[pos_mask],
            torch.zeros_like(probs[pos_mask]),
            reduction='mean'
        )

        # nnPU correction
        if neg_risk < -self.beta:
            neg_risk = -self.gamma * neg_risk

        risk = self.prior * pos_loss + neg_risk
        return risk


class MultiModalPUPredictor(pl.LightningModule):
    def __init__(
        self,
        pheno_dim=3,
        ppi_dim=440,
        hidden_pheno=16,
        hidden_ppi=64,
        fusion_dim=64,
        lr=1e-3,
        weight_decay=1e-4,
        prior=0.1,
        loss_type='nnpu',
        pos_weight=1.0,
        unl_weight=1.0,
        l1_lambda=0.0,
        use_ppi=True,
        use_phenotypic=True,
        fusion_type='concat', # 'concat', 'attention', 'matmul'
        ppi_net_type='mlp'    # 'mlp', 'attention'
    ):
        super().__init__()

        self.save_hyperparameters()
        self.use_ppi = use_ppi
        self.use_phenotypic = use_phenotypic
        self.fusion_type = fusion_type
        self.ppi_net_type = ppi_net_type

        if not self.use_ppi and not self.use_phenotypic:
            raise ValueError("At least one modality must be used.")

        # Metrics
        self.val_auroc = torchmetrics.classification.BinaryAUROC()
        self.val_pr_auc = torchmetrics.classification.BinaryAveragePrecision()

        self.val_preds = []
        self.val_labels = []

        self.alpha_weights = []
        self.beta_weights = []

        self.pheno_unscaled_l2s = []
        self.ppi_unscaled_l2s = []
        self.pheno_scaled_l2s = []
        self.ppi_scaled_l2s = []
        self.val_pheno_unscaled_embs = []
        self.val_ppi_unscaled_embs = []

        # ------------------
        # Phenotype branch
        # ------------------
        if self.use_phenotypic:
            self.pheno_net = nn.Sequential(
                nn.Linear(pheno_dim, hidden_pheno),
                nn.LayerNorm(hidden_pheno),
                nn.ReLU(),
                nn.Dropout(0.1),  # mild

                nn.Linear(hidden_pheno, hidden_pheno),
                nn.LayerNorm(hidden_pheno),
                nn.ReLU()
            )

        # ------------------
        # PPI branch
        # ------------------
        if self.use_ppi:
            if self.ppi_net_type == 'mlp':
                self.ppi_net = nn.Sequential(
                    nn.Linear(ppi_dim, hidden_ppi),
                    nn.LayerNorm(hidden_ppi),
                    nn.ReLU(),
                    nn.Dropout(0.3),  # stronger regularization

                    nn.Linear(hidden_ppi, hidden_ppi),
                    nn.LayerNorm(hidden_ppi),
                    nn.ReLU()
                )
            elif self.ppi_net_type == 'attention':
                # Replace MLP with a self-attention mechanism
                # Project the input vector, treat it as a sequence of length 1, apply attention
                self.ppi_proj = nn.Linear(ppi_dim, hidden_ppi)
                self.ppi_attn = nn.MultiheadAttention(embed_dim=hidden_ppi, num_heads=4, batch_first=True, dropout=0.3)
                self.ppi_norm = nn.LayerNorm(hidden_ppi)
                self.ppi_ff = nn.Sequential(
                    nn.Linear(hidden_ppi, hidden_ppi),
                    nn.ReLU()
                )
            else:
                raise ValueError(f"Unknown ppi_net_type: {self.ppi_net_type}")
        
        # ------------------
        # Modality gating
        # ------------------
        if self.use_ppi and self.use_phenotypic:

            self.pheno_confidence = nn.Sequential(
                nn.Linear(hidden_pheno, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

            self.ppi_confidence = nn.Sequential(
                nn.Linear(hidden_ppi, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        # ------------------
        # Fusion
        # ------------------
        if not self.use_ppi:
            # Ablation study: Only Phenotype features used
            self.fusion_net = nn.Sequential(
                nn.Linear(hidden_pheno, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1)
            )
        elif not self.use_phenotypic:
            # Ablation study: Only PPI features used
            self.fusion_net = nn.Sequential(
                nn.Linear(hidden_ppi, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1)
            )
        else:
            if self.fusion_type == 'concat':
                input_dim = hidden_pheno + hidden_ppi
            elif self.fusion_type == 'matmul':
                # Performs a matrix multiplication of the embeddings:
                # (B, hidden_ppi, 1) @ (B, 1, hidden_pheno) -> (B, hidden_ppi, hidden_pheno)
                # Then flattened. This conceptually matches the nx1 @ 1xm dimensional transformation.
                input_dim = hidden_ppi * hidden_pheno
            elif self.fusion_type == 'attention':
                self.attn_dim = fusion_dim
                self.pheno_proj = nn.Linear(hidden_pheno, self.attn_dim)
                self.ppi_proj_attn = nn.Linear(hidden_ppi, self.attn_dim)
                self.cross_attn = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=4, batch_first=True)
                input_dim = self.attn_dim * 2
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")

            self.fusion_net = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1)
            )

        self.loss_type = loss_type
        if self.loss_type == 'nnpu':
            self.criterion = nnPULoss(prior=prior)
        elif self.loss_type == 'wbce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight, unl_weight=unl_weight)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

    def forward(self, pheno=None, ppi=None, return_embeddings=False):
        embeddings_dict = {}

        if self.use_phenotypic:
            pheno_emb = self.pheno_net(pheno)
            embeddings_dict["pheno_emb"] = pheno_emb
        else:
            pheno_emb = None

        if self.use_ppi:
            # PPI forward
            if self.ppi_net_type == 'mlp':
                ppi_emb = self.ppi_net(ppi)
            elif self.ppi_net_type == 'attention':
                x = self.ppi_proj(ppi).unsqueeze(1) # (B, 1, hidden_ppi)
                attn_out, _ = self.ppi_attn(x, x, x)
                x = self.ppi_norm(x + attn_out)
                x = x + self.ppi_ff(x)
                ppi_emb = x.squeeze(1) # (B, hidden_ppi)
            
            embeddings_dict["ppi_emb"] = ppi_emb
        else:
            ppi_emb = None

        if not self.use_ppi:
            # Only pheno
            fusion_emb = self.fusion_net[:-1](pheno_emb)
            logits = self.fusion_net[-1](fusion_emb).squeeze(-1)
        elif not self.use_phenotypic:
            # Only PPI
            fusion_emb = self.fusion_net[:-1](ppi_emb)
            logits = self.fusion_net[-1](fusion_emb).squeeze(-1)
        else:
            # Fusion
            if self.fusion_type == 'concat':
                # --------------------------------------
                # Adaptive modality weighting
                # --------------------------------------
                pheno_logits = self.pheno_confidence(pheno_emb)
                ppi_logits = self.ppi_confidence(ppi_emb)

                gate_logits = torch.cat(
                    [pheno_logits, ppi_logits],
                    dim=1
                )

                gate_weights = torch.softmax(gate_logits, dim=1)

                alpha = gate_weights[:, 0].unsqueeze(1)
                beta  = gate_weights[:, 1].unsqueeze(1)

                embeddings_dict["pheno_emb_unscaled"] = pheno_emb
                embeddings_dict["ppi_emb_unscaled"] = ppi_emb

                pheno_emb = alpha * pheno_emb
                ppi_emb   = beta  * ppi_emb

                # Update the embeddings dictionary
                embeddings_dict["pheno_emb"] = pheno_emb
                embeddings_dict["ppi_emb"] = ppi_emb

                fusion_input = torch.cat([pheno_emb, ppi_emb], dim=1)

                embeddings_dict["alpha"] = alpha
                embeddings_dict["beta"] = beta
            elif self.fusion_type == 'matmul':
                # Matrix multiplication between extracted features to fuse them
                # (B, hidden_ppi, 1) @ (B, 1, hidden_pheno) -> (B, hidden_ppi, hidden_pheno)
                batch_size = pheno_emb.size(0)
                res = torch.bmm(ppi_emb.unsqueeze(2), pheno_emb.unsqueeze(1))
                fusion_input = res.view(batch_size, -1)
            elif self.fusion_type == 'attention':
                # Attention-based fusion (Self-attention over projected embeddings)
                p_proj = self.pheno_proj(pheno_emb).unsqueeze(1) # (B, 1, attn_dim)
                p_ppi = self.ppi_proj_attn(ppi_emb).unsqueeze(1) # (B, 1, attn_dim)
                
                seq = torch.cat([p_proj, p_ppi], dim=1) # (B, 2, attn_dim)
                attn_out, _ = self.cross_attn(seq, seq, seq)
                
                fusion_input = attn_out.reshape(attn_out.size(0), -1) # (B, 2 * attn_dim)

            fusion_emb = self.fusion_net[:-1](fusion_input)
            logits = self.fusion_net[-1](fusion_emb).squeeze(-1)

        embeddings_dict["fusion_emb"] = fusion_emb

        if return_embeddings:
            return logits, embeddings_dict

        return logits

    def training_step(self, batch, batch_idx):
        pheno = batch.get("pheno") if self.use_phenotypic else None
        ppi = batch.get("ppi") if self.use_ppi else None
        labels = batch["label"].float()

        logits, embeddings = self(pheno, ppi, return_embeddings=True)
        loss = self.criterion(logits, labels)

        if self.l1_lambda > 0:
            l1_loss = embeddings["fusion_emb"].abs().mean()
            if self.use_phenotypic:
                l1_loss += embeddings["pheno_emb"].abs().mean()
            if self.use_ppi:
                l1_loss += embeddings["ppi_emb"].abs().mean()
            loss = loss + self.l1_lambda * l1_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pheno = batch.get("pheno") if self.use_phenotypic else None
        ppi = batch.get("ppi") if self.use_ppi else None
        labels = batch["label"].float()

        logits, embeddings = self(pheno, ppi, return_embeddings=True)
        probs = torch.sigmoid(logits)

        self.val_preds.append(probs.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        loss = self.criterion(logits, labels)

        self.val_auroc.update(probs, labels.long())
        self.val_pr_auc.update(probs, labels.long())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mean_prob", probs.mean(), on_epoch=True)

        if "alpha" in embeddings:
            self.alpha_weights.append(embeddings["alpha"].detach().cpu())
        if "beta" in embeddings:
            self.beta_weights.append(embeddings["beta"].detach().cpu())

        if "pheno_emb_unscaled" in embeddings and "ppi_emb_unscaled" in embeddings:
            self.pheno_unscaled_l2s.append(torch.norm(embeddings["pheno_emb_unscaled"], p=2, dim=1).detach().cpu())
            self.ppi_unscaled_l2s.append(torch.norm(embeddings["ppi_emb_unscaled"], p=2, dim=1).detach().cpu())
            self.pheno_scaled_l2s.append(torch.norm(embeddings["pheno_emb"], p=2, dim=1).detach().cpu())
            self.ppi_scaled_l2s.append(torch.norm(embeddings["ppi_emb"], p=2, dim=1).detach().cpu())
            
            self.val_pheno_unscaled_embs.append(embeddings["pheno_emb_unscaled"].detach().cpu())
            self.val_ppi_unscaled_embs.append(embeddings["ppi_emb_unscaled"].detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        # Compute epoch-level metrics
        auroc = self.val_auroc.compute()
        pr_auc = self.val_pr_auc.compute()

        # Log to PyTorch Lightning (and thereby WandB)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_pr_auc", pr_auc, prog_bar=True)

        # Reset metric states for the next epoch
        self.val_auroc.reset()
        self.val_pr_auc.reset()

        # -------------------------
        # Top-k metrics
        # -------------------------
        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        
        # Clear buffers early
        self.val_preds.clear()
        self.val_labels.clear()

        num_samples = len(labels)
        if num_samples == 0:
            return

        # Sort by predicted probability (descending)
        sorted_idx = torch.argsort(preds, descending=True)
        sorted_labels = labels[sorted_idx]

        actual_top100 = min(100, num_samples)
        self.log("val_mean_top100_prob", preds[sorted_idx[:actual_top100]].mean())

        total_positives = labels.sum().item()
        baseline_rate = total_positives / num_samples

        ks = [50, 100, 200]

        for k in ks:
            actual_k = min(k, num_samples)
            topk_labels = sorted_labels[:actual_k]
            num_pos_topk = topk_labels.sum().item()

            recall_k = num_pos_topk / total_positives if total_positives > 0 else 0.0
            precision_k = num_pos_topk / actual_k if actual_k > 0 else 0.0
            enrichment_k = precision_k / baseline_rate if baseline_rate > 0 else 0.0

            self.log(f"val_recall@{k}", recall_k, prog_bar=True)
            self.log(f"val_precision@{k}", precision_k)
            self.log(f"val_enrichment@{k}", enrichment_k)
        
        if self.alpha_weights:
            alpha = torch.cat(self.alpha_weights).squeeze(1)
            self.log("alpha_mean", alpha.mean())
            self.log("alpha_std", alpha.std())
            self.alpha_weights.clear()

        if self.beta_weights:
            beta = torch.cat(self.beta_weights).squeeze(1)
            self.log("beta_mean", beta.mean())
            self.log("beta_std", beta.std())
            self.beta_weights.clear()

        if self.pheno_unscaled_l2s:
            pheno_unscaled_l2 = torch.cat(self.pheno_unscaled_l2s)
            ppi_unscaled_l2 = torch.cat(self.ppi_unscaled_l2s)
            pheno_scaled_l2 = torch.cat(self.pheno_scaled_l2s)
            ppi_scaled_l2 = torch.cat(self.ppi_scaled_l2s)

            pheno_scaled_mean = pheno_scaled_l2.mean()
            ppi_scaled_mean = ppi_scaled_l2.mean()

            self.log("pheno_unscaled_l2_mean", pheno_unscaled_l2.mean())
            self.log("pheno_unscaled_l2_std", pheno_unscaled_l2.std())
            self.log("ppi_unscaled_l2_mean", ppi_unscaled_l2.mean())
            self.log("ppi_unscaled_l2_std", ppi_unscaled_l2.std())

            self.log("pheno_scaled_l2_mean", pheno_scaled_mean)
            self.log("pheno_scaled_l2_std", pheno_scaled_l2.std())
            self.log("ppi_scaled_l2_mean", ppi_scaled_mean)
            self.log("ppi_scaled_l2_std", ppi_scaled_l2.std())

            if pheno_scaled_mean > 0:
                self.log("ppi_to_pheno_scaled_l2_ratio", ppi_scaled_mean / pheno_scaled_mean)

            self.pheno_unscaled_l2s.clear()
            self.ppi_unscaled_l2s.clear()
            self.pheno_scaled_l2s.clear()
            self.ppi_scaled_l2s.clear()

        if self.val_pheno_unscaled_embs:
            pheno_embs = torch.cat(self.val_pheno_unscaled_embs, dim=0)
            self.log("val_pheno_unscaled_std_mean", pheno_embs.std(dim=0).mean())
            self.val_pheno_unscaled_embs.clear()

        if self.val_ppi_unscaled_embs:
            ppi_embs = torch.cat(self.val_ppi_unscaled_embs, dim=0)
            self.log("val_ppi_unscaled_std_mean", ppi_embs.std(dim=0).mean())
            self.val_ppi_unscaled_embs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)