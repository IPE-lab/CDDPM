from typing import Optional
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from model.EpisonTheta import Epison
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

class Diffusion(pl.LightningModule):
    def __init__(self, params, traindataset = None, val_dataset = None):
        super().__init__()
        

        self.traindataset = traindataset
        self.val_dataset = val_dataset

        beta_schedule = params['beta_schedule']
        beta_end = params['beta_end']
        self.diff_steps = params['diff_steps']

        if beta_schedule == 'linear':
            betas = np.linspace(1e-4, beta_end, self.diff_steps)
        elif beta_schedule == "quad":
            betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, self.diff_steps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = torch.from_numpy(1 - betas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("betas", torch.from_numpy(betas), persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas), persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod), persistent=False)


        self.target_dim = params['target_dim']
        self.emb_feature_dim = 16
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        self.is_unconditional = False
        
        params['episonparams']['side_dim'] = self.emb_feature_dim + 128 + 1
        params['episonparams']['num_steps'] = self.diff_steps
        self.denoisemodel = Epison(params['episonparams'])

        self.target_mask_type = params['target_mask_type']
    def get_randmask(self, observed_mask):
        if self.target_mask_type == "random":
            rand_for_mask = torch.rand_like(observed_mask) * observed_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        elif self.target_mask_type == "interval":
            B, K, L = observed_mask.shape
            cond_mask = torch.ones_like(observed_mask)
            for b in range(B):
                num_selected_feat = np.random.randint(0, K, 1)
                selected_feats = np.random.randint(0, K, num_selected_feat)
                for selected_feat in selected_feats:
                    time_steps = np.random.randint(0, L - 1, 2)
                    time_steps.sort()
                    cond_mask[b, selected_feat, time_steps[0] : time_steps[1]] = 0
            cond_mask = torch.where(cond_mask > observed_mask, observed_mask, cond_mask)
        return cond_mask

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_side_info(self, depth_info, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(depth_info)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def loss_fn(self, observed_log, cond_mask, observed_mask, side_info):
        B, K, L = observed_log.shape
        t = torch.randint(0, self.diff_steps, [B]).to(self.device)
        current_alpha = self.alphas_cumprod[t].unsqueeze(1).unsqueeze(1)
        noise = torch.randn_like(observed_log)
        noisy_data = (current_alpha ** 0.5) * observed_log + (1.0 - current_alpha) ** 0.5 * noise
        
        total_input = self.set_input_to_diffmodel(noisy_data, observed_log, cond_mask)
        
        total_input = total_input.to(torch.float)
        side_info = side_info.to(torch.float)
        predicted = self.denoisemodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input
    
    def process_bath(self, batch):
        observed_log = batch['observed_log'].permute(0, 2, 1).to(torch.float)
        observed_mask = batch['observed_mask'].permute(0, 2, 1).to(torch.float)
        depth_info = batch['depth_info'].to(torch.float)
        cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(depth_info, cond_mask)
        return observed_log, cond_mask, observed_mask, side_info

    def training_step(self, batch, batch_idx):
        observed_log, cond_mask, observed_mask, side_info = self.process_bath(batch)
        loss = self.loss_fn(observed_log, cond_mask, observed_mask, side_info)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        observed_log, cond_mask, observed_mask, side_info = self.process_bath(batch)
        loss = self.loss_fn(observed_log, cond_mask, observed_mask, side_info)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.traindataset, batch_size=12, shuffle=True, num_workers=12)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=12, shuffle=False, num_workers=12)
    
    @torch.no_grad()
    def impute(self, observed_log, observed_mask, side_info, samples):
        B, K, L = observed_log.shape
        observed_log = observed_log.unsqueeze(1).expand(samples, 1, -1, -1)
        observed_mask = observed_mask.unsqueeze(1).expand(samples, 1, -1, -1)
        side_info = side_info.expand(samples, -1, -1, -1)
        imputed_log = torch.zeros(B, samples, K, L)
    
        current_sample = torch.randn_like(observed_log)
        for t in tqdm(range(self.diff_steps - 1, -1, -1)):
            masked_observed_log = (observed_log * observed_mask)
            noisy_target = ((1 - observed_mask) * current_sample)
            diff_input = torch.cat([masked_observed_log, noisy_target], dim=1)
            timestep = torch.tensor([t]).to(self.device).expand(samples)
            predicted = self.denoisemodel(diff_input, side_info, timestep).unsqueeze(1)

            coef1 = 1 / self.alphas[t] ** 0.5
            coef2 = (1 - self.alphas[t]) / (1 - self.alphas_cumprod[t]) ** 0.5
            
            current_sample = coef1 * (current_sample - coef2 * predicted)
            if t > 0:
                noise = torch.randn_like(current_sample)
                sigma = (
                    (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - self.alphas_cumprod[t]) * self.betas[t]
                    ) ** 0.5
                current_sample += noise * sigma
            
        imputed_log = current_sample * (1 - observed_mask) + observed_log * observed_mask
        return imputed_log

    def imputation(self, observed_log, observed_mask, deep_info, samples):
        observed_mask = observed_mask.permute(0, 2, 1).to(torch.float)
        observed_log = observed_log.permute(0, 2, 1).to(torch.float)
        side_info = self.get_side_info(deep_info, observed_mask)
        imputed_log = self.impute(observed_log, observed_mask, side_info, samples)
        return imputed_log.permute(0, 1, 3, 2)

        