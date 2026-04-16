

import torch
import torch.nn.functional as F
import torch.nn as nn

from controller.marl.logger import MetricTracker



class VQ_VAE(nn.Module):
    
    def __init__(self, vocab_size, latent_dim, rq_levels=1, commitment_cost=0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.rq_levels = rq_levels
        self.commitment_cost = commitment_cost
                
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, latent_dim) for _ in range(rq_levels)
        ])
        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.5, 0.5)

    def forward(self, latent, tracker: MetricTracker = None):
        loss, quantised = self.quantise(latent, tracker)
        return loss, quantised

    @torch.no_grad()
    def encode(self, latent):
        _, quantised = self.quantise(latent)
        return quantised


    def quantise(self, latent, tracker: MetricTracker = None):
        
        loss = torch.tensor(0.0, device=latent.device)

        curr_residual = latent
        code_sum = torch.zeros_like(latent)
        
        for i in range(self.rq_levels):
            
            embedding = self.embeddings[i]

            distances = torch.cdist(curr_residual, embedding.weight)

            min_distance_index = distances.argmin(-1)

            quantised = embedding(min_distance_index)

            codebook_loss = F.mse_loss(quantised, curr_residual.detach())
            commitment_loss = F.mse_loss(quantised.detach(), curr_residual)

            if tracker is not None:
                tracker.update(f"commitment_loss_{i}", commitment_loss.item())

            loss += codebook_loss + self.commitment_cost * commitment_loss

            code_sum = code_sum + quantised
            curr_residual = curr_residual - quantised
        
        quantised = latent + (code_sum - latent).detach()

        # if self.training:
        return loss, quantised
                
        # return quantised.detach()

    def get_embedding(self, rq_level):
        return self.embeddings[rq_level]