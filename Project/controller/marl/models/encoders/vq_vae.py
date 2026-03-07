

import torch
import torch.nn.functional as F
import torch.nn as nn



class VQ_VAE(nn.Module):
    
    def __init__(self, vocab_size, latent_dim, hq_vae=1, commitment_cost=0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hq_vae = hq_vae
        self.commitment_cost = commitment_cost
                
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, latent_dim) for _ in range(hq_vae)
        ])
        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.5, 0.5)


    def forward(self, latent):
        
        loss = torch.tensor(0.0, device=latent.device)

        curr_residual = latent
        code_sum = torch.zeros_like(latent)
        
        for i in range(self.hq_vae):
            
            embedding = self.embeddings[i]

            distances = torch.cdist(curr_residual, embedding.weight)

            min_distance_index = distances.argmin(-1)

            quantised = embedding(min_distance_index)

            codebook_loss = F.mse_loss(quantised, curr_residual.detach())
            commitment_loss = F.mse_loss(quantised.detach(), curr_residual)

            loss += codebook_loss + self.commitment_cost * commitment_loss

            code_sum = code_sum + quantised
            curr_residual = curr_residual - quantised
        
        quantised = latent + (code_sum - latent).detach()

        # if self.training:
        return loss, quantised
                
        # return quantised.detach()
