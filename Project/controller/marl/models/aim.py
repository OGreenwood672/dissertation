
from numpy import save, load
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.encoder import Encoder

class AIM(nn.Module):

    def __init__(
            self, 
            obs_dim,
            hidden_size, latent_dim, vocab_size,
            commitment_cost=0.5,
            num_training_steps=None, hq_vae=1,
            init_temperature=1, min_temperature=0.1,
            autoencoder_type="vq-vae", kl_weight=0.001
        ):
        super().__init__()

        self.encoder = Encoder(
            obs_dim,
            hidden_size, latent_dim, vocab_size,
            commitment_cost=commitment_cost,
            hq_vae=hq_vae, num_training_steps=num_training_steps,
            init_temperature=init_temperature, min_temperature=min_temperature,
            autoencoder_type=autoencoder_type, kl_weight=kl_weight
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_dim)
        )

        self.vocab_size = vocab_size


    def forward(self, x):

        loss, quantised = self.encoder(x)

        reconstructed = self.decoder(quantised)

        loss += F.mse_loss(reconstructed, x) * 0.5

        return loss
    
    def save_encoder(self):
        self.encoder.save()
    
    @staticmethod
    def compute_metrics(quantised_indices, vocab_size):
        entropies = []
        perplexities = []
        
        for i in range(quantised_indices.shape[-1]):

            indices = quantised_indices[..., i]

            counts = torch.bincount(indices.flatten().long(), minlength=vocab_size).float()
            
            probs = counts / (counts.sum() + 1e-10)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(entropy.item())
            
            perplexity = torch.exp(entropy)
            perplexities.append(perplexity.item())
        
        return sum(entropies) / len(entropies), sum(perplexities) / len(perplexities)
