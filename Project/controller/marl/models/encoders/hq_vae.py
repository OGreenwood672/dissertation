import torch
import torch.nn as nn

from .sq_vae import SQ_VAE 

class HQ_VAE(nn.Module):
    def __init__(
            self,
            vocab_size, latent_dims,
            init_temperature=1.0, min_temperature=0.1,
            num_training_steps=50000, 
            rq_levels=[1, 1],
            kl_weight=0.001
        ):
        super().__init__()

        self.top_quantiser = SQ_VAE(
            vocab_size=vocab_size, 
            latent_dim=latent_dims[0], 
            init_temperature=init_temperature, min_temperature=min_temperature,
            num_training_steps=num_training_steps, rq_levels=rq_levels[0], kl_weight=kl_weight
        )
        
        self.bottom_quantiser = SQ_VAE(
            vocab_size=vocab_size, 
            latent_dim=latent_dims[1], 
            init_temperature=init_temperature, min_temperature=min_temperature,
            num_training_steps=num_training_steps, rq_levels=rq_levels[1], kl_weight=kl_weight
        )

    def forward(self, top_latent, bottom_latent):
        
        top_loss, top_quantised = self.top_quantiser(top_latent)
        bottom_loss, bottom_quantised = self.bottom_quantiser(bottom_latent)
        
        return top_loss + bottom_loss, torch.cat([top_quantised, bottom_quantised], dim=-1)

    @torch.no_grad()
    def encode(self, top_latent, bottom_latent):
        top_quantised = self.top_quantiser.encode(top_latent)
        bottom_quantised = self.bottom_quantiser.encode(bottom_latent)
        return torch.cat([top_quantised, bottom_quantised], dim=-1)
