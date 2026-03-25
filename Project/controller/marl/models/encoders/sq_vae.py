

from torch import nn
import torch
import torch.nn.functional as F
import math


class SQ_VAE(nn.Module):
    def __init__(
            self,
            vocab_size, latent_dim,
            init_temperature=1.0, min_temperature=0.1,
            num_training_steps=50000, hq_vae=1, kl_weight=0.001
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hq_vae = hq_vae
        self.kl_weight = kl_weight
        self.min_temp = min_temperature

        completion_percentages = []
        if hq_vae == 1:
            completion_percentages = [0.75]
        else:
            for i in range(hq_vae):
                completion_percentages.append(0.4 + (0.4 * (i / (hq_vae - 1))))

        print(f"Codebook(s) will freeze at {', '.join([f'{p*100:.2g}%' for p in completion_percentages])} of training steps.")

        self.embeddings = nn.ModuleList()
        self.logit_scales = nn.ParameterList()
        
        self.register_buffer('temperatures', torch.full((hq_vae,), init_temperature))
        self.register_buffer('anneal_rates', torch.zeros(hq_vae))

        for i in range(hq_vae):
            rate = math.exp(math.log(min_temperature / init_temperature) / (num_training_steps * completion_percentages[i]))
            self.anneal_rates[i] = rate
            
            self.embeddings.append(nn.Embedding(vocab_size, latent_dim))            
            self.logit_scales.append(nn.Parameter(torch.tensor(10.0)))

    def forward(self, latent):
        loss, quantised = self.quantise(latent)
        return loss, quantised

    @torch.no_grad()
    def encode(self, latent):
        _, quantised = self.quantise(latent)
        return quantised

    def quantise(self, latent):
        loss = torch.tensor(0.0, device=latent.device)

        curr_residual = latent
        code_sum = torch.zeros_like(latent)
        
        for hq_level, (embedding, logit_scale) in enumerate(zip(self.embeddings, self.logit_scales)):

            if hq_level > 0:
                scale = curr_residual.std(dim=-1, keepdim=True) + 1e-5
                scaled_input = curr_residual / scale
            else:
                scaled_input = curr_residual
                scale = torch.tensor(1.0, dtype=curr_residual.dtype, device=curr_residual.device)

            flat_input = scaled_input.view(-1, self.latent_dim)

            x_norm = F.normalize(flat_input, p=2.0, dim=-1)

            bounded_weight = torch.sigmoid(embedding.weight) 
            embed_norm = F.normalize(bounded_weight , p=2.0, dim=-1)

            logits = torch.matmul(x_norm, embed_norm.t()) * logit_scale
                        
            temp = self.temperatures[hq_level].item()
            
            if self.training:
                soft_one_hot = F.gumbel_softmax(logits, tau=temp, hard=True)

                with torch.no_grad():
                    self.temperatures[hq_level] = max(
                        self.min_temp, 
                        temp * self.anneal_rates[hq_level].item()
                    )
            else:
                indices = logits.argmax(dim=-1)
                soft_one_hot = F.one_hot(indices, self.vocab_size).float()
                
            quantised_scaled = (soft_one_hot @ bounded_weight).view_as(scaled_input)

            probs = F.softmax(logits / temp, dim=-1)
            log_probs = F.log_softmax(logits / temp, dim=-1)
            uniform_prior = torch.tensor(1.0 / self.vocab_size, device=latent.device)
            
            kl_divergence = torch.sum(probs * (log_probs - torch.log(uniform_prior)), dim=-1).mean()
            
            if self.training:
                loss += self.kl_weight * kl_divergence

            if hq_level > 0:
                quantised = quantised_scaled * scale
            else:
                quantised = quantised_scaled

            code_sum = code_sum + quantised
            curr_residual = curr_residual - quantised
        
        # quantised = latent + (code_sum - latent).detach()
        quantised = code_sum
        
        # if self.training:
        return loss, quantised
            
        # return quantised.detach()