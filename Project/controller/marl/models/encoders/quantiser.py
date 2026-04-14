

import torch
import torch.nn.functional as F
import torch.nn as nn

class Quantiser(nn.Module):

    def __init__(self, vocab_size, comm_dim, commitment_cost=0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.comm_dim = comm_dim
        self.commitment_cost = commitment_cost

        # The codebook
        self.embedding = nn.Embedding(vocab_size, comm_dim)

        # codebook initialisation
        self.embedding.weight.data.uniform_(-0.5, 0.5)

        self.register_buffer('usage_count', torch.zeros(vocab_size))

    def forward(self, x):
        
        distances = torch.cdist(x, self.embedding.weight, p=2.0)

        min_distance_index = distances.argmin(-1)

        # Dead Code Revival
        if self.training:
            with torch.no_grad():
                unique, counts = torch.unique(min_distance_index, return_counts=True)
                self.usage_count[unique] += counts.float()

                dead_codes = torch.where(self.usage_count < 1.01)[0]
                if len(dead_codes) > 0:
                    random_inputs = x[torch.randint(0, x.size(0), (len(dead_codes),), device=x.device)]
                    self.embedding.weight.data[dead_codes] = random_inputs.detach()
                    self.usage_count[dead_codes] = 10.0
                
                self.usage_count *= 0.99

        quantised = self.embedding(min_distance_index)

        codebook_loss = F.mse_loss(quantised, x.detach())
        e_latent_loss = F.mse_loss(quantised.detach(), x)

        loss = codebook_loss + self.commitment_cost * e_latent_loss

        # to allow backprop, effectivly x + constant = quantised
        quantised = x + (quantised - x).detach()

        return quantised, loss, min_distance_index
    
    def compute_metrics(self, quantised_indices):
        
        counts = torch.bincount(quantised_indices.flatten(), minlength=self.vocab_size).float()
        
        probs = counts / counts.sum()
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        perplexity = torch.exp(entropy)
        
        return entropy, perplexity
    
    @staticmethod
    def get_active_codebook_usage(indices):
        return len(torch.unique(indices.flatten()))
