import json
import os

import numpy as np
import torch
import torch.nn as nn
from .hq_vae import HQ_VAE

class HQ_Encoder(nn.Module):
    def __init__(
            self,
            obs_dim, hidden_size, latent_dims, vocab_size,
            commitment_cost,
            hq_vae_levels, kl_weight,
            init_temperature, min_temperature,
            num_training_steps
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.latent_dims = latent_dims

        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.top_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_dims[0])
        )

        self.bottom_head = nn.Sequential(
            nn.Linear(hidden_size + latent_dims[0], hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_dims[1])
        )

        self.latent_handler = HQ_VAE(
            vocab_size=vocab_size, 
            latent_dims=latent_dims,
            init_temperature=init_temperature, 
            min_temperature=min_temperature,
            num_training_steps=num_training_steps, 
            hq_vae_levels=hq_vae_levels, 
            kl_weight=kl_weight
        )

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        
        top_latent = self.top_head(shared_features)
        
        bottom_input = torch.cat([shared_features, top_latent], dim=-1)
        bottom_latent = self.bottom_head(bottom_input)
        
        return self.latent_handler(top_latent, bottom_latent)

    def encode(self, x):
        shared_features = self.shared_encoder(x)
        top_latent = self.top_head(shared_features)
        
        bottom_input = torch.cat([shared_features, top_latent], dim=-1)
        bottom_latent = self.bottom_head(bottom_input)
        
        return self.latent_handler.encode(top_latent, bottom_latent)
        
    def get_all_embeddings(self):
        top_embs = list(self.latent_handler.top_quantizer.embeddings)
        bottom_embs = list(self.latent_handler.bottom_quantizer.embeddings)
        return top_embs + bottom_embs
        
    def get_codebooks(self):
        return np.array([emb.weight.detach().cpu().numpy() for emb in self.get_all_embeddings()])
    
    def save_codebook(self, save_folder):
        np.save(os.path.join(save_folder, "codebook.npy"), self.get_codebooks())

    def save_encoder(self, save_folder):
        to_save = {
            "top_head": self.top_head.state_dict(),
            "bottom_head": self.bottom_head.state_dict(),
            "shared_encoder": self.shared_encoder.state_dict()
        }
        torch.save(to_save, os.path.join(save_folder, "encoders.pt"))

    def save_config(self, save_folder):
        config = {
            "obs_dim": self.obs_dim,
            "hidden_size": self.hidden_size,
            "latent_dim": self.latent_dim,
            "vocab_size": self.vocab_size,
            "hq_vae": self.hq_vae,
            "commitment_cost": self.commitment_cost,
            "kl_weight": self.kl_weight,
            "init_temperature": self.init_temperature,
            "min_temperature": self.min_temperature,
            "num_training_steps": self.num_training_steps,
            "autoencoder_type": self.autoencoder_type,
        }
        json.dump(config, open(os.path.join(save_folder, "config.json"), "w"))


    def save(self, save_folder):

        self.save_codebook(save_folder)
        self.save_encoder(save_folder)
        self.save_config(save_folder)

    @classmethod
    def load(cls, folder_path, device):

        ae_config = json.load(open(folder_path / "config.json", "r"))

        encoder = cls(
            ae_config["obs_dim"], ae_config["hidden_size"], ae_config["latent_dim"], ae_config["vocab_size"],
            ae_config["commitment_cost"], ae_config["hq_vae"], ae_config["kl_weight"],
            ae_config["init_temperature"], ae_config["min_temperature"], ae_config["num_training_steps"],
            ae_config["autoencoder_type"]
        ).to(device)

        embeddings = np.load(folder_path / "codebook.npy")
        for i, embedding_weights in enumerate(embeddings):
            encoder.get_embedding(i).weight.data = torch.tensor(embedding_weights, dtype=torch.float32, device=device)

        encoder_path = folder_path / "encoders.pt"
        assert os.path.exists(encoder_path)

        encoders = torch.load(encoder_path, map_location=device)

        encoder.top_head.load_state_dict(encoders["top_head"])
        encoder.bottom_head.load_state_dict(encoders["bottom_head"])
        encoder.shared_encoder.load_state_dict(encoders["shared_encoder"])

        return encoder
