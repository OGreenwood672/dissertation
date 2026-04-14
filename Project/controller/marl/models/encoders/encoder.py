
from datetime import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn

from controller.marl.config import GenerativeLangType
from project_paths import LANGUAGES_DIR

from .vq_vae import VQ_VAE
from .sq_vae import SQ_VAE


class Encoder(nn.Module):

    def __init__(
            self,
            obs_dim, hidden_size, latent_dim, vocab_size,
            commitment_cost,
            rq_levels, kl_weight, 
            init_temperature, min_temperature,
            num_training_steps,
            autoencoder_type
        ):
        
        super().__init__()

        self.autoencoder_type = autoencoder_type

        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.rq_levels = rq_levels
        self.commitment_cost = commitment_cost
        self.kl_weight = kl_weight
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.num_training_steps = num_training_steps
        self.autoencoder_type = autoencoder_type

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_dim),
        )

        if autoencoder_type == GenerativeLangType.VQ_VAE:
            self.latent_handler = VQ_VAE(vocab_size, latent_dim, rq_levels, commitment_cost)
        elif autoencoder_type == GenerativeLangType.SQ_VAE:
            self.latent_handler = SQ_VAE(vocab_size, latent_dim, init_temperature, min_temperature, num_training_steps, rq_levels, kl_weight)
        else:
            raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")

    def forward(self, x):
        return self.latent_handler(self.encoder(x))
    
    @torch.no_grad()
    def get_continuous_latent(self, x):       
        return (self.encoder(x),)

    # def encode(self, x):
    #     return self.latent_handler.encode(self.encoder(x))
    
    # def get_codebooks(self):
    #     return np.array([self.get_embedding(i).weight.detach().cpu().numpy() for i in range(self.rq_levels)])
    
    # def get_embedding(self, index):
    #     return self.latent_handler.get_embedding(index)
    
    # def save_codebook(self, save_folder):
    #     np.save(os.path.join(save_folder, "codebook.npy"), self.get_codebooks())

    # def save_encoder(self, save_folder):
    #     torch.save(self.encoder.state_dict(), os.path.join(save_folder, "encoder.pt"))

    # def save_config(self, save_folder):
    #     config = {
    #         "obs_dim": self.obs_dim,
    #         "hidden_size": self.hidden_size,
    #         "latent_dim": self.latent_dim,
    #         "vocab_size": self.vocab_size,
    #         "rq_levels": self.rq_levels,
    #         "commitment_cost": self.commitment_cost,
    #         "kl_weight": self.kl_weight,
    #         "init_temperature": self.init_temperature,
    #         "min_temperature": self.min_temperature,
    #         "num_training_steps": self.num_training_steps,
    #         "autoencoder_type": self.autoencoder_type,
    #     }
    #     json.dump(config, open(os.path.join(save_folder, "config.json"), "w"))

    # def save(self, save_folder):

    #     self.save_codebook(save_folder)
    #     self.save_encoder(save_folder)
    #     self.save_config(save_folder)

    # @classmethod
    # def load(cls, folder_path, device):

    #     ae_config = json.load(open(folder_path / "config.json", "r"))

    #     encoder = cls(
    #         ae_config["obs_dim"], ae_config["hidden_size"], ae_config["latent_dim"], ae_config["vocab_size"],
    #         ae_config["commitment_cost"], ae_config["rq_levels"], ae_config["kl_weight"],
    #         ae_config["init_temperature"], ae_config["min_temperature"], ae_config["num_training_steps"],
    #         ae_config["autoencoder_type"]
    #     ).to(device)

    #     embeddings = np.load(folder_path / "codebook.npy")
    #     for i, embedding_weights in enumerate(embeddings):
    #         encoder.get_embedding(i).weight.data = torch.tensor(embedding_weights, dtype=torch.float32, device=device)

    #     encoder_path = folder_path / "encoder.pt"
    #     assert os.path.exists(encoder_path)

    #     encoder.encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    #     return encoder


    def encode(self, x):
        return self.latent_handler.encode(self.encoder(x))
    
    def get_all_embeddings(self):
        return list(self.latent_handler.embeddings)
    
    def get_codebooks(self):
        return [emb.weight.detach().cpu().numpy() for emb in self.get_all_embeddings()]
    
    def get_embedding(self, index):
        return self.latent_handler.get_embedding(index)
    
    def save_codebook(self, save_folder):
        codebooks = self.get_codebooks()
        np.savez(os.path.join(save_folder, "codebooks.npz"), *codebooks)

    def save_encoder(self, save_folder):
        torch.save(self.encoder.state_dict(), os.path.join(save_folder, "encoder.pt"))

    def save_config(self, save_folder):
        config = {
            "obs_dim": self.obs_dim,
            "hidden_size": self.hidden_size,
            "latent_dim": self.latent_dim,
            "vocab_size": self.vocab_size,
            "rq_levels": self.rq_levels,
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
            ae_config["commitment_cost"], ae_config["rq_levels"], ae_config["kl_weight"],
            ae_config["init_temperature"], ae_config["min_temperature"], ae_config["num_training_steps"],
            ae_config["autoencoder_type"]
        ).to(device)

        embeddings = np.load(folder_path / "codebooks.npz")
        for i, emb in enumerate(encoder.get_all_embeddings()):
            emb.weight.data = torch.tensor(embeddings[f'arr_{i}'], dtype=torch.float32, device=device)

        encoder_path = folder_path / "encoder.pt"
        assert os.path.exists(encoder_path)

        encoder.encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        return encoder