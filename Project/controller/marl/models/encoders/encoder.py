
from datetime import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn

from controller.marl.config import MappoConfig

from .vq_vae import VQ_VAE
from .sq_vae import SQ_VAE


class Encoder(nn.Module):

    def __init__(
            self,
            obs_dim, hidden_size, latent_dim, vocab_size,
            commitment_cost,
            hq_vae, kl_weight, 
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
        self.hq_vae = hq_vae
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

        if autoencoder_type == "vq-vae":
            self.latent_handler = VQ_VAE(vocab_size, latent_dim, hq_vae, commitment_cost)
        elif autoencoder_type == "sq-vae":
            self.latent_handler = SQ_VAE(vocab_size, latent_dim, init_temperature, min_temperature, num_training_steps, hq_vae)
        else:
            raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")

    def forward(self, x):
        return self.latent_handler(self.encoder(x))
    
    def get_codebooks(self):
        return np.array([self.get_embedding(i).weight.detach().cpu().numpy() for i in range(self.hq_vae)])
    
    def get_embedding(self, index):
        return self.latent_handler.embeddings[index]
    
    def save_codebook(self, save_folder):
        np.save(os.path.join(save_folder, "codebook.npy"), self.get_codebooks())

    def save_encoder(self, save_folder):
        torch.save(self.encoder.state_dict(), os.path.join(save_folder, "encoder.pt"))

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

    def save(self):

        result_folder = f"./results/languages/"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        new_folder = os.path.join(result_folder, self.get_folder_name())
        os.makedirs(new_folder)

        self.save_codebook(new_folder)
        self.save_encoder(new_folder)
        self.save_config(new_folder)


    @staticmethod
    def get_folder_name():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(timestamp)


    @classmethod
    def load(cls, config: MappoConfig, is_training=False):

        result_path = "./results/languages/"

        for folder in os.listdir(result_path)[::-1]:
            check_config = json.load(open(os.path.join(result_path, folder, "config.json"), "r"))

            for key, value in config:
                if key in check_config and check_config[key] != value:
                    break
            else:
                folder_path = os.path.join(result_path, folder)
                found = True
                break

        if not found:
            raise ValueError(f"No matching folder found")

        print(f"Loading languag from folder: {folder_path}")
        
        ae_config = json.load(open(os.path.join(folder_path, "config.json"), "r"))

        encoder = cls(
            ae_config["obs_dim"], ae_config["hidden_size"], ae_config["latent_dim"], ae_config["vocab_size"],
            ae_config["commitment_cost"], ae_config["hq_vae"], ae_config["kl_weight"],
            ae_config["init_temperature"], ae_config["min_temperature"], ae_config["num_training_steps"],
            ae_config["autoencoder_type"]
        )
        
        embeddings = np.load(folder_path + "/codebook.npy")
        for i, embedding_weights in enumerate(embeddings):
            encoder.get_embedding(i).weight.data = torch.tensor(embedding_weights, dtype=torch.float32)

        encoder_path = folder_path + "/encoder.pt"
        assert os.path.exists(encoder_path)

        encoder.encoder.load_state_dict(torch.load(encoder_path))

        return encoder
