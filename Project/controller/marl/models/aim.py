
from datetime import datetime
import json

from numpy import save, load
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.marl.config import MappoConfig
from project_paths import LANGUAGES_DIR

from .encoders.encoder import Encoder


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class AIM(nn.Module):

    def __init__(
            self, 
            obs_dim, action_count,
            hidden_size, latent_dim, vocab_size,
            commitment_cost=0.5,
            num_training_steps=None, hq_vae=1,
            init_temperature=1, min_temperature=0.1,
            autoencoder_type="vq-vae", kl_weight=0.001,
            obs_mask=None
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

        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.obs_mask = obs_mask



    def forward(self, x, target_reward=None):

        loss, quantised = self.encoder(x)

        reconstructed = self.decoder(quantised)

        predictions = torch.sigmoid(reconstructed)

        if self.obs_mask is not None:
            bce_loss = F.binary_cross_entropy(predictions[:, self.obs_mask], x[:, self.obs_mask])
            mse_loss = F.mse_loss(predictions[:, ~self.obs_mask], x[:, ~self.obs_mask])
            loss = loss + bce_loss + mse_loss * 1.75
        else:
            loss = loss + F.mse_loss(predictions, x)

        # MSE
        # loss += F.mse_loss(reconstructed, x) * 0.5

        # BCE - can't do when negative obs exist
        # loss = loss + F.binary_cross_entropy(torch.sigmoid(reconstructed), x)
        # Look into Sliced Composite Loss
        
        if target_reward is not None: # inverse dynamics model
            # quantised_protected = ScaleGrad.apply(quantised, 0.05)
            reward = self.reward_head(quantised)
            reward_loss = F.mse_loss(reward, target_reward)
            loss = loss + reward_loss * 0.1

        return loss
    
    def get_individual_losses(self, x, target_reward=None):

        loss, quantised = self.encoder(x)

        reconstructed = self.decoder(quantised)

        predictions = torch.sigmoid(reconstructed)

        node_losses = torch.zeros(x.size(1), device=x.device)

        if self.obs_mask is not None:
            bce_loss = F.binary_cross_entropy(predictions[:, self.obs_mask], x[:, self.obs_mask], reduction='none')
            mse_loss = F.mse_loss(predictions[:, ~self.obs_mask], x[:, ~self.obs_mask], reduction='none')
            node_losses[self.obs_mask] = bce_loss.mean(dim=0)
            node_losses[~self.obs_mask] = mse_loss.mean(dim=0)
        else:
            loss = loss + F.mse_loss(predictions, x, reduction='none')
            node_losses = loss.mean(dim=0)

        return node_losses

    
    def check(self, x, target_reward=None):

        loss, quantised = self.encoder(x)

        reconstructed = self.decoder(quantised)
        
        predictions = torch.sigmoid(reconstructed)

        if self.obs_mask is not None:
            bce_loss = F.binary_cross_entropy(predictions[:, self.obs_mask], x[:, self.obs_mask])
            mse_loss = F.mse_loss(predictions[:, ~self.obs_mask], x[:, ~self.obs_mask])
            loss = loss + bce_loss + mse_loss
        else:
            loss = loss + F.mse_loss(predictions, x)

        # MSE
        # loss += F.mse_loss(reconstructed, x) * 0.5

        # BCE - can't do when negative obs exist
        # loss = loss + F.binary_cross_entropy(torch.sigmoid(reconstructed), x)
        # Look into Sliced Composite Loss
        
        # if self.training and target_reward is not None: # inverse dynamics model
        if target_reward is not None:
            # quantised_protected = ScaleGrad.apply(quantised, 0.05)
            reward = self.reward_head(quantised)
            reward_loss = F.mse_loss(reward, target_reward)

        return loss, reward_loss, reconstructed, quantised
    
    def save(self):
        if not os.path.exists(LANGUAGES_DIR):
            os.makedirs(LANGUAGES_DIR)

        new_folder = os.path.join(LANGUAGES_DIR, self.get_folder_name())
        os.makedirs(new_folder)

        self.save_encoder(new_folder)
        self.save_decoder(new_folder)

    def save_encoder(self, save_folder):
        self.encoder.save(save_folder)

    def save_decoder(self, save_folder):
        torch.save(self.decoder.state_dict(), os.path.join(save_folder, "decoder.pt"))

    @classmethod
    def load(cls, saved_folder, config: MappoConfig, obs_mask=None, obs_dim=0, action_count=5):

        aim = cls(
            obs_dim, action_count,
            512, config.communication_size * config.num_comms, config.vocab_size,
            commitment_cost=config.commitment_cost,
            hq_vae=config.hq_layers, num_training_steps=config.ae_epochs,
            init_temperature=config.init_temperature, min_temperature=config.min_temperature,
            autoencoder_type=config.autoencoder_type, kl_weight=config.kl_weight, obs_mask=obs_mask
        )

        aim.encoder = Encoder.load(saved_folder)
        aim.decoder.load_state_dict(torch.load(saved_folder / "decoder.pt"))

        return aim


    @staticmethod
    def get_folder_name():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(timestamp)
    
    @staticmethod
    def get_folder(config):
        folder_path = None
        for folder in os.listdir(LANGUAGES_DIR)[::-1]:
            with open(LANGUAGES_DIR / folder / "config.json", "r") as json_file:
                check_config = json.load(json_file)

                for key, value in config:
                    if key in check_config and check_config[key] != value:
                        break
                else:
                    folder_path = LANGUAGES_DIR / folder
                    break

        if folder_path is None:
            raise ValueError(f"No matching folder found")

        print(f"Loading language from folder: {folder_path}")

        return folder_path

    
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
