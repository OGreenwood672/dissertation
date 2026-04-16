
from datetime import datetime
import json

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.marl.config import AIMConfig, CommConfig, GenerativeLangType
from controller.marl.models.encoders.hq_encoder import HQ_Encoder
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
            obs_dim,
            comm_config: CommConfig, aim_config: AIMConfig,
            obs_mask=None,
            num_training_steps=None
        ):
        super().__init__()

        if comm_config.autoencoder_type != GenerativeLangType.HQ_VAE:
            self.encoder = Encoder(
                obs_dim,
                aim_config.hidden_size, comm_config.communication_size, comm_config.vocab_size,
                commitment_cost=aim_config.commitment_cost,
                rq_levels=comm_config.rq_levels, num_training_steps=num_training_steps if num_training_steps is not None else aim_config.ae_epochs,
                init_temperature=aim_config.init_temperature, min_temperature=aim_config.min_temperature,
                autoencoder_type=comm_config.autoencoder_type, kl_weight=aim_config.kl_weight
            )
        else:
            macro_dim = comm_config.communication_size // 3
            micro_dim = comm_config.communication_size - macro_dim
            self.encoder = HQ_Encoder(
                obs_dim,
                aim_config.hidden_size, 
                [macro_dim, micro_dim],
                comm_config.vocab_size,
                commitment_cost=aim_config.commitment_cost,
                rq_levels=[comm_config.rq_levels, comm_config.rq_levels],
                kl_weight=aim_config.kl_weight,
                init_temperature=aim_config.init_temperature,
                min_temperature=aim_config.min_temperature,
                num_training_steps=aim_config.ae_epochs
            )

        self.decoder = nn.Sequential(
            nn.Linear(comm_config.communication_size, aim_config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(aim_config.hidden_size // 2, aim_config.hidden_size),
            nn.ReLU(),
            nn.Linear(aim_config.hidden_size, obs_dim)
        )

        self.aim_config = aim_config
        self.comm_config = comm_config
        self.obs_mask = obs_mask



    def forward(self, x, tracker=None):

        encoder_loss, quantised = self.encoder(x, tracker)

        reconstructed = self.decoder(quantised)

        predictions = torch.sigmoid(reconstructed)

        if self.obs_mask is not None:
            bce_loss = F.binary_cross_entropy(predictions[:, self.obs_mask], x[:, self.obs_mask])
            mse_loss = F.mse_loss(predictions[:, ~self.obs_mask], x[:, ~self.obs_mask])
            reconstruction_loss = bce_loss + mse_loss * 1.75
            loss = encoder_loss + reconstruction_loss
        else:
            reconstruction_loss = F.mse_loss(predictions, x)
            loss = encoder_loss + reconstruction_loss

        if tracker is not None:
            tracker.update("reconstruction_loss", reconstruction_loss.item())

        # MSE
        # loss += F.mse_loss(reconstructed, x) * 0.5

        # BCE - can't do when negative obs exist
        # loss = loss + F.binary_cross_entropy(torch.sigmoid(reconstructed), x)
        # Look into Sliced Composite Loss
        
        return loss
    
    def get_individual_losses(self, x):

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
    def load(cls, saved_folder, aim_config: AIMConfig, comm_config: CommConfig, obs_dim: int, obs_loss_mask: torch.Tensor, device: torch.device):

        aim = cls(obs_dim, comm_config, aim_config, obs_loss_mask)

        if comm_config.autoencoder_type == GenerativeLangType.HQ_VAE:
            aim.encoder = HQ_Encoder.load(saved_folder, device)
        else:
            aim.encoder = Encoder.load(saved_folder, device)
            
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
                    if key == "rq_levels" and config.autoencoder_type == GenerativeLangType.HQ_VAE: value = [value, value]
                    if key == "autoencoder_type": value = value.value
                    if key in check_config and check_config[key] != value:
                        print(key, check_config[key], value)
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
