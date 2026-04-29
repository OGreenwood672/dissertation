from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.marl.core.config import ActorHyperparameters, CommConfig, GenerativeLangType
from controller.marl.models.aim import AIM
from controller.marl.models.encoders.encoder import Encoder
from controller.marl.models.encoders.hq_encoder import HQ_Encoder

from project_paths import LANGUAGES_DIR

from .comms import Comms



class AimComms(Comms, nn.Module):

    def __init__(self, config: CommConfig, actor_config: ActorHyperparameters, log: Callable[[str, float], None], device: torch.device, **kwargs):
        nn.Module.__init__(self)

        self.device = device
        self.to(device)

        self.config = config
        self.actor_config = actor_config
        self.num_agents = kwargs["num_agents"]

        self.log = log
        if config.comm_folder:
            language_folder = LANGUAGES_DIR / config.comm_folder
        else:
            language_folder = AIM.get_folder(config)

        if config.autoencoder_type == GenerativeLangType.HQ_VAE:
            self.encoder = HQ_Encoder.load(language_folder, 10001, device=device)
        else:
            self.encoder = Encoder.load(language_folder, device=device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # self.aim_codebook = torch.tensor(self.encoder.get_codebooks(), dtype=torch.float32, device=device)
        self.aim_codebooks = [
            torch.tensor(cb, dtype=torch.float32, device=device) 
            for cb in self.encoder.get_codebooks()
        ]
        # self.register_buffer('aim_codebook', aim_codebook)

        # Teaching the fixed codebook
        # Sender must understand what its sending
        # Predict other agent rewards
        self.predict_agents_returns = nn.Linear(config.communication_size * config.num_comms + actor_config.lstm_hidden_size, self.num_agents)
        
        # Reciever must understand how it is getting effected
        # Predict critic value
        self.predict_critic_value = nn.Linear(config.communication_size * config.num_comms + actor_config.lstm_hidden_size, 1)

        # self.comm_heads = nn.ModuleList([
        #     nn.Linear(actor_config.lstm_hidden_size + config.communication_size * hq_level * config.num_comms, config.num_comms * config.vocab_size)
        #     for hq_level in range(self.config.rq_levels)
        # ])

        comm_heads = []
        accumulated_dims = 0
        for hq_level in range(len(self.aim_codebooks)):
            in_features = actor_config.lstm_hidden_size + (accumulated_dims * config.num_comms)
            comm_heads.append(nn.Linear(in_features, config.num_comms * config.vocab_size))
            accumulated_dims += self.aim_codebooks[hq_level].shape[-1]
            
        self.comm_heads = nn.ModuleList(comm_heads)

        # Action Predictors
        # self.predict_intent_sender = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)
        # self.predict_intent_receiver = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)

        # Target Predictiors
        self.predict_intent_sender_goal = nn.Linear(config.communication_size * config.num_comms + actor_config.lstm_hidden_size, 2)
        self.predict_intent_receiver_goal = nn.Linear(config.communication_size * config.num_comms + actor_config.lstm_hidden_size, 2)

        self.obs_external_mask = kwargs["obs_external_mask"]

        self.percieve_out_features = (~self.obs_external_mask).sum().item() + config.communication_size * config.num_comms
        self.out_features = config.communication_size * config.num_comms


    def percieve(self, x: torch.Tensor):
        self.encoder.eval()
        internal = x[..., ~self.obs_external_mask]
        external = x[..., self.obs_external_mask]
        return torch.cat([internal, self.encoder.encode(external)], dim=-1)
    
    def get_comm_logits(self, x: torch.Tensor):
        return None
    
    def get_comms_during_rollout(self, x: torch.Tensor):

        B, _, N, _ = x.shape

        comm_log_probs = torch.zeros((B, N), device=self.device)
        
        # codewords = torch.zeros((B, N, self.config.num_comms, self.config.rq_levels, self.config.communication_size), device=self.device)
        quantised = torch.zeros((B, N, self.config.num_comms, len(self.aim_codebooks)), dtype=torch.long, device=self.device)

        level_codewords = []
        comm_conditions = x.squeeze(1)

        for rq_level in range(len(self.aim_codebooks)):

            comm_logits = self.comm_heads[rq_level](comm_conditions).reshape(B, N, self.config.num_comms, self.config.vocab_size)
            probs = F.softmax(comm_logits.float(), dim=-1)

            quantised_index = torch.multinomial(probs.view(-1, self.config.vocab_size), 1).view(B, N, self.config.num_comms)
            
            log_probs_all = F.log_softmax(comm_logits.float(), dim=-1)
            level_log_probs = log_probs_all.gather(dim=-1, index=quantised_index.unsqueeze(-1)).squeeze(-1)
            comm_log_probs += level_log_probs.sum(dim=-1)
            quantised[:, :, :, rq_level] = quantised_index

            cb = self.aim_codebooks[rq_level]
            current_codewords = cb[quantised_index]
            level_codewords.append(current_codewords)

            # codewords[:, :, :, rq_level, :] = self.aim_codebook[rq_level, quantised_index].view(B, N, self.config.num_comms, self.config.communication_size)

            comm_conditions = torch.cat([comm_conditions, current_codewords.reshape(B, N, self.config.num_comms * cb.shape[-1])], dim=-1)

        # comm_output = codewords.sum(dim=-2)
        if self.config.autoencoder_type == GenerativeLangType.HQ_VAE:
            comm_output = torch.cat(level_codewords, dim=-1)
        else:
            comm_output = torch.stack(level_codewords, dim=-2).sum(dim=-2)

        return comm_output, quantised.detach(), comm_log_probs
    
    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):

        comm_conditions = x

        B, T, N, _ = comm_conditions.shape
        NC = self.config.num_comms
        C = self.config.communication_size

        new_comm_log_probs = torch.zeros(B, T, N, device=self.device)
        comm_entropy = torch.zeros(B, T, N, device=self.device)

        # codewords = torch.zeros((B, T, N, NC, self.config.rq_levels, C), device=self.device)
        # soft_codewords = torch.zeros_like(codewords) # For STE
        
        level_codewords = []
        level_soft_codewords = [] # For STE

        # Note: Teacher Forcing - no dist.sample - similar to chatgpt training
        for rq_level in range(len(self.aim_codebooks)):
            comm_logits = self.comm_heads[rq_level](comm_conditions).reshape(B, T, N, NC, self.config.vocab_size)
            
            dist = torch.distributions.Categorical(logits=comm_logits)
            quantised_index = comms[..., rq_level].long()

            code_counts = torch.bincount(quantised_index.flatten(), minlength=self.config.vocab_size).float()
            usage_pct = (code_counts > 0).float().sum()
            usage_std = code_counts.std()
            topk_usage = code_counts.topk(10)[0].sum()
            # assert False # 10 not 5
            self.log(f"usage_count_{rq_level}", usage_pct.item())
            self.log(f"std_usage_{rq_level}", usage_std.item())
            self.log(f"topk_usage_{rq_level}", topk_usage.item())
            
            new_comm_log_probs += dist.log_prob(quantised_index).sum(dim=-1)
            comm_entropy += dist.entropy().sum(dim=-1)

            cb = self.aim_codebooks[rq_level]
            
            current_codewords = cb[quantised_index]
            current_soft_codewords = dist.probs @ cb

            level_codewords.append(current_codewords)
            level_soft_codewords.append(current_soft_codewords)

            comm_conditions = torch.cat([comm_conditions, current_codewords.reshape(B, T, N, NC * cb.shape[-1])], dim=-1)


            # codewords[:, :, :, :, rq_level, :] = self.aim_codebook[rq_level, quantised_index]
            # soft_codewords[:, :, :, :, rq_level, :] = dist.probs @ self.aim_codebook[rq_level]

            # comm_conditions = torch.cat([comm_conditions, codewords[:, :, :, :, rq_level, :].reshape(B, T, N, NC * C)], dim=-1)


        final_codewords = []
        for hard, soft in zip(level_codewords, level_soft_codewords):
            final_codewords.append(soft + (hard - soft).detach())

        if self.config.autoencoder_type == GenerativeLangType.HQ_VAE:
            comm_output = torch.cat(final_codewords, dim=-1).view(-1, NC * self.config.communication_size)
        else:
            comm_output = torch.stack(final_codewords, dim=-2).sum(dim=-2).view(-1, NC * self.config.communication_size)

        # codewords = soft_codewords + (codewords - soft_codewords).detach()
        # comm_output = codewords.sum(dim=-2).view(-1, NC * C)

        return comm_output, new_comm_log_probs, comm_entropy

    
    def return_loss(self, context, true_returns):

        predicted_agents_returns = self.predict_agents_returns(context)

        target_returns = true_returns.view(-1, self.num_agents).repeat_interleave(self.num_agents, dim=0)
        normalized_targets = (target_returns - target_returns.mean()) / (target_returns.std() + 1e-5)

        predicted_returns_loss = F.mse_loss(predicted_agents_returns, normalized_targets)
        return predicted_returns_loss
    


    
    def get_sender_context(self, comm_output, x):
        return torch.cat([comm_output, x.reshape(-1, self.actor_config.lstm_hidden_size)], dim=-1)

    def critic_loss(self, context, new_critic_values):
        predicted_critic_values = self.predict_critic_value(context)
        critic_loss = F.mse_loss(new_critic_values, predicted_critic_values.flatten())
        return critic_loss
    
    def sender_goal_loss(self, context, true_targets, true_target_exists):
        predicted_sender_goal = self.predict_intent_sender_goal(context)
        predicted_sender_goal_loss = F.mse_loss(predicted_sender_goal, true_targets.view(-1, 2), reduction='none').mean(dim=-1)
        predicted_sender_goal_loss = (predicted_sender_goal_loss * true_target_exists.view(-1)).sum() / (true_target_exists.sum() + 1e-8)
        return predicted_sender_goal_loss

    def get_reciever_context(self, x, comms, mask):

        B, T, N, _ = x.shape

        NC = self.config.num_comms
        C = self.config.communication_size
        
        expanded_comms = comms.view(B, T, N, NC * C).unsqueeze(2).expand(B, T, N, N, NC * C)
        other_comms = expanded_comms[mask.expand(B, T, N, N)].reshape(B, T, N, N - 1, NC * C).detach().reshape(-1, NC * C)

        # lstm_expanded = x.unsqueeze(2).expand(B, T, N, N, self.actor_config.lstm_hidden_size)
        # other_lstm = lstm_expanded[mask.expand(B, T, N, N)].reshape(B, T, N, N - 1, self.actor_config.lstm_hidden_size).reshape(-1, self.actor_config.lstm_hidden_size)
        lstm_output = x.unsqueeze(3).expand(B, T, N, N - 1, self.actor_config.lstm_hidden_size).reshape(-1, self.actor_config.lstm_hidden_size)

        return torch.cat([other_comms, lstm_output], dim=-1)

    def receiver_goal_loss(self, context, true_targets, true_target_exists, mask):
        
        B, T, N, _ = true_targets.shape
    
        predicted_reciever_goal = self.predict_intent_receiver_goal(context)

        expanded_goals = true_targets.unsqueeze(2).expand(B, T, N, N, 2)
        other_goals = expanded_goals[mask.expand(B, T, N, N), :].view(-1, 2)

        # expanded_goals = true_targets.unsqueeze(2).expand(B, T, N, N, 2)
        # other_goals = expanded_goals[..., mask, :].reshape(-1, 2)

        expanded_exists = true_target_exists.unsqueeze(2).expand(B, T, N, N)
        other_exists = expanded_exists[..., mask].reshape(-1)

        predicted_reciever_goal_loss = F.mse_loss(predicted_reciever_goal, other_goals, reduction='none').mean(dim=-1)
        predicted_reciever_goal_loss = (predicted_reciever_goal_loss * other_exists).sum() / (other_exists.sum() + 1e-8)

        return predicted_reciever_goal_loss


    def get_loss(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):

        return 0.0

        sender_context = self.get_sender_context(comm_output, x)

        predicted_returns_loss = self.return_loss(sender_context, true_returns)
        # predicted_critic_loss = self.critic_loss(sender_context, new_critic_values)

        true_targets = targets[..., :2]
        true_target_exists = targets[..., 2]

        predicted_sender_goal_loss = self.sender_goal_loss(sender_context, true_targets, true_target_exists)

        mask = ~torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        receiver_context = self.get_reciever_context(x, all_comms, mask)
        predicted_receiver_goal_loss = self.receiver_goal_loss(receiver_context, true_targets, true_target_exists, mask)

        intent_loss = predicted_sender_goal_loss + predicted_receiver_goal_loss

        self.log("predicted_return_loss", predicted_returns_loss.item())
        self.log("predicted_sender_goal_loss", predicted_sender_goal_loss.item())
        self.log("predicted_receiver_goal_loss", predicted_receiver_goal_loss.item())
        
        return (0.2 * (predicted_returns_loss + intent_loss))