from logging_utils.decorators import LoggingFunctionIdentification
from time import sleep
import environment.environment as environment
import torch
import torch.nn.functional as F
import numpy as np
from .buffer import RolloutBuffer
from .models import PPO_Actor, PPO_Critic



@LoggingFunctionIdentification("CONTROLLER")
def main():
    print("starting program")

    env = environment.Simulation('./configs/simulation.yaml')
    
    buffer_size = 20
    num_agents = 1
    obs_shape = (25,)
    act_shape = (5,)
    training_timesteps = 1000
    simulation_timesteps = 900
    ppo_epochs = 10
    lstm_hidden_size = 64
    gamma = 0.99
    gae_lambda = 0.95

    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    actor = PPO_Actor(num_agents, obs_shape[0], act_shape[0], lstm_hidden_size).to(device)
    critic = PPO_Critic(num_agents, obs_shape[0], lstm_hidden_size).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        num_agents=num_agents,
        obs_space_shape=obs_shape,
        action_space_shape=act_shape,
        device=device
    )

    for i in range(training_timesteps):

        for ep in range(buffer_size):
            
            current_episode_data = {
                "obs": [], "actions": [], "log_probs": [], "rewards": [], "c_values": [], "a_hidden_states": [], "c_hidden_states": []
            }
            
            curr_obs = env.reset()
            
            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden()
            critic_hidden_states = critic.init_hidden()
            
            for j in range(simulation_timesteps):

                # Move hidden states to CPU for storage
                current_episode_data["a_hidden_states"].append([(h.cpu(), c.cpu()) for h, c in actor_hidden_states])
                
                # Get action logits from the model and update memory
                obs_tensor = torch.tensor(np.stack(curr_obs)).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    action_logits, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                    value, critic_hidden_states = critic(obs_tensor, critic_hidden_states)
                

                action_logits = action_logits.squeeze(0)
                # Basically softmax to determine chosen action
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()

                log_probs = dist.log_prob(actions)

                # Use chosen actions in env
                action_choice = [action.cpu().numpy() for action in actions]
                next_obs, rewards = env.step(action_choice)
                # sleep(0.5);

                # Save to current episode
                current_episode_data["obs"].append(curr_obs)
                current_episode_data["actions"].append(actions.cpu().numpy())
                current_episode_data["log_probs"].append(log_probs.cpu().numpy())
                current_episode_data["rewards"].append(rewards)
                current_episode_data["c_values"].append(value.squeeze(0).cpu())
                current_episode_data["c_hidden_states"].append([(h.cpu(), c.cpu()) for h, c in critic_hidden_states])

                curr_obs = next_obs

            # Save to buffer
            buffer.add_episode(current_episode_data)
            print(f"Collected episode {ep+1}/{buffer_size}")

        print("Buffer is full. Creating batch...")

        buffer.flatten_episodes_to_batch()
        buffer.compute_advantages(0, gamma, gae_lambda)
        
        for _ in range(ppo_epochs):
            for batch in buffer.get_minibatches():
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs = batch["log_probs"]
                rewards = batch["rewards"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                a_hidden_states = batch["a_hidden_states"]
                c_hidden_states = batch["c_hidden_states"]
                c_values = batch["c_values"]
                
                batchsize = obs.shape[0]

                a_h, a_c = a_hidden_states.squeeze(3).unbind(dim=2)
                c_h, c_c = c_hidden_states.squeeze(3).unbind(dim=2)

                batch_a_hidden_states = [(a_h[:, i, :], a_c[:, i, :]) for i in range(num_agents)]
                batch_c_hidden_states = [(c_h[:, i, :], c_c[:, i, :]) for i in range(num_agents)]
                
                new_logits, _ = actor(obs, batch_a_hidden_states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_log_p = new_dist.log_prob(actions)
                entropy = new_dist.entropy()
                
                # Critic Loss
                new_values, _ = critic(obs, batch_c_hidden_states)
                critic_loss = F.mse_loss(new_values.reshape(-1), returns.reshape(-1))
                
                # PPO Actor Loss
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Calculate ratio
                log_ratio = new_log_p - log_probs
                ratio = log_ratio.exp()
                
                # PPO clipped loss
                loss1 = advantages * ratio
                loss2 = advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                actor_loss = -torch.min(loss1, loss2).mean()
                
                # Total Loss
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss
                
                # Update
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                
                actor_optimizer.step()
                critic_optimizer.step()

if __name__ == "__main__":
    main()