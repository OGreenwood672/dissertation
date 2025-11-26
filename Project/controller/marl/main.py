from logging_utils.decorators import LoggingFunctionIdentification
from time import sleep
import environment.environment as environment
import torch
import torch.nn.functional as F
import numpy as np
import concurrent.futures

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
    simulation_timesteps = 3000
    ppo_epochs = 10
    lstm_hidden_size = 64
    gamma = 0.99
    gae_lambda = 0.95

    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.04

    worlds_parallised = 4
    batch_runs = buffer_size // worlds_parallised

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

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=worlds_parallised)

    def parallel_step(args):
        world_id, action = args
        return env.step(world_id, action)

    def parallel_reset(world_id):
        return env.reset(world_id)

    for i in range(training_timesteps):

        for batch_run in range(batch_runs):
            
            current_episodes_data = [{
                "obs": [], "actions": [], "log_probs": [], "rewards": [], "c_values": [], "a_hidden_states": [], "c_hidden_states": []
            } for _ in range(worlds_parallised)]
            
            curr_obs_for_batch = list(executor.map(parallel_reset, range(worlds_parallised)))
            
            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden(batch_size=worlds_parallised)
            critic_hidden_states = critic.init_hidden(batch_size=worlds_parallised)
            
            for j in range(simulation_timesteps):

                # Move hidden states to CPU for storage
                for w in range(worlds_parallised):
                        
                    world_agents_states = []
                    for agent in range(num_agents):
                        h_batch, c_batch = actor_hidden_states[agent]
                        world_agents_states.append(
                            (h_batch[w].detach().cpu(), c_batch[w].detach().cpu())
                        )

                    current_episodes_data[w]["a_hidden_states"].append(world_agents_states)

                    world_critic_states = []
                    for agent in range(num_agents):
                        h_batch, c_batch = critic_hidden_states[agent]
                        world_critic_states.append(
                            (h_batch[w].detach().cpu(), c_batch[w].detach().cpu())
                        )
                        
                    current_episodes_data[w]["c_hidden_states"].append(world_critic_states)

                # Get action logits from the model and update memory
                obs_tensor = torch.tensor(np.stack(curr_obs_for_batch)).float().to(device)
                with torch.no_grad():
                    action_logits, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                    value, critic_hidden_states = critic(obs_tensor, critic_hidden_states)
                
                # action_logits = action_logits.squeeze(0)
                # Basically softmax to determine chosen action
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()

                log_probs = dist.log_prob(actions)

                # Use chosen actions in env
                action_choice = actions.cpu().numpy()
                # Add world id to actions
                step_args = zip(range(worlds_parallised), action_choice)
                results = list(executor.map(parallel_step, step_args))
                
                next_obs = [r[0] for r in results]
                rewards = [r[1] for r in results]
                # sleep(0.5)

                # Save to current episodes
                for w in range(worlds_parallised):
                    current_episodes_data[w]["obs"].append(curr_obs_for_batch[w])
                    current_episodes_data[w]["actions"].append(action_choice[w])
                    current_episodes_data[w]["log_probs"].append(log_probs[w].cpu().numpy())
                    current_episodes_data[w]["rewards"].append(rewards[w])
                    current_episodes_data[w]["c_values"].append(value[w].cpu())

                curr_obs_for_batch = next_obs

            # Save to buffer
            for w in range(worlds_parallised):
                buffer.add_episode(current_episodes_data[w])
            
            print(f"Collected batch {batch_run+1}/{batch_runs} (Total {worlds_parallised * (batch_run+1)} episodes)")

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