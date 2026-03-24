import argparse
import torch
import numpy as np

from controller.marl.models.aim import AIM
from project_paths import PROJECT_ROOT

from .config import CommunicationType, MappoConfig
import environment.environment as environment
from .checkpoint_manager import CheckpointManager
from .models import Encoder, PPO_Actor, PPO_Critic
from .env_wrapper import SimWrapper
from .train import train
from .run_sim import run_sim
from .train_ae import train_language


def get_args():
    parser = argparse.ArgumentParser(description="MAPPO Controller")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run in", choices=["train", "eval", "train-language", "eval-language"])
    
    return parser.parse_args()    

def setup(config, device, mode):

    cm = CheckpointManager(config, default_load_previous=mode == "eval")
    SEED = cm.get_seed()
    config.seed = SEED

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sim = SimWrapper(
        environment.Simulation(
            './configs/simulation.yaml',
            worlds_parallised=config.worlds_parallised
        ), 
        comm_dim=config.communication_size,
        num_comms=config.num_comms
    )

    num_agents = sim.get_num_agents()
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    comms_shape = (sim.get_world_comms_size(),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(),)

    # load encoder
    encoder = None
    aim = None
    if config.communication_type == CommunicationType.AIM:
        folder = AIM.get_folder(config)
        encoder = Encoder.load(folder)
        aim = AIM.load(folder, config, obs_mask=sim.get_agent_obs_mask(0), obs_dim=agent_obs_shape[0], action_count=act_shape[0])
        

    actor = PPO_Actor(
        num_agents, agent_obs_shape[0], act_shape[0], comm_dim=config.communication_size,
        communication_type=config.communication_type,
        lstm_hidden_size=config.lstm_hidden_size,
        vocab_size=config.vocab_size, num_comms=config.num_comms,
        encoder=encoder, device=device
    ).to(device)
    critic = PPO_Critic(num_agents, global_obs_shape[0], config.communication_size, config.num_comms, feature_dim=config.feature_dim).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    start_step = 0
    if not cm.is_new_run:
        actor_state, critic_state, actor_optimiser_state, critic_optimizer_state, start_step = cm.load_checkpoint_models()

        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)

        if mode == "train":
            actor_optimizer.load_state_dict(actor_optimiser_state)
            critic_optimizer.load_state_dict(critic_optimizer_state)

        print(f"Loaded checkpoint from step {start_step}")

    return {
        "sim": sim,
        "actor": actor,
        "critic": critic,
        "actor_opt": actor_optimizer,
        "critic_opt": critic_optimizer,
        "start_step": start_step,
        "checkpoint_manager": cm,
        "num_agents": num_agents,
        "agent_obs_shape": agent_obs_shape,
        "input_comms_shape": comms_shape,
        "act_shape": act_shape,
        "aim": aim
    }, config

def setup_language_analysis(config, device):
    
    
    cm = CheckpointManager(config, default_load_previous=True)
    SEED = cm.get_seed()
    config.seed = SEED

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sim = SimWrapper(
        environment.Simulation(
            str(PROJECT_ROOT / "configs" / "simulation.yaml"),
            worlds_parallised=config.worlds_parallised
        ), 
        comm_dim=config.communication_size,
        num_comms=config.num_comms
    )

    num_agents = sim.get_num_agents()
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    comms_shape = (sim.get_world_comms_size(),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(),)

    # load encoder
    encoder = None

    folder = AIM.get_folder(config)

    if config.communication_type == CommunicationType.AIM:
        encoder = Encoder.load(folder)

    aim = AIM.load(folder, config, obs_mask=torch.tensor(sim.get_agent_obs_mask(0), dtype=torch.bool, device=device), obs_dim=agent_obs_shape[0], action_count=act_shape[0])
        

    actor = PPO_Actor(
        num_agents, agent_obs_shape[0], act_shape[0], comm_dim=config.communication_size,
        communication_type=config.communication_type,
        lstm_hidden_size=config.lstm_hidden_size,
        vocab_size=config.vocab_size, num_comms=config.num_comms,
        encoder=encoder, device=device
    ).to(device)
    critic = PPO_Critic(num_agents, global_obs_shape[0], config.communication_size, config.num_comms, feature_dim=config.feature_dim).to(device)

    start_step = 0
    if not cm.is_new_run:
        actor_state, critic_state, _, _, start_step = cm.load_checkpoint_models()

        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)

        print(f"Loaded checkpoint from step {start_step}")

    return {
        "sim": sim,
        "actor": actor,
        "critic": critic,
        "start_step": start_step,
        "checkpoint_manager": cm,
        "num_agents": num_agents,
        "agent_obs_shape": agent_obs_shape,
        "input_comms_shape": comms_shape,
        "act_shape": act_shape,
        "aim": aim
    }, config



def evaluate(system, config, device):

    RUNS = 5

    baseline = run_sim(system, config, device, RUNS)

    print(f"Baseline: {np.mean(baseline)}")

    # Communication test
    no_comm_baseline = run_sim(system, config, device, RUNS, zero_comms=True)

    print(f"No Communication: {np.mean(no_comm_baseline)}")


def evaluate_language(system, config, device):

    RUNS = 15

    cm = system["checkpoint_manager"]

    baseline = run_sim(system, config, device, RUNS, collect_comms_folder=cm.get_result_path() + "/comms/")

    print(f"Reward Mean over {RUNS} runs: {np.mean(baseline)}")


def main():

    config_pth = './configs/mappo_settings.yaml'

    args = get_args()

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = MappoConfig.from_yaml(config_pth)

    system, config = setup(config, device, mode=args.mode)

    if args.mode == "train":
        system["actor"].train()
        train(system, config, device)
    elif args.mode == "eval":
        system["actor"].eval()
        evaluate(system, config, device)
    elif args.mode == "train-language":
        system["actor"].eval()
        train_language(system, config, device)
    elif args.mode == "eval-language":
        system["actor"].eval()
        evaluate_language(system, config, device)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()