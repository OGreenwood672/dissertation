import argparse
import torch
import numpy as np

from controller.marl.runners.imitation_learning import imitation_learning
from controller.marl.core.logger import MetricTracker
from controller.marl.models.aim import AIM
from project_paths import PROJECT_ROOT, LANGUAGES_DIR

from .core.config import CommunicationType, Config
import environment.environment as environment
from .core.checkpoint_manager import CheckpointManager
from .models import Encoder, PPO_Actor, PPO_Critic
from .env_wrapper import SimWrapper
from .runners.ppo_train import train
from .runners.sim_runner import run_sim
from .runners.autoencoder import train_language


def get_args():
    parser = argparse.ArgumentParser(description="MAPPO Controller")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run in", choices=["train", "eval", "train-language", "eval-language", "imitate"])
    
    return parser.parse_args()    

def setup(config: Config, device: torch.device, load_agent_architecture: bool = True, imitate: bool = False):

    cm = CheckpointManager(config)
    SEED = cm.get_seed()
    config.training.seed = SEED

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sim = SimWrapper(
        environment.Simulation(
            str(PROJECT_ROOT / "configs" / "simulation.yaml"),
            worlds_parallised=config.training.worlds_parallised
        ), 
        comm_dim=config.comms.communication_size,
        num_comms=config.comms.num_comms
    )

    num_agents = len(config.simulation.agents)
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    comms_shape = (sim.get_world_comms_size(),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(),)
    obs_external_mask = torch.tensor(sim.get_agent_external_obs_mask(0), dtype=torch.bool, device=device)

    actor = None
    critic = None
    actor_optimizer = None
    critic_optimizer = None
    start_step = 0

    tracker = MetricTracker()

    if load_agent_architecture:

        actor = PPO_Actor(
            num_agents, agent_obs_shape[0], act_shape[0], obs_external_mask, config.actor, config.comms, tracker.update, device=device
        ).to(device)
        critic = PPO_Critic(num_agents, global_obs_shape[0], config.critic, config.comms).to(device)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.mappo.actor_learning_rate)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.mappo.critic_learning_rate)

        start_step = 0
        if not cm.is_new_run:
            actor_state, critic_state, actor_optimiser_state, critic_optimizer_state, start_step = cm.load_checkpoint_models()

            if actor_state and critic_state:
                actor.load_state_dict(actor_state)
                critic.load_state_dict(critic_state)

                try:
                    actor_optimizer.load_state_dict(actor_optimiser_state)
                    critic_optimizer.load_state_dict(critic_optimizer_state)
                    print("Successfully loaded optimizer states.")
                except ValueError as e:
                    if "different number of parameter groups" in str(e):
                        print("Optimizer parameter groups changed. Starting with fresh optimizer states.")
                    else:
                        raise e

                print(f"Loaded checkpoint from step {start_step}")

        elif imitate:
            actor_state, critic_state, actor_optimiser_state, critic_optimizer_state, start_step = cm.load_base_models()

            actor.load_state_dict(actor_state)
            critic.load_state_dict(critic_state)

            print(f"Loaded base models!")



    return {
        "sim": sim,
        "actor": actor,
        "critic": critic,
        "actor_opt": actor_optimizer,
        "critic_opt": critic_optimizer,
        "start_step": start_step,
        "checkpoint_manager": cm,
        "metric_tracker": tracker,
        "num_agents": num_agents,
        "agent_obs_shape": agent_obs_shape,
        "input_comms_shape": comms_shape,
        "act_shape": act_shape,
    }, config

def setup_language_analysis(config: Config, device: torch.device):
    
    
    # cm = CheckpointManager(config)
    # SEED = cm.get_seed()
    # config.training.seed = SEED

    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(SEED)

    sim = SimWrapper(
        environment.Simulation(
            str(PROJECT_ROOT / "configs" / "simulation.yaml"),
            worlds_parallised=config.training.worlds_parallised
        ), 
        comm_dim=config.comms.communication_size,
        num_comms=config.comms.num_comms
    )

    num_agents = len(config.simulation.agents)
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    comms_shape = (sim.get_world_comms_size(),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(),)
    obs_external_mask = torch.tensor(sim.get_agent_external_obs_mask(0), dtype=torch.bool, device=device)

    filtered_obs_dim = obs_external_mask.sum().int().item()

    loss_obs_mask = torch.tensor(sim.get_agent_obs_mask(0), dtype=torch.bool, device=device)[obs_external_mask]

    if not config.comms.comm_folder:
        folder = AIM.get_folder(config)
    else:
        folder = LANGUAGES_DIR / config.comms.comm_folder

    aim = AIM.load(folder, config.aim_training, config.comms, obs_dim=filtered_obs_dim, obs_loss_mask=loss_obs_mask, device=device)
    aim.eval()

    metric_tracker = MetricTracker()

    actor = PPO_Actor(
        num_agents, agent_obs_shape[0], act_shape[0], obs_external_mask, config.actor, config.comms, metric_tracker.update, device=device
    ).to(device)
    critic = PPO_Critic(num_agents, global_obs_shape[0], config.critic, config.comms).to(device)

    # start_step = 0
    # if not cm.is_new_run:
    #     actor_state, critic_state, _, _, start_step = cm.load_checkpoint_models()

    #     actor.load_state_dict(actor_state)
    #     critic.load_state_dict(critic_state)

    #     print(f"Loaded checkpoint from step {start_step}")

    return {
        "sim": sim,
        "actor": actor,
        "critic": critic,
        # "start_step": start_step,
        # "checkpoint_manager": cm,
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

    config_pth = PROJECT_ROOT / "configs"

    args = get_args()

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config.from_yaml(config_pth)
    
    load_agent_architecture = True
    if args.mode == "train-language":
        if input("Use optimal actions? (Y/n) ").lower() != "n":
            load_agent_architecture = False

    system, config = setup(config, device, load_agent_architecture, imitate=args.mode == "train")

    if args.mode == "train":
        system["actor"].train()
        train(system, config, device)
    elif args.mode == "eval":
        system["actor"].eval()
        evaluate(system, config, device)
    elif args.mode == "train-language":
        train_language(system, config, device, use_optimal=not load_agent_architecture)
    elif args.mode == "eval-language":
        system["actor"].eval()
        evaluate_language(system, config, device)
    elif args.mode == "imitate":
        imitation_learning(system, config, device)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()