from pathlib import WindowsPath

from pydantic import BaseModel, ConfigDict
from typing import List
import yaml


from enum import Enum
class CommunicationType(str, Enum):
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"
    AIM = "AIM"
    NONE = "NONE"
    REFLECTIVE = "REFLECTIVE"

class GenerativeLangType(str, Enum):
    SQ_VAE = "SQ-VAE"
    VQ_VAE = "VQ-VAE"
    HQ_VAE = "HQ-VAE"

class AgentConfig(BaseModel):
    id: int
    inputs: List[str]
    output: str

class StationConfig(BaseModel):
    id: int
    type: str
    resource: str

class TrainingConfig(BaseModel):
    training_timesteps: int
    simulation_timesteps: int
    timestep: float
    batch_size: int
    rollout_phases: int
    worlds_parallised: int
    seed: int
    periodic_save_interval: int

class SimulationConfig(BaseModel):
    arena_width: int
    arena_height: int
    headless: bool
    websocket_url: str
    websocket_path: str
    agent_agent_visibility: int
    agent_station_visibility: int
    initial_agent_layout: str
    agents: List[AgentConfig]
    station_layout: str
    stations: List[StationConfig]

class MappoHyperparameters(BaseModel):
    ppo_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    vf_coef: float
    ent_coef: float
    actor_learning_rate: float
    critic_learning_rate: float

class ImitationHyperparameters(BaseModel):
    epochs: int
    obs_runs: int

class ActorHyperparameters(BaseModel):
    lstm_hidden_size: int
    feature_dim: int

class CriticHyperparameters(BaseModel):
    feature_dim: int


class AimTrainingConfig(BaseModel):
    communication_size: int
    aim_learning_rate: float
    obs_runs: int
    obs_runs_noise: float
    aim_batch_size: int
    ae_epochs: int

    commitment_cost: float

    kl_weight: float
    min_temperature: float
    init_temperature: float

class CommConfig(BaseModel):
    communication_type: CommunicationType

    vocab_size: int
    num_comms: int
    autoencoder_type: GenerativeLangType
    rq_levels: int
    communication_size: int

    comm_folder: str


class AIMConfig(BaseModel):

    aim_learning_rate: float
    obs_runs: int
    obs_runs_noise: float
    aim_batch_size: int
    ae_epochs: int

    hidden_size: int

    commitment_cost: float
    kl_weight: float
    min_temperature: float
    init_temperature: float


class Config(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    training: TrainingConfig
    simulation: SimulationConfig
    mappo: MappoHyperparameters
    actor: ActorHyperparameters
    critic: CriticHyperparameters
    aim_training: AIMConfig
    comms: CommConfig
    imitation: ImitationHyperparameters
    
    # @classmethod
    # def from_json(cls, config_path: str):

    #     try:
    #         raw_config = json.load(open(config_path))
    #     except:
    #         raise FileNotFoundError(f"Config file not found at {config_path}")
        
    #     valid_keys = {k: v for k, v in raw_config.items() if k in cls.__annotations__}

    #     return cls(**valid_keys)
    
    # def get_dict(self):
    #     return self.model_dump()
    
    @staticmethod
    def load(config_file_path: WindowsPath):
        with open(config_file_path) as f:
            raw_config = yaml.safe_load(f)
    
            config = {}
            while raw_config.items():
                k, v = raw_config.popitem()
                if isinstance(v, dict):
                    raw_config.update(v)
                else:
                    config[k] = v

            if "communication_type" in config:
                config["communication_type"] = CommunicationType(config["communication_type"].upper())
            if "autoencoder_type" in config:
                config["autoencoder_type"] = GenerativeLangType(config["autoencoder_type"].upper())

            return config
    
    @classmethod
    def from_yaml(cls, configs_path: WindowsPath):
        
        raw_aim = Config.load(configs_path / 'aim_training.yaml')
        raw_comms = Config.load(configs_path / 'comms.yaml')
        raw_mappo = Config.load(configs_path / 'mappo.yaml')
        raw_sim = Config.load(configs_path / 'simulation.yaml')
        raw_train = Config.load(configs_path / 'training.yaml')
        raw_actor = Config.load(configs_path / 'actor.yaml')
        raw_critic = Config.load(configs_path / 'critic.yaml')
        raw_imitation = Config.load(configs_path / 'imitation.yaml')


        combined_data = {
            "training": raw_train,
            "simulation": raw_sim,
            "mappo": raw_mappo,
            "aim_training": raw_aim,
            "comms": raw_comms,
            "actor": raw_actor,
            "critic": raw_critic,
            "imitation": raw_imitation
        }

        return cls(**combined_data)
    
    def save(self, save_path: WindowsPath):
        data = self.model_dump(mode='json')
        
        file_mapping = {
            "training": "training.yaml",
            "simulation": "simulation.yaml",
            "mappo": "mappo.yaml",
            "actor": "actor.yaml",
            "critic": "critic.yaml",
            "aim_training": "aim_training.yaml",
            "comms": "comms.yaml",
            "imitation": "imitation.yaml"
        }
        
        for field, filename in file_mapping.items():
            file_path = save_path / filename
            with open(file_path, 'w') as f:
                yaml.safe_dump(data[field], f, default_flow_style=False)