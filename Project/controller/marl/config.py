from pydantic import BaseModel, ConfigDict
import json
import yaml

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    

from enum import Enum
class CommunicationType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    AIM = "aim"
    NONE = "none"


class MappoConfig(BaseModel):

    model_config = ConfigDict(frozen=True)

    timestep: float
    buffer_size: int
    training_timesteps: int
    simulation_timesteps: int
    worlds_parallised: int
    seed: int
    periodic_save_interval: int
    
    ppo_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    vf_coef: float
    ent_coef: float

    actor_learning_rate: float
    critic_learning_rate: float

    lstm_hidden_size: int
    feature_dim: int

    communication_type: CommunicationType
    communication_size: int
    aim_seed: int
    vocab_size: int
    num_comms: int
    
    @classmethod
    def from_yaml(cls, config_path: str):

        raw_config = load_yaml(config_path)
        
        config = {}
        for section in raw_config.values():
            if isinstance(section, dict):
                config.update(section)
            else:
                config = raw_config
                break
        
        if "communication_type" in config:
            config["communication_type"] = CommunicationType(config["communication_type"])

        valid_keys = {k: v for k, v in config.items() if k in cls.__annotations__}
        
        return cls(**valid_keys)
    
    @classmethod
    def from_json(cls, config_path: str):

        try:
            raw_config = json.load(open(config_path))
        except:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        valid_keys = {k: v for k, v in raw_config.items() if k in cls.__annotations__}

        return cls(**valid_keys)
    
    def get_dict(self):
        return self.model_dump()