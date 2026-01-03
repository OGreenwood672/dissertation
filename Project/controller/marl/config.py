from dataclasses import dataclass
import yaml

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    

@dataclass(frozen=True)
class MappoConfig:

    buffer_size: int
    training_timesteps: int
    simulation_timesteps: int
    worlds_parallised: int
    
    ppo_epochs: int
    lstm_hidden_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    vf_coef: float
    ent_coef: float
    
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

        valid_keys = {k: v for k, v in config.items() if k in cls.__annotations__}
        
        return cls(**valid_keys)