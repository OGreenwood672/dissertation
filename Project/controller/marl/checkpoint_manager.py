import os
from datetime import datetime
import re
import json
import torch

from .config import MappoConfig


class CheckpointManager:

    def __init__(self, config: MappoConfig, default_load_previous=False):

        seed = config.seed
        
        result_path = "./results/" + config.communication_type.value + "/"
        
        os.makedirs(result_path, exist_ok=True)
        
        # find new seed
        if seed == -1 and not default_load_previous:
            used_seeds = list(map(lambda x: int(re.search(r"seed_(\d+)", x).group(1)), os.listdir(result_path)))
            test_used_seed = 0
            while True:
                if test_used_seed not in used_seeds:
                    seed = test_used_seed
                    break
                test_used_seed += 1
            result_path += self.get_folder_name(seed)
            self.gen_result_folder(result_path, config)
            self.is_new_run = True
        
        # find largest seed
        elif seed == -1 and default_load_previous:
            used_seeds = list(map(lambda x: int(re.search(r"seed_(\d+)", x).group(1)), os.listdir(result_path)))
            result_path += self.get_folder_name(max(used_seeds))
            self.is_new_run = False

        else: # Seed given
            folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
            if len(folders) > 0:
                result_path += folders[0]
                self.is_new_run = False
            else:
                result_path += self.get_folder_name(seed)
                self.gen_result_folder(result_path, config)
                self.is_new_run = True

        self.result_path = result_path
        self.seed = seed

    def load_config(self):
        config_path = self.result_path + "/config.json"
        return MappoConfig.from_json(config_path)

    def load_checkpoint_models(self, checkpoint_step=None):

        checkpoints_path = self.result_path + "/checkpoints"

        if checkpoint_step is None:
            avaliable_checkpoints = list(map(lambda x: int(re.search(r"checkpoint_(\d+)", x).group(1)), os.listdir(checkpoints_path)))
            if len(avaliable_checkpoints) == 0:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_path}")
            checkpoint_step = max(avaliable_checkpoints)


        model_path = self.result_path + "/checkpoints/checkpoint_" + str(checkpoint_step) + ".pth"
        try:
            checkpoint = torch.load(model_path)
        except:
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        actor = checkpoint["actor"]
        critic = checkpoint["critic"]
        actor_optimizer = checkpoint["actor_optimizer"]
        critic_optimizer = checkpoint["critic_optimizer"]

        return actor, critic, actor_optimizer, critic_optimizer, checkpoint_step
        

    def save_checkpoint(self, actor, critic, actor_optimizer, critic_optimizer, step):
        state = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "step": step
        }

        save_path = self.result_path + "/checkpoints/checkpoint_" + str(step) + ".pth"
        torch.save(state, save_path)

    @staticmethod
    def get_folder_name(seed):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{timestamp}_seed_{seed}"
    
    def gen_result_folder(self, result_path, config: MappoConfig):
        os.makedirs(result_path)

        # generate config (hyperparams)
        json.dump(config.get_dict(), open(result_path + "/config.json", "w"))

        # generate checkpoints folder
        os.makedirs(result_path + "/checkpoints")
    

    def get_seed(self):
        return self.seed
    
    def get_result_path(self):
        return self.result_path
    
