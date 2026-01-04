import yaml
import json
import os


CONFIGS_PATH = "../configs/"
MAPPO_CONFIG_PATH = os.path.join(CONFIGS_PATH, "mappo_settings.yaml")
SIM_CONFIG_PATH = os.path.join(CONFIGS_PATH, "simulation.yaml")

FRONTEND_DEST_PATH = './public/frontend_config.json'

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Syncing configs...")

    if not os.path.exists(MAPPO_CONFIG_PATH) or not os.path.exists(SIM_CONFIG_PATH):
        print("[ERROR] Config not found!")
        return

    mappo_config = load_yaml(MAPPO_CONFIG_PATH)
    sim_config = load_yaml(SIM_CONFIG_PATH)

    num_worlds = mappo_config['training']['worlds_parallised']
    
    sim_config['worlds_parallised'] = num_worlds

    os.makedirs(os.path.dirname(FRONTEND_DEST_PATH), exist_ok=True)
    
    with open(FRONTEND_DEST_PATH, 'w') as f:
        json.dump(sim_config, f, indent=4)

    print(f"Config created successfully!")

if __name__ == "__main__":
    main()