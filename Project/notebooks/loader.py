import json
import os
import pandas as pd
import shutil

from controller.marl.core.config import GenerativeLangType
from project_paths import LANGUAGES_DIR, RESULTS_DIR


def load_log_file(communication_type, seed):

    result_path = RESULTS_DIR / communication_type
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        logfile = result_path / folders[0] / "log.csv"
        assert os.path.exists(logfile), f"Log file {logfile} does not exist"
        return pd.read_csv(logfile)
    else:
        raise FileNotFoundError(f"No log file found for seed <{seed}> and communication type <{communication_type}>")
    
def load_comm_file(communication_type, seed, comm_index):

    result_path = RESULTS_DIR / communication_type
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        logfile = result_path / folders[0] / "comms" / f"comms_{comm_index}.csv"
        assert os.path.exists(logfile), f"Log file {logfile} does not exist"
        return pd.read_csv(logfile)
    else:
        raise FileNotFoundError(f"No comms folder found for seed <{seed}> and communication type <{communication_type}>")


def remove_result_folder(communication_type, seed):
    result_path = RESULTS_DIR / communication_type
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        folder = result_path / folders[0]
        assert os.path.exists(folder), f"Folder {folder} does not exist"
        if os.path.exists(folder):
            shutil.rmtree(folder)
    else:
        raise FileNotFoundError(f"No Folder found for seed <{seed}> and communication type <{communication_type}>")


def get_seeds(communication_type):
    result_path = RESULTS_DIR / communication_type
    return list(map(lambda x: int(x.split("_")[-1]), os.listdir(result_path)))

def load_vae_training_log(vae_type: GenerativeLangType, most_recent: int = 1):

    folder_paths = []
    for folder in os.listdir(LANGUAGES_DIR)[::-1]:
        with open(LANGUAGES_DIR / folder / "config.json", "r") as json_file:
            check_config = json.load(json_file)

            if check_config["autoencoder_type"] == vae_type.value:
                folder_paths.append(LANGUAGES_DIR / folder)
                if len(folder_paths) == most_recent:
                    break

    print(f"Loading {len(folder_paths)} languages from folders")

    df = pd.concat([pd.read_csv(f / "log.csv") for f in folder_paths], ignore_index=True)

    return df


def load_specified_vae_training_log(TZs):

    folder_paths = []
    for folder in TZs:
        os.path.exists(LANGUAGES_DIR / folder)
        folder_paths.append(LANGUAGES_DIR / folder)

    print(f"Loading {len(folder_paths)} languages from folders")

    df = pd.concat([pd.read_csv(f / "log.csv") for f in folder_paths], ignore_index=True)

    return df