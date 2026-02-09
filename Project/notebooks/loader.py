import os
import pandas as pd
import shutil


def load_log_file(communication_type, seed):

    result_path = "../../results/" + communication_type + "/"
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        logfile = result_path + folders[0] + "/log.csv"
        assert os.path.exists(logfile), f"Log file {logfile} does not exist"
        return pd.read_csv(logfile)
    else:
        raise FileNotFoundError(f"No log file found for seed <{seed}> and communication type <{communication_type}>")


def remove_result_folder(communication_type, seed):
    result_path = "../../results/" + communication_type + "/"
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        folder = result_path + folders[0]
        assert os.path.exists(folder), f"Folder {folder} does not exist"
        if os.path.exists(folder):
            shutil.rmtree(folder)
    else:
        raise FileNotFoundError(f"No Folder found for seed <{seed}> and communication type <{communication_type}>")


def get_seeds(communication_type):
    result_path = "../../results/" + communication_type + "/"
    return list(map(lambda x: int(x.split("_")[-1]), os.listdir(result_path)))