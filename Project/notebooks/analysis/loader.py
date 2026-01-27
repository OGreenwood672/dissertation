import os
import pandas as pd


def load_log_file(communication_type, seed):

    result_path = "../../results/" + communication_type + "/"
    folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
    if len(folders) > 0:
        logfile = result_path + folders[0] + "/log.csv"
        assert os.path.exists(logfile), f"Log file {logfile} does not exist"
        return pd.read_csv(logfile)
    else:
        raise FileNotFoundError(f"No log file found for seed <{seed}> and communication type <{communication_type}>")
    