# %%
import sys
import os
sys.path.append(os.path.abspath('..'))

from loader import load_log_file, remove_result_folder, get_seeds

# %%
# Remove all folders with less than X logs
CUTOFF = 250

for communication_type in ["discrete", "continuous", "aim", "none"]:
    possible_seeds = get_seeds(communication_type)
    for seed in possible_seeds:
        try:
            logs = load_log_file(communication_type, seed)
            if len(logs) < CUTOFF:
                remove_result_folder(communication_type, seed)
        except AssertionError as e:
            remove_result_folder(communication_type, seed)
            



