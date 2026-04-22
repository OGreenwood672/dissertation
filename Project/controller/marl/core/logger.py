import os
import csv

from controller.marl.core.config import Config, GenerativeLangType
from controller.marl.core.metric_tracker import MetricTracker
from logging_utils.decorators import LoggingFunctionIdentification


class Logging:
    def __init__(self, save_folder: str, config: Config, start_step: int = 0, num_codebooks: int = 1, mode: str = "RL"):
        
        self.config = config
        
        assert os.path.exists(save_folder), f"Save folder {save_folder} does not exist"
        self.save_file = os.path.join(save_folder, "log.csv")

        if mode == "RL":
            
            self.headers = [
                "timestep",
                "reward_mean", "reward_std",
                "actor_loss", "critic_loss",
                "entropy_loss", "action_loss", "comm_loss",
                "lstm_grad_norm", "comm_grad_norm", "action_grad_norm",
                "comm_entropy", "comm_perplexity",
                "predicted_return_loss", "predicted_sender_goal_loss", "predicted_receiver_goal_loss"
            ]
            self.headers += [s for x in range(num_codebooks) for s in [f"usage_count_{x}", f"std_usage_{x}", f"topk_usage_{x}"]]
        
        elif mode == "language":
            self.headers = [
                "timestep",
                "train_reconstruction_loss",
                "validation_reconstruction_loss"
            ]
            if config.comms.autoencoder_type == GenerativeLangType.VQ_VAE:
                self.headers += [s for x in range(num_codebooks) for s in [f"train_commitment_loss_{x}", f"validation_commitment_loss_{x}"]]
            elif config.comms.autoencoder_type == GenerativeLangType.SQ_VAE:
                self.headers += [s for x in range(num_codebooks) for s in [f"train_kl_divergence_{x}", f"validation_kl_divergence_{x}"]]
            elif config.comms.autoencoder_type == GenerativeLangType.HQ_VAE:
                self.headers += [s for x in range(num_codebooks) for s in [
                    f"train_kl_divergence_{x}",
                    f"validation_kl_divergence_{x}"
                    f"train_secondary_kl_divergence_{x}",
                    f"validation_secondary_kl_divergence_{x}"
                ]]
            else:
                assert False

        elif mode == "imitate":
            self.headers = [
                "timestep",
                "train_actor_loss", "train_critic_loss", "train_accuracy",
                "validation_actor_loss", "validation_critic_loss", "validation_accuracy",
                "test_actor_loss", "test_critic_loss", "test_accuracy",
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")


        file_exists = os.path.exists(self.save_file)

        if file_exists:
            self.clear_logs(start_step)

        self.file = open(self.save_file, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    @LoggingFunctionIdentification("LOGGER")
    def log(self, timestep: int, tracker: MetricTracker):
        
        datarow = {h: tracker.get_average(h, default="") for h in self.headers}
        datarow["timestep"] = timestep
        
        try:
            self.writer.writerow(datarow)
            self.file.flush()
        except ValueError as e:
            print(f"Failed to log row: {e}")
    
    
    def clear_logs(self, start_step):

        if input(f"Logs must be truncated after time step {start_step}. Do you wish to proceed (Y/n)").lower() == "n":
            quit()

        print(f"Truncating log at timestep {start_step} and onwards...")

        try:
            rows = []
            with open(self.save_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if int(float(row["timestep"])) < start_step:
                            rows.append(row)
                    except ValueError:
                        continue

            with open(self.save_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()

        except Exception as e:
            print(f"Error clearing logs: {e}")

    def close(self):
        self.file.close()



# class ObsLogger:

#     def __init__(self, save_file):

#         self.save_file = save_file

#         file_exists = os.path.exists(self.save_file)

#         if file_exists:
#             os.remove(self.save_file)

#         self.file = open(self.save_file, "a", newline="")
#         self.writer = csv.writer(self.file)

#     @LoggingFunctionIdentification("LOGGER")
#     def log(self, obs):
#         try:
#             self.writer.writerows(obs)
#         except ValueError as e:
#             print(f"Failed to log row: {e}")


#     def close(self):
#         self.file.close()

# class CommsLogger:

#     def __init__(self, save_folder, vocab_size):

#         self.save_folder = save_folder
#         self.vocab_size = vocab_size

#         os.makedirs(save_folder, exist_ok=True)

#         for file in os.listdir(save_folder):
#             os.remove(os.path.join(save_folder, file))

#         self.save_files = [open(os.path.join(save_folder, f"comms_{i}.csv"), "a", newline="") for i in range(vocab_size)]

#         self.writers = [csv.writer(file) for file in self.save_files]


#     @LoggingFunctionIdentification("LOGGER")
#     def log(self, comm_index, obs):
#         try:
#             self.writers[comm_index].writerow(obs)
#         except ValueError as e:
#             print(f"Failed to log row: {e}")

#     def close(self):
#         for f in self.save_files:
#             f.close()
