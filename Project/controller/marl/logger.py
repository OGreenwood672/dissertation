import json
import os
import csv

from logging_utils.decorators import LoggingFunctionIdentification



class Logger:

    def __init__(self, save_folder, start_step=0):
        
        assert os.path.exists(save_folder), f"Save folder {save_folder} does not exist"
        self.save_file = os.path.join(save_folder, "log.csv")

        self.headers = [
            "timestep",
            "reward_mean", "reward_std",
            "communication_entropy_mean", "communication_perplexity_mean",
            "codebook_loss", "communication_active_codebook_usage",
            "actor_loss", "critic_loss", "entropy_loss", "vq_loss"
        ]

        file_exists = os.path.exists(self.save_file)

        if file_exists:
            self.clear_logs(start_step)

        self.file = open(self.save_file, "a")
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()
        
    @LoggingFunctionIdentification("LOGGER")
    def log(
        self, timestep="",
        reward_mean="", reward_std="",
        communication_entropy_mean="", communication_perplexity_mean="",
        codebook_loss="", communication_active_codebook_usage="",
        actor_loss="", critic_loss="", entropy_loss="", vq_loss=""
    ):
        
        try:
            self.writer.writerow({
                "timestep": timestep,
                "reward_mean": reward_mean, "reward_std": reward_std,
                "communication_entropy_mean": communication_entropy_mean, "communication_perplexity_mean": communication_perplexity_mean,
                "codebook_loss": codebook_loss, "communication_active_codebook_usage": communication_active_codebook_usage,
                "actor_loss": actor_loss, "critic_loss": critic_loss, "entropy_loss": entropy_loss, "vq_loss": vq_loss
            })
            self.file.flush()
        except ValueError as e:
            print(f"Failed to log row: {e}")


    def clear_logs(self, start_step):

        if input("Logs must be truncated after time step {start_step}. Do you wish to proceed (Y/n)").lower() == "n":
            quit()

        print(f"Truncating log at timestep {start_step} and onwards...")

        try:
            rows = []
            with open(self.save_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if int(float(row["timestep"])) <= start_step:
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
    
