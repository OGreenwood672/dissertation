import csv
import os

from controller.marl.core.logger import Logging
from controller.marl.core.config import GenerativeLangType
from dummy_config import Config


class Tracker:
    def __init__(self, values):
        self.values = values
        self.metrics = {k: [v] for k, v in values.items()}

    def get_average(self, key, default=""):
        return self.values.get(key, default)


def make_config(auto_type):
    config = Config()
    config.comms.autoencoder_type = auto_type
    return config


def test_logging_rl_writes_row(tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()

    config = make_config(GenerativeLangType.VQ_VAE)
    tracker = Tracker(
        {
            "reward_mean": 1.0,
            "reward_std": 0.1,
            "actor_loss": 0.5,
            "critic_loss": 0.6,
            "entropy_loss": 0.01,
            "action_loss": 0.02,
            "comm_loss": 0.03,
            "lstm_grad_norm": 0.9,
            "comm_grad_norm": 1.0,
            "action_grad_norm": 1.1,
            "comm_entropy": 0.7,
            "comm_perplexity": 2.0,
            "predicted_return_loss": 0.4,
            "predicted_sender_goal_loss": 0.5,
            "predicted_receiver_goal_loss": 0.6,
            "usage_count_0": 10,
            "std_usage_0": 0.2,
            "topk_usage_0": 3,
        }
    )

    logger = Logging(save_folder=str(folder), config=config, start_step=0, num_codebooks=1, mode="RL")

    logger.log(5, tracker)
    logger.close()

    log_file = folder / "log.csv"
    assert log_file.exists()

    with log_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["timestep"] == "5"
    assert rows[0]["reward_mean"] == "1.0"
    assert rows[0]["actor_loss"] == "0.5"


def test_logging_language(tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()

    config = make_config(GenerativeLangType.VQ_VAE)

    logger = Logging(save_folder=str(folder), config=config, start_step=0, num_codebooks=2, mode="language")

    assert "train_reconstruction_loss" in logger.headers
    assert "validation_reconstruction_loss" in logger.headers
    assert "train_commitment_loss_0" in logger.headers
    assert "validation_commitment_loss_1" in logger.headers

    logger.close()


def test_logging_clear_logs(monkeypatch, tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()

    config = make_config(GenerativeLangType.VQ_VAE)
    log_file = folder / "log.csv"

    headers = [
        "timestep",
        "reward_mean",
        "reward_std",
        "actor_loss",
        "critic_loss",
        "entropy_loss",
        "action_loss",
        "comm_loss",
        "lstm_grad_norm",
        "comm_grad_norm",
        "action_grad_norm",
        "comm_entropy",
        "comm_perplexity",
        "predicted_return_loss",
        "predicted_sender_goal_loss",
        "predicted_receiver_goal_loss",
    ]

    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow({"timestep": "1", "reward_mean": "1.0", "reward_std": "0.1"})
        writer.writerow({"timestep": "5", "reward_mean": "2.0", "reward_std": "0.2"})

    monkeypatch.setattr("builtins.input", lambda prompt: "Y")

    logger = Logging(save_folder=str(folder), config=config, start_step=3, num_codebooks=0, mode="RL")
    logger.close()

    with log_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["timestep"] == "1"


# def test_obs_logger_overwrite(tmp_path):
#     file = tmp_path / "obs.csv"
#     file.write_text("old stuff\n")

#     logger = ObsLogger(str(file))
#     logger.log([[1, 2, 3], [4, 5, 6]])
#     logger.close()

#     with file.open("r", newline="") as f:
#         reader = csv.reader(f)
#         rows = list(reader)

#     assert rows == [["1", "2", "3"], ["4", "5", "6"]]


# def test_comms_logger_creates_files(tmp_path):
#     folder = tmp_path / "comms"
#     folder.mkdir()

#     logger = CommsLogger(str(folder), vocab_size=3)

#     logger.log(0, [1, 2, 3])
#     logger.log(2, [7, 8, 9])
#     logger.close()

#     files = sorted(os.listdir(folder))
#     assert files == ["comms_0.csv", "comms_1.csv", "comms_2.csv"]

#     with (folder / "comms_0.csv").open("r", newline="") as f:
#         rows0 = list(csv.reader(f))

#     with (folder / "comms_2.csv").open("r", newline="") as f:
#         rows2 = list(csv.reader(f))

#     assert rows0 == [["1", "2", "3"]]
#     assert rows2 == [["7", "8", "9"]]