import yaml

from controller.marl.core.config import Config, CommunicationType, GenerativeLangType


def test_load(tmp_path):
    file = tmp_path / "comms.yaml"
    file.write_text(yaml.safe_dump({
        "communication_type": "discrete",
        "vocab_size": 8,
        "num_comms": 2,
        "autoencoder_type": "vq-vae",
        "rq_levels": 3,
        "communication_size": 16,
        "comm_folder": "test",
    }))

    data = Config.load(file)

    assert data["communication_type"] == CommunicationType.DISCRETE
    assert data["autoencoder_type"] == GenerativeLangType.VQ_VAE
    assert data["vocab_size"] == 8


def test_from_yaml(tmp_path):
    for name, data in {
        "training.yaml": {
            "training_timesteps": 1000,
            "simulation_timesteps": 100,
            "timestep": 0.1,
            "batch_size": 32,
            "rollout_phases": 4,
            "worlds_parallised": 2,
            "seed": 1,
            "periodic_save_interval": 50,
        },
        "simulation.yaml": {
            "arena_width": 10,
            "arena_height": 10,
            "headless": True,
            "websocket_url": "ws://localhost",
            "websocket_path": "/ws",
            "agent_agent_visibility": 5,
            "agent_station_visibility": 5,
            "initial_agent_layout": "grid",
            "agents": [{"id": 1, "inputs": ["x"], "output": "y"}],
            "station_layout": "fixed",
            "stations": [{"id": 1, "type": "mine", "resource": "iron"}],
        },
        "mappo.yaml": {
            "ppo_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
        },
        "actor.yaml": {"lstm_hidden_size": 64, "feature_dim": 128},
        "critic.yaml": {"feature_dim": 128},
        "aim_training.yaml": {
            "aim_learning_rate": 0.001,
            "obs_runs": 5,
            "obs_runs_noise": 0.1,
            "aim_batch_size": 16,
            "ae_epochs": 2,
            "hidden_size": 32,
            "commitment_cost": 0.25,
            "kl_weight": 0.01,
            "min_temperature": 0.5,
            "init_temperature": 1.0,
        },
        "comms.yaml": {
            "communication_type": "discrete",
            "vocab_size": 8,
            "num_comms": 2,
            "autoencoder_type": "vq-vae",
            "rq_levels": 3,
            "communication_size": 16,
            "comm_folder": "test",
        },
        "imitation.yaml": {"epochs": 3, "obs_runs": 5},
    }.items():
        (tmp_path / name).write_text(yaml.safe_dump(data))

    config = Config.from_yaml(tmp_path)

    assert config.training.training_timesteps == 1000
    assert config.simulation.arena_width == 10
    assert config.comms.communication_type == CommunicationType.DISCRETE


def test_save(tmp_path):
    test_from_yaml(tmp_path)
    config = Config.from_yaml(tmp_path)

    out = tmp_path / "out"
    out.mkdir()
    config.save(out)

    assert (out / "training.yaml").exists()
    assert (out / "comms.yaml").exists()