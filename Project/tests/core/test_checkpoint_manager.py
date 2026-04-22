from controller.marl.core.checkpoint_manager import CheckpointManager
from dummy_config import Config


def test_get_folder_name():
    name = CheckpointManager.get_folder_name(7)

    assert name.endswith("_seed_7")
    assert "seed_7" in name


def test_init_existing_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.RESULTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.makedirs",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        lambda path: ["2026-04-19_12-00-00_seed_1"],
    )

    manager = CheckpointManager(Config())

    assert manager.get_seed() == 1
    assert manager.is_new_run is False
    assert manager.get_result_path() == tmp_path / "DISCRETE" / "2026-04-19_12-00-00_seed_1"


def test_init_new_seed_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.RESULTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.makedirs",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        lambda path: [],
    )

    called = {"made": False}

    def fake_gen_result_folder(self, result_path, config):
        called["made"] = True

    def fake_get_folder_name(self, seed):
        return f"folder_seed_{seed}"

    monkeypatch.setattr(CheckpointManager, "gen_result_folder", fake_gen_result_folder)
    monkeypatch.setattr(CheckpointManager, "get_folder_name", fake_get_folder_name)

    manager = CheckpointManager(Config())

    assert manager.get_seed() == 1
    assert manager.is_new_run is True
    assert called["made"] is True
    assert manager.get_result_path() == tmp_path / "DISCRETE" / "folder_seed_1"


def test_load_checkpoint_models_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.RESULTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.makedirs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        lambda path: [],
    )

    def fake_gen_result_folder(self, path, config):
        pass

    def fake_get_folder_name(self, seed):
        return f"folder_seed_{seed}"

    monkeypatch.setattr(CheckpointManager, "gen_result_folder", fake_gen_result_folder)
    monkeypatch.setattr(CheckpointManager, "get_folder_name", fake_get_folder_name)

    manager = CheckpointManager(Config())
    manager.result_path = tmp_path / "DISCRETE" / "folder_seed_1"

    def fake_listdir(path):
        return ["checkpoint_5.pth", "checkpoint_12.pth"]

    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        fake_listdir,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.torch.load",
        lambda path: {
            "actor": "actor_state",
            "critic": "critic_state",
            "actor_optimizer": "actor_opt",
            "critic_optimizer": "critic_opt",
        },
    )

    actor, critic, actor_opt, critic_opt, step = manager.load_checkpoint_models()

    assert actor == "actor_state"
    assert critic == "critic_state"
    assert actor_opt == "actor_opt"
    assert critic_opt == "critic_opt"
    assert step == 12


def test_save_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.RESULTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.makedirs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        lambda path: [],
    )

    def fake_gen_result_folder(self, path, config):
        pass

    def fake_get_folder_name(self, seed):
        return f"folder_seed_{seed}"

    monkeypatch.setattr(CheckpointManager, "gen_result_folder", fake_gen_result_folder)
    monkeypatch.setattr(CheckpointManager, "get_folder_name", fake_get_folder_name)

    manager = CheckpointManager(Config())
    manager.result_path = tmp_path / "DISCRETE" / "folder_seed_1"

    saved = {}

    def fake_save(state, path):
        saved["state"] = state
        saved["path"] = path

    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.torch.save",
        fake_save,
    )

    class Model:
        def state_dict(self):
            return {"weights": 1}

    class Optimiser:
        def state_dict(self):
            return {"lr": 0.001}

    manager.save_checkpoint(Model(), Model(), Optimiser(), Optimiser(), 20)

    assert saved["state"]["step"] == 20
    assert saved["path"] == (
        tmp_path / "DISCRETE" / "folder_seed_1" / "checkpoints" / "checkpoint_20.pth"
    )


def test_get_seed_and_result_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.RESULTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.makedirs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "controller.marl.core.checkpoint_manager.os.listdir",
        lambda path: [],
    )

    def fake_gen_result_folder(self, path, config):
        pass

    def fake_get_folder_name(self, seed):
        return f"folder_seed_{seed}"

    monkeypatch.setattr(CheckpointManager, "gen_result_folder", fake_gen_result_folder)
    monkeypatch.setattr(CheckpointManager, "get_folder_name", fake_get_folder_name)

    manager = CheckpointManager(Config())

    assert manager.get_seed() == 1
    assert manager.get_result_path() == tmp_path / "DISCRETE" / "folder_seed_1"