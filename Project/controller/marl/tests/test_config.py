import pytest
import yaml
from controller.marl.config import load_config

def test_load_config_success(tmp_path):
    """
    Tests that a valid YAML configuration file is loaded correctly.
    """
    config_data = {
        'hyperparameters': {
            'learning_rate': 0.001,
            'gamma': 0.9
        },
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    loaded_config = load_config(str(config_file))
    assert loaded_config == config_data

def test_load_config_file_not_found():
    """
    Tests that a FileNotFoundError is raised when the config file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file.yaml")

def test_load_config_invalid_yaml(tmp_path):
    """Tests that a YAMLError is raised for a malformed YAML file."""
    invalid_content = "setting1: value1\n  - invalid_indent:"
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(invalid_content)

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))