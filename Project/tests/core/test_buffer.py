import pytest
import torch
import math

from controller.marl.core.buffer import RolloutBuffer

MINIBATCH_SIZE = 5

@pytest.fixture
def buffer_config():
    return {
        "episodes": 2,
        "timesteps_per_run": 20,
        "num_worlds_per_run": 10,
        "num_agents": 2,
        "num_obs": 294,
        "comm_dim": 8,
        "num_comms": 2,
        "num_global_obs": 100,
        "hidden_state_size": 128,
        "num_codebooks": 0,
        "device": "cpu"
    }

def test_buffer_shapes(buffer_config):
    """Checking shapes of tensors in buffer"""

    buffer = RolloutBuffer(**buffer_config)

    T = buffer_config["timesteps_per_run"]
    N = buffer_config["num_agents"]
    O = buffer_config["num_obs"]
    
    expected = buffer_config["episodes"] * buffer_config["num_worlds_per_run"]
    
    assert buffer.pos == 0
    assert buffer.obs.shape == (expected, T, N, O)
    assert buffer.actions.shape == (expected, T, N)
    assert buffer.global_obs.shape == (expected, T, buffer_config["num_global_obs"])

def test_minibatches(buffer_config):
    """Checking minibatches cover all required data with the correct sizes"""

    buffer = RolloutBuffer(**buffer_config)
    
    buffer.pos = buffer.buffer_size 
    batches = list(buffer.get_minibatches(batch_size=MINIBATCH_SIZE))
    
    assert len(batches) == 4, f"Expected 4 batches, got {len(batches)}"
    
    for key in ["obs", "targets", "actions", "rewards", "action_log_probs", "advantages"]:
        assert key in batches[0], f"Key <{key}> is missing"
        
    assert batches[0]["obs"].shape[0] == MINIBATCH_SIZE

def test_minibatches_with_remainder(buffer_config):
    """Checking minibatches can handle non-perfect batch sizes"""
    
    buffer = RolloutBuffer(**buffer_config)
    
    buffer.pos = 17 
    
    batches = list(buffer.get_minibatches(batch_size=MINIBATCH_SIZE))
    
    assert len(batches) == math.ceil(17 / MINIBATCH_SIZE)
    assert batches[-1]["obs"].shape[0] == 2, "Remainder batch did not slice correctly"

def test_minibatches_shuffling(buffer_config):
    """Ensure minibatches are shuffled correctly"""

    buffer = RolloutBuffer(**buffer_config)
    buffer.pos = buffer.buffer_size

    for i in range(buffer.buffer_size):
        buffer.obs[i] = i
        
    batches = list(buffer.get_minibatches(batch_size=buffer.buffer_size))

    assert len(batches) == 1

    episode_ids = batches[0]["obs"][:, 0, 0, 0].to(torch.long)
    assert not torch.equal(episode_ids, torch.arange(buffer.buffer_size)), "The buffer generator is returning unshuffled data!"