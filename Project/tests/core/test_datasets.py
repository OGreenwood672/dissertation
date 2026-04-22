import numpy as np
import torch

from controller.marl.core.datasets import FilteredObsData, ObsData

def test_filtered_obs_data(tmp_path):

    data = np.array([
        [1, 2, 3, 4, 10, 11, 1, 0.1, 0.2, 0.3, 0.9, 1.5],
        [5, 6, 7, 8, 12, 13, 0, 0.4, 0.5, 0.6, 1.1, 2.0],
    ], dtype=np.float32)

    file = tmp_path / "filtered.csv"
    np.savetxt(file, data, delimiter=",", header="c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12")

    mask = torch.tensor([True, False, True, False])
    ds = FilteredObsData(obs_log_path=file, obs_count=4, global_obs_count=2, mask=mask, device="cpu")

    assert len(ds) == 2

    filtered_obs, global_obs, actions, targets, critic_value, return_value = ds[0]
    assert torch.equal(filtered_obs, torch.tensor([1.0, 3.0]))
    assert torch.equal(global_obs, torch.tensor([10.0, 11.0]))
    assert actions.dtype == torch.long
    assert torch.equal(actions, torch.tensor([0]))
    assert torch.equal(targets, torch.tensor([0.1, 0.2, 0.3]))
    assert torch.equal(critic_value, torch.tensor([0.9]))
    assert torch.equal(return_value, torch.tensor([1.5]))


def test_obs_data(tmp_path):

    raw = np.array([
        [1, 2, 10, 11, 0, 0.1, 0.2, 0.3, 0.9, 1.0],
        [3, 4, 12, 13, 1, 0.4, 0.5, 0.6, 1.1, 2.0],
        [5, 6, 14, 15, 0, 0.7, 0.8, 0.9, 1.2, 3.0],
        [7, 8, 16, 17, 1, 1.0, 1.1, 1.2, 1.3, 4.0],
    ], dtype=np.float32)

    file = tmp_path / "obs.csv"
    np.savetxt(file, raw, delimiter=",", comments="")

    ds = ObsData(obs_log_path=file, global_obs_count=2, T=2, W=2, N=1, device="cpu")

    assert len(ds) == 2

    obs, global_obs, actions, targets, critic_value, return_value = ds[0]

    assert obs.shape == (2, 1, 2)
    assert global_obs.shape == (2, 1, 2)
    assert actions.shape == (2, 1, 1)
    assert targets.shape == (2, 1, 3)
    assert critic_value.shape == (2, 1, 1)
    assert return_value.shape == (2, 1, 1)

    assert obs.dtype == torch.float32
    assert global_obs.dtype == torch.float32
    assert actions.dtype == torch.long
    assert targets.dtype == torch.float32
    assert critic_value.dtype == torch.float32
    assert return_value.dtype == torch.float32