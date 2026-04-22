import pytest

from controller.marl.core.visualiser import Visualiser, get_line_graph
from dummy_config import Config


class Tracker:
    def __init__(self):
        self.metrics = {"reward_mean": [1, 2, 3]}

    def get_average(self, key):
        return 2.5


def test_get_line_graph_empty():
    assert get_line_graph([], "reward_mean") == ""


def test_get_line_graph_same_values():
    data = [
        {"reward_mean": 1.0},
        {"reward_mean": 1.0},
        {"reward_mean": 1.0},
    ]
    assert get_line_graph(data, "reward_mean") == "   "


def test_get_line_graph_changes():
    data = [
        {"reward_mean": 1.0},
        {"reward_mean": 2.0},
        {"reward_mean": 3.0},
    ]
    result = get_line_graph(data, "reward_mean")

    assert len(result) == 3
    assert result != ""


def test_update_step_adds_history_and_status(monkeypatch):
    times = iter([10.0, 12.0, 15.0])

    monkeypatch.setattr(
        "controller.marl.core.visualiser.time.perf_counter",
        lambda: next(times),
    )

    config = Config()
    config.comms.communication_type = "DISCRETE"

    vis = Visualiser(config)

    tracker = Tracker()
    vis.update_step(1, 1, 10, "collecting", tracker, checkpoint=True)

    assert vis.curr_status == "collecting"
    assert len(vis.recent_history) == 1
    assert vis.recent_history[0]["reward_mean"] == 2.5


def test_update_step_records_timing(monkeypatch):
    times = iter([10.0, 12.0, 15.5])

    monkeypatch.setattr(
        "controller.marl.core.visualiser.time.perf_counter",
        lambda: next(times),
    )

    config = Config()
    config.comms.communication_type = "DISCRETE"

    vis = Visualiser(config)
    tracker = Tracker()

    vis.update_step(1, 1, 10, "collecting", tracker)
    vis.update_step(2, 2, 10, "training", tracker)

    assert vis.timings["collecting"] == [3.5]
    assert vis.curr_status == "training"


def test_stop_calls_live_stop():
    config = Config()
    config.comms.communication_type = "DISCRETE"

    vis = Visualiser(config)

    called = False

    def fake_stop():
        nonlocal called
        called = True

    vis.live.stop = fake_stop
    vis.stop()

    assert called is True