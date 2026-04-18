import pytest

from controller.marl.core.metric_tracker import MetricTracker

@pytest.fixture
def tracker():
    return MetricTracker()

def test_initial(tracker):
    assert tracker.get_average('fake_key') == ""

def test_update(tracker):
    tracker.update('test', 10)
    assert tracker.get_average('test') == 10.0

def test_update_multiple(tracker):
    tracker.update('test', 10)
    tracker.update('test', 20)
    assert tracker.get_average('test') == 15.0

def test_different_keys(tracker):
    tracker.update('test', 5)
    tracker.update('test2', 10)
    assert tracker.get_average('test') == 5.0
    assert tracker.get_average('test2') == 10.0

def test_reset(tracker):
    tracker.update('test', 10)
    tracker.reset_tracker()
    assert tracker.get_average('test') == ""