"""Exercise real Action 3 commands with ROS transport replaced by a recorder."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manus_calibration as calibration


@pytest.fixture
def controller(monkeypatch):
    sent = []
    monkeypatch.setattr(calibration.Node, "__init__", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "create_publisher",
                        lambda self, *args: SimpleNamespace(publish=sent.append))
    monkeypatch.setattr(calibration.Node, "create_subscription", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "create_timer", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "get_logger",
                        lambda self: SimpleNamespace(info=lambda message: None))

    def make(hand="left"):
        node = calibration.ManusCalibrationNode(hand, "/fake_cmd", "/fake_feedback", 12.0)
        node.current_positions = [float(mid * 10) for mid in range(1, 29)]
        node.current_velocities = [0.0] * 28
        node.start_action("3")
        node._run_action_three(now=0.0, elapsed=0.0)
        return node, sent

    return make


def test_opposition_currents_and_all_four_finger_position_holds(controller):
    node, sent = controller()
    initial = list(node.current_positions)
    assert node.action3_hold_positions == initial
    node.current_positions = [value + 7.0 for value in initial]
    node._run_action_three(now=2.399, elapsed=2.399)
    currents = {14: -0.25, 27: -0.2, 4: -0.25, 6: 0.1, 7: 0.1, 8: 0.1}
    locked = {1, 2, 3, 9, 10, 11, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 28}
    assert node.action3_lock_motors == locked
    assert len(sent[-1].motors) == 28
    for mid, motor in enumerate(sent[-1].motors, 1):
        idx = mid - 1
        if mid in locked:
            assert motor.pos == initial[idx]
            assert motor.kp == node.position_ctrl_kp[idx]
            assert motor.kd == node.position_ctrl_kd[idx]
            assert motor.tor == node.position_ctrl_tor[idx]
        else:
            assert motor.tor == pytest.approx(currents.get(mid, -0.01))
            assert motor.kp == motor.kd == 0.0


def assert_initial_position_command(node, message):
    assert len(message.motors) == 28
    for idx, motor in enumerate(message.motors):
        assert motor.pos == node.action3_hold_positions[idx]
        assert motor.kp == node.position_ctrl_kp[idx]
        assert motor.kd == node.position_ctrl_kd[idx]
        assert motor.tor == node.position_ctrl_tor[idx]


def test_returns_for_one_second_then_repeats_without_overwriting_initial_pose(controller):
    node, sent = controller()
    initial = list(node.action3_hold_positions)
    node.current_positions = [value + 7.0 for value in initial]
    for elapsed in (2.4, 3.399):
        node._run_action_three(now=elapsed, elapsed=elapsed)
        assert_initial_position_command(node, sent[-1])
    node._run_action_three(now=3.4, elapsed=3.4)
    assert sent[-1].motors[13].tor == pytest.approx(-0.25)
    assert sent[-1].motors[13].kp == 0.0
    assert node.action3_hold_positions == initial


@pytest.mark.parametrize("elapsed", [1.0, 2.7])
def test_stop_during_either_phase_returns_and_holds_initial_pose(controller, elapsed):
    node, sent = controller()
    node.current_positions = [value + 7.0 for value in node.current_positions]
    assert node.request_stop_for_loop_actions()
    node._run_action_three(now=elapsed, elapsed=elapsed)
    assert_initial_position_command(node, sent[-1])
    assert node.current_action is None
    assert node.action_done_event.is_set()
    assert node.idle_mode == "hold_action3"
    node._publish_idle()
    assert_initial_position_command(node, sent[-1])


def test_right_hand_opposition_currents_are_unchanged(controller):
    node, sent = controller("right")
    assert node.action3_thumb_currents == {28: -0.45, 13: -0.5, 18: -0.45}
    for mid in (20, 21, 22):
        assert sent[-1].motors[mid - 1].tor == pytest.approx(-0.01)
        assert sent[-1].motors[mid - 1].kp == sent[-1].motors[mid - 1].kd == 0.0
