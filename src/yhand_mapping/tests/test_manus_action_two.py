"""Verify Action 2 targets, signed feedforward currents and its stop command."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manus_calibration as calibration


@pytest.fixture(params=["left", "right"])
def controller(monkeypatch, request):
    sent = []
    monkeypatch.setattr(calibration.Node, "__init__", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "create_publisher",
                        lambda self, *args: SimpleNamespace(publish=sent.append))
    monkeypatch.setattr(calibration.Node, "create_subscription", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "create_timer", lambda self, *args: None)
    monkeypatch.setattr(calibration.Node, "get_logger",
                        lambda self: SimpleNamespace(info=lambda message: None))
    node = calibration.ManusCalibrationNode(request.param, "/fake_cmd", "/fake_feedback", 12.0)
    node.current_positions = [float(mid * 10) for mid in range(1, 29)]
    node._record_action1_positions()
    node.start_action("2")
    node._run_action_two(now=0.0, elapsed=0.0)
    return node, sent


def assert_commands(node, message, closed):
    ext, flex, shallow = {
        "left": ({25, 23, 10, 11}, {15, 2, 17, 13}, {3, 16}),
        "right": ({11, 9, 24, 25}, {1, 16, 3, 27}, {17, 2}),
    }[node.hand]
    assert len(message.motors) == 28
    for mid, motor in enumerate(message.motors, 1):
        idx = mid - 1
        assert motor.pos == (node.action1_recorded_positions[idx] if closed else 0.0)
        prefix = "action2_position_ctrl" if closed else "position_ctrl"
        assert motor.kp == getattr(node, prefix + "_kp")[idx]
        assert motor.kd == getattr(node, prefix + "_kd")[idx]
        assert motor.vel == getattr(node, prefix + "_vel")[idx]
        if mid in ext:
            expected = 0.2 if closed else -0.2
        elif mid in flex:
            expected = -0.15 if closed else 0.2
        elif not closed and mid in shallow:
            expected = 0.2
        else:
            expected = node.position_ctrl_tor[idx]
            if closed and mid in node.ring_position_ctrl_motors:
                expected *= calibration.ACTION2_RING_CLOSE_TOR_EXTRA_SCALE
        assert motor.tor == pytest.approx(expected)


def test_two_second_stages_repeat_with_exact_currents(controller):
    node, sent = controller
    for elapsed, closed in ((0.0, False), (1.999, False), (2.0, True),
                            (3.999, True), (4.0, False), (6.0, True)):
        node._run_action_two(now=elapsed, elapsed=elapsed)
        assert_commands(node, sent[-1], closed)


def test_stop_keeps_action_two_zero_command_without_changing_action_zero(controller):
    node, sent = controller
    assert node.request_stop_for_loop_actions()
    node._run_action_two(now=2.5, elapsed=2.5)
    assert node.current_action is None
    assert node.action_done_event.is_set()
    assert_commands(node, sent[-1], closed=False)
    node._publish_idle()
    assert_commands(node, sent[-1], closed=False)
    node.publish_all_zero_position()
    assert [motor.tor for motor in sent[-1].motors] == node.position_ctrl_tor


def test_missing_action_one_pose_aborts_to_original_zero_hold(controller):
    node, sent = controller
    node.has_action1_recorded_positions = False
    node.phase = "init"
    node._run_action_two(now=0.0, elapsed=0.0)
    assert node.current_action is None
    assert node.idle_mode == "hold_zero_pos"
    assert [motor.pos for motor in sent[-1].motors] == [0.0] * 28
    assert [motor.tor for motor in sent[-1].motors] == node.position_ctrl_tor
