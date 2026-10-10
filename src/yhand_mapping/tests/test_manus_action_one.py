"""Check Action 1 timing and all 28 commands without creating ROS nodes."""
import sys
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manus_calibration as calibration


def controller(hand):
    sent = []
    motors = calibration.HAND_MOTORS_BY_SIDE[hand]
    fist, _, _, _ = calibration._build_action_currents(motors)
    state = SimpleNamespace(
        phase="init", phase_start_time=0.0, current_action="1", stop_requested=False,
        _state_lock=threading.Lock(), action_done_event=threading.Event(),
        current_positions=[float(mid) for mid in range(1, 29)],
        current_velocities=[0.0] * 28, has_action1_recorded_positions=False,
        fist_current=fist, fist_hold_current=calibration._build_fist_hold_currents(motors),
        cmd_publisher=SimpleNamespace(publish=sent.append),
        get_logger=lambda: SimpleNamespace(info=lambda message: None),
        _log_phase_once=lambda message: None,
    )
    for name in ("publish_currents", "_run_action_one", "_record_action1_positions",
                 "_finish_action", "_publish_idle"):
        setattr(state, name, MethodType(getattr(calibration.ManusCalibrationNode, name), state))
    return state, sent


@pytest.mark.parametrize("hand", ["left", "right"])
def test_pretension_lasts_half_a_second(hand):
    state, sent = controller(hand)
    state._run_action_one(now=0.499, elapsed=0.499)
    assert state.phase == "init"
    assert [motor.tor for motor in sent[-1].motors] == [-0.01] * 28
    state._run_action_one(now=0.5, elapsed=0.5)
    assert state.phase == "fist"
    assert state.phase_start_time == 0.5
    assert all(motor.kp == motor.kd == 0.0 for motor in sent[-1].motors)


@pytest.mark.parametrize("hand,active", [("left", {15, 2, 17, 13}), ("right", {1, 16, 3, 27})])
def test_fist_pulls_only_deep_flexors_and_releases_all_other_channels(hand, active):
    state, sent = controller(hand)
    state.phase = "fist"
    state._run_action_one(now=5.499, elapsed=4.999)
    assert not state.action_done_event.is_set()
    assert len(sent[-1].motors) == 28
    for mid, motor in enumerate(sent[-1].motors, 1):
        assert motor.tor == pytest.approx(-0.2 if mid in active else 0.05)
        assert motor.kp == motor.kd == 0.0
        assert motor.pos == float(mid)


@pytest.mark.parametrize("hand,hold", [
    ("left", {15: -0.09, 3: -0.06, 2: -0.09, 17: -0.12, 13: -0.09}),
    ("right", {1: -0.09, 17: -0.06, 16: -0.09, 3: -0.12, 27: -0.09}),
])
def test_after_five_seconds_records_all_positions_and_keeps_original_hold(hand, hold):
    state, sent = controller(hand)
    state.phase = "fist"
    state._run_action_one(now=5.5, elapsed=5.0)
    assert state.action_done_event.is_set()
    assert state.current_action is None
    assert state.has_action1_recorded_positions
    assert state.action1_recorded_positions == [float(mid) for mid in range(1, 29)]
    assert state.idle_mode == "hold_fist"
    state._publish_idle()
    for mid, motor in enumerate(sent[-1].motors, 1):
        assert motor.tor == pytest.approx(hold.get(mid, -0.01))
        assert motor.kp == motor.kd == 0.0
