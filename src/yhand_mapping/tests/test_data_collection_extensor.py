"""Exercise staged sampling with motor feedback and a fake clock, no hardware."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))
import data_collection_main as dc


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Motors:
    def __init__(self, hand, clock, blocked=(), stale=False, nan_motor=None):
        self.hand = hand
        self.clock = clock
        self.blocked = set(blocked)
        self.stale = stale
        self.nan_motor = nan_motor
        self.positions = np.zeros(28)
        self.matrix = [[0.0] * 5 for _ in range(28)]
        self.commands = []

    def set_command_matrix(self, matrix):
        self.matrix = [list(row) for row in matrix]
        self.commands.append((self.clock.now, self.matrix))

    def snapshot(self):
        for mid, row in enumerate(self.matrix, 1):
            if row[3] > 0 and mid not in self.blocked:
                self.positions[mid - 1] = row[0]
        pos = self.positions.copy()
        if self.nan_motor is not None and self.matrix[self.nan_motor - 1][3] > 0:
            pos[self.nan_motor - 1] = float('nan')
        return dict(pos=pos, vel=np.zeros(28), tor=np.zeros(28),
                    kp=np.zeros(28), kd=np.zeros(28), glove48=np.zeros(48),
                    glove_skeleton176=np.zeros(176),
                    feedback_age=2.0 if self.stale else 0.0)


def execute(monkeypatch, hand='left', finger='index', role='extensor',
            blocked=(), stale=False, nan_motor=None, single=False,
            disable_assist=False, runtime=None, preuniform=False):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    monkeypatch.setattr(dc, 'ANTAGONIST_RELEASE_ENABLE', not disable_assist)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    ext = muscles[role]
    sway = muscles['adduction']
    targets = {ext: -80.0}
    if not single:
        targets[sway] = -60.0
        if finger == 'thumb':
            targets[muscles['opposition']] = -40.0
    node = Motors(hand, clock, blocked=blocked, stale=stale, nan_motor=nan_motor)
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    result = dc._execute_point(
        node=node, defaults_hand=defaults, finger_motor_set=dc._finger_motor_sets(hand),
        finger=finger, combo=SimpleNamespace(finger=finger), target_by_motor=targets,
        active_motors=list(targets), active_mode='full' if finger == 'thumb' else 'xy',
        point_tolerance=dc.POINT_ARRIVAL_TOL, point_id=1, nominal_total=2000,
        runtime_tick=(lambda: runtime(node, clock)) if runtime else None,
        preuniform=preuniform,
    )
    return node, result, defaults, targets


CASES = [(hand, finger, 'extensor') for hand in ('left', 'right')
         for finger in ('index', 'middle', 'ring', 'little')]
CASES += [(hand, 'thumb', role) for hand in ('left', 'right')
          for role in ('ext_short', 'ext_long')]

PREUNIFORM_CASES = [(hand, finger, role)
                    for hand, fingers in dc.HAND_MOTOR_MAP.items()
                    for finger, muscles in fingers.items()
                    for role in muscles if 'flex' in role or role in ('abduction', 'adduction')]


@pytest.mark.parametrize('hand,finger,role', CASES)
@pytest.mark.parametrize('blocked', [False, True])
def test_preuniform_extensor_releases_one_second_then_tensions_sway(monkeypatch, hand, finger, role, blocked):
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    active = muscles[role]
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role,
        single=True, preuniform=True, blocked={active} if blocked else ())
    assert result['motion_policy'] == 'extensor_first'
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    assert len(node.commands) == 3
    first = node.commands[0][1]
    zero_time, neutral = node.commands[1]
    assert 1.0 <= zero_time < 1.0 + dc.CONTROL_PERIOD_SEC
    passive = {mid for name, mid in muscles.items()
               if 'flex' in name or name in ('abduction', 'adduction')}
    sway = {muscles['abduction'], muscles['adduction']}
    for mid in passive:
        assert first[mid - 1] == [0, 0, 0.05, 0, 0]
        assert neutral[mid - 1] == [0, 0, -0.03 if mid in sway else 0.0, 0, 0]
    for mid in set(range(1, 29)) - passive:
        assert first[mid - 1] == neutral[mid - 1]
    assert neutral[active - 1][0] == targets[active]
    assert neutral[active - 1][3] == defaults['kp'][active - 1]
    assert node.commands[-1][1] == dc._build_idle_tor_matrix(defaults)
    assert result['assist_release_sec'] == 1.0
    assert result['assist_release_plan_sec'] == dict.fromkeys(passive, 1.0)
    if blocked:
        assert node.commands[-1][0] >= dc.EXTENSOR_FIRST_TIMEOUT_SEC
        assert result['timeout_stage'] == 'extensor_first'


@pytest.mark.parametrize('stop_at', [0.5, 1.5])
def test_preuniform_extensor_emergency_keeps_stop_command(monkeypatch, stop_at):
    active = dc.HAND_MOTOR_MAP['left']['index']['extensor']

    def emergency(node, clock):
        if clock.now >= stop_at:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}

    node, result, _, _ = execute(monkeypatch, single=True, preuniform=True,
                                blocked={active}, runtime=emergency)
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert len(node.commands) == (2 if stop_at < 1.0 else 3)
    assert result['assist_release_sec'] == min(1.0, result['move_elapsed_sec'])


@pytest.mark.parametrize('hand,finger,role', PREUNIFORM_CASES)
def test_preuniform_assist_releases_half_second_then_zero_current_until_arrival(monkeypatch, hand, finger, role):
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role,
                                            single=True, preuniform=True)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    assert result['reached'] and result['motion_policy'] == 'preuniform_assist'
    assert len(node.commands) == 3  # Release, passive current0, default tension for sampling.
    _, first = node.commands[0]
    active = muscles[role]
    assert first[active - 1][0] == targets[active] and first[active - 1][3] > 0
    ext_ids = {mid for name, mid in muscles.items() if name.startswith('ext')}
    if 'flex' in role:
        expected = dict.fromkeys(ext_ids, 0.05)
    else:
        expected = dict.fromkeys(ext_ids | {mid for name, mid in muscles.items() if 'flex' in name}, 0.0)
        opposite = 'adduction' if role == 'abduction' else 'abduction'
        expected[muscles[opposite]] = 0.05
    for mid, current in expected.items():
        assert first[mid - 1] == [0, 0, current, 0, 0]
    for mid in set(range(1, 29)) - {active} - set(expected):
        assert first[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    neutral_time, neutral = node.commands[1]
    assert 0.5 <= neutral_time < 0.5 + dc.CONTROL_PERIOD_SEC
    for mid in expected:
        assert neutral[mid - 1] == [0, 0, 0, 0, 0]
    for mid in set(range(1, 29)) - set(expected):
        assert neutral[mid - 1] == first[mid - 1]
    stop_time, stop = node.commands[-1]
    assert stop_time >= dc.ARRIVAL_CONFIRM_SEC
    assert stop == dc._build_idle_tor_matrix(defaults)
    assert result['assist_release_tor_plan'] == expected
    assert result['assist_release_sec'] == 0.5
    assert result['assist_release_plan_sec'] == {mid: 0.5 if current else 0.0
                                               for mid, current in expected.items()}
    assert result['result_sample_count'] > 0


@pytest.mark.parametrize('role', ['flex_deep', 'abduction', 'adduction'])
def test_preuniform_zero_current_holds_until_timeout_then_restores_tension(monkeypatch, role):
    active = dc.HAND_MOTOR_MAP['left']['index'][role]
    node, result, defaults, _ = execute(monkeypatch, role=role, single=True,
                                      preuniform=True, blocked={active})
    assert result['timeout'] and not result['reached']
    assert len(node.commands) == 3
    assert 0.5 <= node.commands[1][0] < 0.5 + dc.CONTROL_PERIOD_SEC
    assert result['assist_release_sec'] == 0.5
    assert node.commands[-1][0] >= dc.MOVE_TIMEOUT_SEC
    assert node.commands[-1][1] == dc._build_idle_tor_matrix(defaults)


@pytest.mark.parametrize('stop_at', [0.25, 0.75])
def test_preuniform_interruption_preserves_emergency_command_in_both_phases(monkeypatch, stop_at):
    def emergency(node, clock):
        if clock.now >= stop_at:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}

    node, result, _, _ = execute(monkeypatch, role='abduction', single=True,
                                preuniform=True, runtime=emergency)
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert len(node.commands) == (2 if stop_at < 0.5 else 3)
    assert result['assist_release_sec'] == min(result['move_elapsed_sec'], 0.5)
    assert not result['result_fb_rows']


@pytest.mark.parametrize('prefix', ['preuniform', 'manual'])
def test_only_preuniform_target_collection_selects_new_assist(monkeypatch, prefix):
    state = quota_state('index', 6)
    state['point_id'] = 0
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', 'left')
    node = Motors('left', Clock())

    class Executed(Exception):
        pass

    def capture(**kwargs):
        assert kwargs['preuniform'] == (prefix == 'preuniform')
        raise Executed

    monkeypatch.setattr(dc, '_execute_point', capture)
    with pytest.raises(Executed):
        dc._collect_one_manual_target_point(node, defaults, dc._finger_motor_sets('left'),
            state, SimpleNamespace(combo_id='index_0'), {15: -80.0}, 'x_only', None,
            {'global_point': 0}, {}, log_prefix=prefix)


@pytest.mark.parametrize('hand,finger,role', CASES)
def test_release_and_extensor_arrival_precede_remaining_targets(monkeypatch, hand, finger, role):
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    ext = muscles[role]
    released = {mid for muscle, mid in muscles.items()
                if 'flex' in muscle or muscle in ('adduction', 'abduction')}
    assert result['reached'] and not result['timeout']
    assert result['motion_policy'] == 'extensor_first'
    assert result['extensor_stage_reached']
    assert len(node.commands) == 3  # extender, remaining targets, default tension
    first_time, first = node.commands[0]
    second_time, second = node.commands[1]
    assert first_time == 0.0  # Immediate motion, no fixed prerelease wait.
    assert second_time >= dc.ARRIVAL_CONFIRM_SEC
    assert first[ext - 1][0] == second[ext - 1][0] == targets[ext]
    assert first[ext - 1][3] == second[ext - 1][3] == defaults['kp'][ext - 1]
    for mid in released:
        assert first[mid - 1] == [0, 0, 0.08, 0, 0]
        if mid in targets:
            assert second[mid - 1][0] == targets[mid] and second[mid - 1][3] > 0
        else:
            assert second[mid - 1] == [0, 0, 0.08, 0, 0]
    if finger == 'thumb':
        opp = muscles['opposition']
        assert first[opp - 1][0] == 0 and first[opp - 1][3] > 0
        assert second[opp - 1][0] == targets[opp]
    for mid in set(range(1, 29)) - set(targets) - released:
        assert first[mid - 1] == second[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    last_time, last = node.commands[-1]
    assert last_time - second_time >= dc.ARRIVAL_CONFIRM_SEC
    assert last == dc._build_idle_tor_matrix(defaults)
    assert result['target_cmd_matrix'][muscles['adduction'] - 1][0] == targets[muscles['adduction']]
    assert result['assist_release_tor_plan'] == {mid: 0.08 for mid in released}
    assert result['result_sample_count'] > 0


def test_extensor_timeout_does_not_start_sway_targets(monkeypatch):
    ext = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, defaults, targets = execute(monkeypatch, blocked=[ext])
    assert result['timeout'] and not result['reached']
    assert not result['extensor_stage_reached']
    assert result['timeout_stage'] == 'extensor_first'
    assert len(node.commands) == 2
    assert node.commands[-1][0] >= dc.EXTENSOR_FIRST_TIMEOUT_SEC
    assert node.commands[-1][1] == dc._build_idle_tor_matrix(defaults)


def test_remaining_targets_have_their_own_timeout(monkeypatch):
    sway = dc.HAND_MOTOR_MAP['left']['index']['adduction']
    node, result, _, _ = execute(monkeypatch, blocked=[sway])
    assert result['extensor_stage_reached']
    assert result['timeout'] and result['timeout_stage'] == 'remaining_targets'
    assert node.commands[-1][0] - node.commands[1][0] >= dc.MOVE_TIMEOUT_SEC


@pytest.mark.parametrize('gap', ['stale', 'out_of_tolerance'])
def test_confirmation_requires_uninterrupted_valid_extensor_feedback(monkeypatch, gap):
    original = Motors.snapshot

    def snapshot(node):
        result = original(node)
        if 0.3 <= node.clock.now <= 0.6:
            if gap == 'stale':
                result['feedback_age'] = 2.0
            else:
                result['pos'][dc.HAND_MOTOR_MAP['left']['index']['extensor'] - 1] = 0.0
        return result

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    node, result, _, _ = execute(monkeypatch)
    assert result['reached']
    assert node.commands[1][0] >= 0.6 + dc.ARRIVAL_CONFIRM_SEC


@pytest.mark.parametrize('bad_feedback', ['stale', 'nan'])
def test_invalid_extensor_feedback_never_advances_to_sway(monkeypatch, bad_feedback):
    ext = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, _, _ = execute(monkeypatch, stale=bad_feedback == 'stale',
                                  nan_motor=ext if bad_feedback == 'nan' else None)
    assert result['timeout_stage'] == 'extensor_first'
    assert not result['extensor_stage_reached']
    assert len(node.commands) == 2


def test_runtime_emergency_during_extensor_stage_keeps_stop_command(monkeypatch):
    def emergency(node, clock):
        if clock.now > 0.2:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}
    node, result, _, _ = execute(monkeypatch, runtime=emergency)
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert not result['extensor_stage_reached']
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert not result['result_fb_rows']


def test_single_extensor_presampling_also_releases_flexors_and_sway(monkeypatch):
    node, result, _, _ = execute(monkeypatch, single=True)
    assert result['extensor_stage_reached'] and result['reached']
    assert len(node.commands) == 2  # No other targets to move after the extensor.


@pytest.mark.parametrize('disabled', [False, True])
def test_flexor_points_and_disabled_assist_use_original_simultaneous_motion(monkeypatch, disabled):
    role = 'extensor' if disabled else 'flex_deep'
    node, result, _, targets = execute(monkeypatch, role=role, disable_assist=disabled)
    assert result['motion_policy'] == 'simultaneous'
    assert result['reached']
    first = node.commands[0][1]
    assert all(first[mid - 1][0] == target and first[mid - 1][3] > 0
               for mid, target in targets.items())


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_zero_return_releases_four_fingers_then_extensors_then_all_motors(monkeypatch, hand):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    node = Motors(hand, clock)
    initial = -np.arange(1, 29, dtype=float) * 3.0
    node.positions[:] = initial
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    result = dc._run_zero_calibration(node, defaults)
    assert result['success'] and result['attempts'] == 1
    released = {mid for finger, muscles in dc.HAND_MOTOR_MAP[hand].items() if finger != 'thumb'
                for role, mid in muscles.items()
                if 'flex' in role or role in ('abduction', 'adduction')}
    assert len(released) == 14  # Six flexors and eight sway channels, four fingers only.
    extensors = {25, 23, 10, 11} if hand == 'left' else {11, 9, 24, 25}
    extensor_time, extensor_matrix = node.commands[-2]
    zero_time, zero_matrix = node.commands[-1]
    assert 1.0 <= extensor_time < 1.0 + dc.CONTROL_PERIOD_SEC + 1e-6
    assert zero_time - extensor_time >= dc.ZERO_CALIBRATE_STABLE_SEC
    expected_zero = dc._build_zero_matrix(defaults)
    for mid in extensors:
        expected_zero[mid - 1][2] = -0.05
    assert zero_matrix == expected_zero
    assert clock.now - zero_time >= dc.ZERO_CALIBRATE_STABLE_SEC
    assert len(node.commands[:-2]) >= 49
    for elapsed, matrix in node.commands[:-2]:
        assert elapsed < 1.0
        for mid in released:
            assert matrix[mid - 1] == [0, 0, 0.08, 0, 0]
        for mid in set(range(1, 29)) - released:
            assert matrix[mid - 1] == [initial[mid - 1], defaults['vel'][mid - 1],
                                     defaults['tor'][mid - 1], defaults['kp'][mid - 1],
                                     defaults['kd'][mid - 1]]
    for mid in extensors:
        assert extensor_matrix[mid - 1] == expected_zero[mid - 1]
    for mid in released:
        assert extensor_matrix[mid - 1] == [0, 0, -0.03, 0, 0]
    held = set(range(1, 29)) - extensors - released
    assert len(held) == 10
    for mid in held:
        assert extensor_matrix[mid - 1][0] == initial[mid - 1]
        assert extensor_matrix[mid - 1][3] > 0


@pytest.mark.parametrize('feedback', ['missing', 'nan', 'stale'])
@pytest.mark.parametrize('hand', ['left', 'right'])
def test_prerelease_without_valid_feedback_uses_current_only(monkeypatch, feedback, hand):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    node = Motors(hand, clock)
    if feedback == 'missing':
        monkeypatch.setattr(node, 'snapshot', lambda: None)
    elif feedback == 'nan':
        node.positions[0] = float('nan')
    else:
        node.stale = True
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    dc._run_zero_prerelease(node, defaults)
    assert clock.now >= 1.0
    for _, matrix in node.commands:
        assert all(row[3] == row[4] == 0 for row in matrix)
        assert np.all(np.isfinite(matrix))


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_failed_zero_extensor_never_returns_other_motors(monkeypatch, hand):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    ext = dc.HAND_MOTOR_MAP[hand]['index']['extensor']
    node = Motors(hand, clock, blocked={ext})
    node.positions[:] = -100.0
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    result = dc._run_zero_calibration(node, defaults)
    assert not result['success'] and result['failed_stage'] == 'extensor_first'
    assert result['attempts'] == dc.ZERO_CALIBRATE_MAX_ATTEMPTS
    extensors = {dc.HAND_MOTOR_MAP[hand][finger]['extensor']
                 for finger in ('index', 'middle', 'ring', 'little')}
    for _, matrix in node.commands:
        assert all(row[0] == -100.0 for mid, row in enumerate(matrix, 1)
                   if mid not in extensors and row[3] > 0)
    assert node.commands[-1][1][ext - 1][0] == -100.0  # Stop at actual failed position.


@pytest.mark.parametrize('feedback', ['missing', 'nan', 'stale'])
def test_invalid_zero_feedback_never_starts_position_return(monkeypatch, feedback):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    node = Motors('left', clock, stale=feedback == 'stale')
    if feedback == 'missing':
        monkeypatch.setattr(node, 'snapshot', lambda: None)
    elif feedback == 'nan':
        node.positions[0] = float('nan')
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', 'left')
    result = dc._run_zero_calibration(node, defaults)
    assert not result['success']
    assert result['failed_stage'] == 'extensor_first'
    assert all(row[3] == row[4] == 0 for _, matrix in node.commands for row in matrix)
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_extensor_phase_tensions_flexors_and_holds_thumb_after_release(monkeypatch, hand):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    node = Motors(hand, clock)
    node.positions[:] = -100.0
    released = dc._zero_flexor_motor_ids(hand) + dc._zero_sway_motor_ids(hand)
    original_snapshot = node.snapshot
    released_once = False

    def snapshot():
        nonlocal released_once
        result = original_snapshot()
        if clock.now >= 1.0 and not released_once:
            node.positions[np.asarray(released) - 1] += 35.0
            result['pos'] = node.positions.copy()
            released_once = True
        return result

    monkeypatch.setattr(node, 'snapshot', snapshot)
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    assert dc._run_zero_calibration(node, defaults)['success']
    extensor_matrix = node.commands[-2][1]
    for mid in released:
        assert extensor_matrix[mid - 1] == [0, 0, -0.03, 0, 0]
    for mid in dc.HAND_MOTOR_MAP[hand]['thumb'].values():
        assert extensor_matrix[mid - 1][0] == -100.0


def test_full_zero_retry_keeps_extensors_zero_without_rewinding_other_motors(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    node = Motors('left', clock, blocked={15})  # Index deep flexor cannot return.
    node.positions[:] = -100.0
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', 'left')
    result = dc._run_zero_calibration(node, defaults)
    assert not result['success'] and result['failed_stage'] == 'all_motors'
    assert result['attempts'] == dc.ZERO_CALIBRATE_MAX_ATTEMPTS
    after_release = [(t, matrix) for t, matrix in node.commands if t >= 1.0]
    assert after_release[0][1][14] == [0, 0, -0.03, 0, 0]  # Current tension until extensors arrive.
    full_returns = after_release[1:-1]
    assert len(full_returns) == 3
    assert all(matrix[14][0] == 0.0 for _, matrix in full_returns)
    assert after_release[-1][1][14][0] == -100.0  # Failure holds measured position.


def quota_state(finger, count):
    combos = [SimpleNamespace(combo_id=f'{finger}_{i}', opposition_mode='on' if i % 2 else 'off')
              for i in range(count)]
    total, lower, upper, targets = dc._build_finger_quota_plan(finger, combos)
    keys, _, mode_min, mode_max = dc._build_active_mode_quota_plan(finger, total)
    return {'finger': finger, 'nominal_total': total, 'combo_min': lower, 'combo_max': upper,
            'combo_target': targets, 'combo_counts': dict.fromkeys(lower, 0),
            'mode_min': mode_min, 'mode_max': mode_max, 'mode_counts': dict.fromkeys(keys, 0),
            'combo_cooldown': dict.fromkeys(lower, 0), 'done': False}


@pytest.mark.parametrize('selection,expected', [
    ('middle,ring,little', ['middle', 'ring', 'little']),
    (' LITTLE , middle ,ring ', ['little', 'middle', 'ring']),
    ('middle,ring,middle,little', ['middle', 'ring', 'little']),
    ('ring', ['ring']),
    ('all', dc.FINGER_ORDER_FIXED),
    ('*', dc.FINGER_ORDER_FIXED),
    ('', dc.FINGER_ORDER_FIXED),
])
def test_target_fingers_preserve_requested_order(selection, expected):
    assert dc._normalize_target_fingers(selection) == expected


@pytest.mark.parametrize('selection', ['middle,bogus', 'all,ring', 'middle,,ring', ',ring', 'ring,'])
def test_invalid_finger_lists_are_rejected(selection):
    with pytest.raises(ValueError, match='Unsupported target_finger'):
        dc._normalize_target_fingers(selection)


@pytest.mark.parametrize('finger,count', [('index', 6), ('middle', 6), ('ring', 4), ('little', 4)])
def test_four_finger_quota_is_1000_per_combo(finger, count):
    state = quota_state(finger, count)
    assert state['nominal_total'] == count * 1000
    assert set(state['combo_target'].values()) == {1000.0}
    assert set(state['combo_min'].values()) == {900}
    assert set(state['combo_max'].values()) == {1100}
    assert state['mode_min']['xy'] == count * 900
    assert state['mode_max']['xy'] == count * 1100


def test_thumb_quota_and_opposition_split_are_unchanged():
    state = quota_state('thumb', 12)
    assert state['nominal_total'] == 24000
    assert state['combo_target']['thumb_1'] == 2800.0
    assert state['combo_target']['thumb_0'] == 1200.0
    assert state['mode_min']['full'] == 21600
    assert state['mode_max']['full'] == 26400


@pytest.mark.parametrize('legacy', [True, False])
def test_old_quota_checkpoint_preserves_samples_but_uses_new_limits(legacy):
    state = quota_state('index', 6)
    payload = {'point_id': 6000, 'combo_counts': dict.fromkeys(state['combo_counts'], 1000),
               'mode_counts': {'xy': 6000}, 'combo_min': dict.fromkeys(state['combo_min'], 1800),
               'combo_max': dict.fromkeys(state['combo_max'], 2200),
               'mode_min': {'xy': 10800}, 'mode_max': {'xy': 13200},
               'combo_cooldown': dict.fromkeys(state['combo_counts'], 60),
               'preuniform_done': True, 'done': True}
    if not legacy:
        payload['quota_base_per_combo'] = 2000
    dc._apply_finger_resume_payload(state, payload)
    assert state['point_id'] == 6000 and state['preuniform_done']
    assert set(state['combo_counts'].values()) == {1000}
    assert set(state['combo_min'].values()) == {900}
    assert set(state['combo_max'].values()) == {1100}
    assert set(state['combo_cooldown'].values()) == {0}
    assert not state['done']  # Main loop must reassess the reduced quota.
    assert dc._finger_done(state['combo_counts'], state['combo_min'], state['combo_max'],
                           state['nominal_total'], state['mode_counts'],
                           state['mode_min'], state['mode_max'])


@pytest.mark.parametrize('finger,count', [('index', 6), ('thumb', 12)])
def test_unchanged_quota_checkpoint_preserves_frozen_limits_and_completion(finger, count):
    state = quota_state(finger, count)
    cid = next(iter(state['combo_counts']))
    state['combo_counts'][cid] = 700
    state['combo_min'][cid] = state['combo_max'][cid] = 700
    state['done'] = True
    payload = dc._finger_state_to_resume_payload(state)
    assert payload['quota_base_per_combo'] == (2000 if finger == 'thumb' else 1000)
    if finger == 'thumb':
        payload.pop('quota_base_per_combo')  # Legacy thumb checkpoints also remain valid.
    restored = quota_state(finger, count)
    dc._apply_finger_resume_payload(restored, payload)
    assert restored['done']
    assert restored['combo_counts'][cid] == 700
    assert restored['combo_min'][cid] == restored['combo_max'][cid] == 700
