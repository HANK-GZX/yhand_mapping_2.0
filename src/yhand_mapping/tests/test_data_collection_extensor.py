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
            disable_assist=False, runtime=None, preuniform=False,
            include_stable=False, opposition=True, sway_role='adduction', opposition_target=-40.0):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    monkeypatch.setattr(dc, 'ANTAGONIST_RELEASE_ENABLE', not disable_assist)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    ext = muscles[role]
    sway = muscles[sway_role]
    targets = {ext: -80.0}
    if not single:
        targets[sway] = -60.0
        if finger == 'thumb' and opposition:
            targets[muscles['opposition']] = float(opposition_target)
    node = Motors(hand, clock, blocked=blocked, stale=stale, nan_motor=nan_motor)
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    result = dc._execute_point(
        node=node, defaults_hand=defaults, finger_motor_set=dc._finger_motor_sets(hand),
        finger=finger, combo=SimpleNamespace(finger=finger), target_by_motor=targets,
        active_motors=list(targets), active_mode='full' if finger == 'thumb' else 'xy',
        point_tolerance=dc.POINT_ARRIVAL_TOL, point_id=1, nominal_total=2000,
        runtime_tick=(lambda: runtime(node, clock)) if runtime else None,
        preuniform=preuniform,
        process_include_post_stable=include_stable,
    )
    if finger == 'thumb' and not result['interrupted']:
        for _, matrix in node.commands:
            for other in ('index', 'middle', 'ring', 'little'):
                for mid in dc.HAND_MOTOR_MAP[hand][other].values():
                    assert matrix[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    return node, result, defaults, targets


def expected_post_matrix(hand, finger, defaults, thumb_combo=True):
    protected = {mid for other in ('index', 'middle', 'ring', 'little')
                 for mid in dc.HAND_MOTOR_MAP[hand][other].values()} if finger == 'thumb' else set()
    if finger == 'thumb' and thumb_combo:
        return [[0, 0, -0.1 if mid in dc.HAND_MOTOR_MAP[hand]['thumb'].values()
                 else defaults['tor'][mid - 1], 0, 0] for mid in range(1, 29)]
    return [[0, 0, defaults['tor'][mid - 1] if mid in protected else -0.08, 0, 0]
            for mid in range(1, 29)]


THUMB_COMBO_CASES = [(hand, role, sway, opp) for hand in ('left', 'right')
                    for role in ('ext_short', 'ext_long', 'flex_long')
                    for sway in ('adduction', 'abduction') for opp in ('on', 'off')]


@pytest.mark.parametrize('hand,role,sway_role,opposition_mode', THUMB_COMBO_CASES)
@pytest.mark.parametrize('failure', [None, 'sway', 'primary', 'opposition'])
def test_thumb_combos_move_sway_first_then_hold_passive_current_and_record_one_second(
        monkeypatch, hand, role, sway_role, opposition_mode, failure):
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']
    sway, primary, opposition = muscles[sway_role], muscles[role], muscles['opposition']
    active = {sway, primary, opposition}
    passive = set(muscles.values()) - active
    other_than_sway = set(muscles.values()) - {sway}
    blocked = {'sway': sway, 'primary': primary, 'opposition': opposition}
    trace = []
    original_snapshot = Motors.snapshot

    def snapshot(node):
        snap = original_snapshot(node)
        # Delay side movement so release->zero has a real waiting interval.
        if node.matrix[sway - 1][3] > 0:
            if node.clock.now < 0.4:
                snap['pos'][sway - 1] = node.positions[sway - 1] = 0.0
            trace.append((node.clock.now, [row[:] for row in node.matrix]))
        snap['tor'] = np.asarray([row[2] for row in node.matrix])
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    node, result, defaults, targets = execute(
        monkeypatch, hand, 'thumb', role, sway_role=sway_role,
        opposition_target=-40.0 if opposition_mode == 'on' else 40.0,
        blocked={blocked[failure]} if failure else (), include_stable=True)
    assert result['motion_policy'] == 'sway_first'
    assert result['reached'] == (failure is None)
    assert result['timeout'] == (failure is not None)
    assert result['sway_stage_reached'] == (failure != 'sway')
    assert result['extensor_stage_reached'] is None
    assert result['timeout_stage'] == ('sway_first' if failure == 'sway' else
                                       'remaining_targets' if failure else '')
    first = node.commands[0][1]
    assert first[sway - 1] == [targets[sway], defaults['vel'][sway - 1],
                              defaults['tor'][sway - 1], defaults['kp'][sway - 1],
                              defaults['kd'][sway - 1]]
    phase1 = [(t, matrix) for t, matrix in trace if matrix[primary - 1][3] == 0]
    phase2 = [(t, matrix) for t, matrix in trace if matrix[primary - 1][3] > 0]
    assert phase1 and any(t >= 1.0 for t, _ in phase1)
    for t, matrix in phase1:
        for mid in other_than_sway:
            assert matrix[mid - 1] == [0, 0, 0.08 if t < 1.0 else 0.0, 0, 0]
    assert bool(phase2) == (failure != 'sway')
    for _, matrix in phase2:
        for mid, target in targets.items():
            assert matrix[mid - 1] == [target, defaults['vel'][mid - 1],
                                      -0.08 if mid == primary and role.startswith('ext') else defaults['tor'][mid - 1],
                                      defaults['kp'][mid - 1], defaults['kd'][mid - 1]]
        for mid in passive:
            if role == 'flex_long' and mid in (muscles['ext_short'], muscles['ext_long']):
                assert matrix[mid - 1][:2] == [0, 0]
                assert matrix[mid - 1][3:] == [0, 0]
                assert matrix[mid - 1][2] in (0.05, 0.02)
            else:
                assert matrix[mid - 1] == [0, 0, 0.02, 0, 0]
    if phase2 and role == 'flex_long':
        for mid in (muscles['ext_short'], muscles['ext_long']):
            currents = [matrix[mid - 1][2] for _, matrix in phase2]
            assert currents[0] == 0.05
            if failure == 'opposition':
                assert currents[-1] == 0.02
                assert currents == sorted(currents, reverse=True)
            else:
                assert all(current == 0.05 for current in currents)
    if failure == 'sway':
        assert result['move_elapsed_sec'] == pytest.approx(3.0, abs=dc.CONTROL_PERIOD_SEC)
    elif failure:
        assert result['move_elapsed_sec'] - result['sway_stage_elapsed_sec'] == pytest.approx(
            3.0, abs=dc.CONTROL_PERIOD_SEC)
    else:
        assert result['move_elapsed_sec'] >= 2.4
    assert result['assist_release_tor_plan'] == dict.fromkeys(other_than_sway, 0.08)
    assert result['assist_release_plan_sec'] == dict.fromkeys(other_than_sway, 1.0)
    tighten_start, tighten = node.commands[-3]
    settle_start, settle = node.commands[-2]
    record_start, record = node.commands[-1]
    assert tighten_start == pytest.approx(result['move_elapsed_sec'])
    assert tighten == settle == record == expected_post_matrix(hand, 'thumb', defaults)
    tighten_sec = 3.0 if role == 'flex_long' else 2.0
    assert settle_start - tighten_start == pytest.approx(tighten_sec)
    assert result['post_arrival_tighten_elapsed_sec'] == pytest.approx(tighten_sec)
    assert result['position_stable_confirmed']
    assert result['position_stable_motor_ids'] == sorted(muscles.values())
    assert record_start - settle_start == pytest.approx(2.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)
    assert node.clock.now - record_start == pytest.approx(1.0, abs=0.0051)
    assert result['result_window_label'] == '0.0-1.0'
    assert 29 <= result['result_sample_count'] <= 31
    assert min(row[0] for row in result['result_fb_rows']) == pytest.approx(0.0)
    assert all(0 <= row[0] <= 1.0 for row in result['result_fb_rows'])
    assert all(row[3::5] == [command[2] for command in record] for row in result['result_fb_rows'])
    assert max(row[0] for row in result['process_fb_rows']) == pytest.approx(
        record_start + 1.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)


@pytest.mark.parametrize('kind', ['emergency', 'pause_hold', 'stop'])
@pytest.mark.parametrize('phase,stop_at', [('release', 0.5), ('zero', 1.5),
                                         ('remaining', 1.5), ('tighten', 4.5), ('settle', 6.5)])
def test_thumb_sway_first_interruption_preserves_commands_in_each_phase(monkeypatch, kind, phase, stop_at):
    muscles = dc.HAND_MOTOR_MAP['left']['thumb']
    blocked = muscles['adduction'] if phase == 'zero' else muscles['ext_short']

    def interrupt(node, clock):
        if clock.now >= stop_at:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': kind}

    node, result, _, _ = execute(
        monkeypatch, 'left', 'thumb', 'ext_short', blocked={blocked}, runtime=interrupt)
    assert result['interrupted'] and result['interrupt_reason'] == kind
    assert node.commands[-1][1] == [[0] * 5 for _ in range(28)]
    assert result['result_window_label'] == '0.0-1.0'
    assert not result['result_fb_rows'] and not result['result_glove_rows']


def test_thumb_combo_keeps_tightening_until_passive_motor_stops_before_recording(monkeypatch):
    settle_start, record_start = [], []
    original_wait = dc._wait_finger_position_stable
    original_collect = dc._collect_window_samples
    original_snapshot = Motors.snapshot
    passive = dc.HAND_MOTOR_MAP['left']['thumb']['ext_long']

    def wait(*args, **kwargs):
        settle_start.append(dc.time.monotonic())
        return original_wait(*args, **kwargs)

    def collect(**kwargs):
        record_start.append(dc.time.monotonic())
        return original_collect(**kwargs)

    def snapshot(node):
        snap = original_snapshot(node)
        if settle_start and not record_start:
            assert node.matrix[passive - 1] == [0, 0, -0.1, 0, 0]
            snap['pos'][passive - 1] = 100.0 * min(node.clock.now - settle_start[0], 3.0)
        snap['glove48'][:] = 7.0 if record_start else -99.0
        return snap

    monkeypatch.setattr(dc, '_wait_finger_position_stable', wait)
    monkeypatch.setattr(dc, '_collect_window_samples', collect)
    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    node, result, _, _ = execute(monkeypatch, 'left', 'thumb', 'flex_long', include_stable=True)
    assert result['position_stable_confirmed'] and result['position_stable_wait_sec'] >= 4.9
    assert record_start[0] - settle_start[0] == pytest.approx(result['position_stable_wait_sec'])
    assert node.clock.now - record_start[0] == pytest.approx(1.0, abs=0.0051)
    assert result['result_glove_avg'] == pytest.approx([7.0] * 48)
    assert max(row[0] for row in result['process_fb_rows']) == pytest.approx(
        record_start[0] + 1.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('scenario', ['late_arrival', 'transient', 'stale', 'nan'])
def test_thumb_flexor_extensor_release_stops_only_after_individual_arrival(monkeypatch, hand, scenario):
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']
    flexor, opposition = muscles['flex_long'], muscles['opposition']
    extensors = (muscles['ext_short'], muscles['ext_long'])
    side = muscles['adduction']
    original_snapshot = Motors.snapshot
    stage_start, trace = [], []
    sustained_from = 0.8 if scenario == 'transient' else 0.6

    def snapshot(node):
        snap = original_snapshot(node)
        if node.matrix[flexor - 1][3] > 0:
            if not stage_start:
                stage_start.append(node.clock.now)
            elapsed = node.clock.now - stage_start[0]
            if scenario in ('late_arrival', 'transient'):
                transient = scenario == 'transient' and 0.3 <= elapsed < 0.4
                if elapsed < sustained_from and not transient:
                    node.positions[flexor - 1] = snap['pos'][flexor - 1] = 0.0
            elif 0.3 <= elapsed < 0.6:
                if scenario == 'stale':
                    snap['feedback_age'] = 2.0
                else:
                    snap['pos'][flexor - 1] = float('nan')
            trace.append((node.clock.now, elapsed, snap['pos'][flexor - 1],
                          [node.matrix[mid - 1][:] for mid in extensors]))
            # The opposite sway keeps the normal +0.02 A hold throughout phase 2.
            opposite_side = muscles['abduction']
            assert node.matrix[opposite_side - 1] == [0, 0, 0.02, 0, 0]
            assert node.matrix[side - 1][3] > 0
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    node, result, _, targets = execute(
        monkeypatch, hand, 'thumb', 'flex_long', blocked={opposition})
    assert result['timeout'] and result['timeout_stage'] == 'remaining_targets'
    first_good = min(t for t, elapsed, pos, _ in trace
                     if elapsed >= sustained_from and np.isfinite(pos)
                     and abs(pos - targets[flexor]) <= 14)
    first_hold = min(t for t, _, _, rows in trace if all(row == [0, 0, 0.02, 0, 0] for row in rows))
    assert 1.0 <= first_hold - first_good <= 1.0 + 2.0 / dc.PROCESS_SAMPLE_HZ
    for t, _, _, rows in trace:
        assert all(row == [0, 0, 0.05 if t < first_hold else 0.02, 0, 0] for row in rows)
    assert result['position_stable_confirmed'] and result['result_window_label'] == '0.0-1.0'
    assert node.commands[-1][1][extensors[0] - 1] == [0, 0, -0.1, 0, 0]


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('kind', ['emergency', 'pause_hold', 'stop'])
def test_thumb_flexor_special_release_preserves_interrupt_commands(monkeypatch, hand, kind):
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']

    def interrupt(node, clock):
        if node.matrix[muscles['ext_short'] - 1][2] == 0.05 and clock.now >= 1.5:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': kind}

    node, result, _, _ = execute(
        monkeypatch, hand, 'thumb', 'flex_long', blocked={muscles['flex_long']}, runtime=interrupt)
    assert result['interrupted'] and result['interrupt_reason'] == kind
    assert node.commands[-1][1] == [[0] * 5 for _ in range(28)]
    assert not result['result_fb_rows']


CASES = [(hand, finger, 'extensor') for hand in ('left', 'right')
         for finger in ('index', 'middle', 'ring', 'little')]
CASES += [(hand, 'thumb', role) for hand in ('left', 'right')
          for role in ('ext_short', 'ext_long')]

PREUNIFORM_CASES = [(hand, finger, role)
                    for hand, fingers in dc.HAND_MOTOR_MAP.items()
                    for finger, muscles in fingers.items()
                    for role in muscles if 'flex' in role or role in ('abduction', 'adduction')]


def test_points_per_combo_is_forwarded_and_sets_six_combo_quotas(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['data_collection_main.py', '--target-finger', 'middle',
                                    '--points-per-combo', '50'])
    args = dc._parse_args()
    assert args.manual_supplement_after_run == 'true'
    command = dc._build_worker_cmd_from_args(args, manual_after_run=True)
    assert command[command.index('--points-per-combo') + 1] == '50'
    combos = [SimpleNamespace(combo_id=str(i)) for i in range(6)]
    total, floors, caps, targets = dc._build_finger_quota_plan('middle', combos, args.points_per_combo)
    assert total == 300
    assert set(floors.values()) == set(caps.values()) == set(targets.values()) == {50}
    default_total, _, _, _ = dc._build_finger_quota_plan('middle', combos)
    assert default_total == 6 * dc.FOUR_FINGER_QUOTA_BASE_PER_COMBO


def test_points_per_combo_rejects_negative_values(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['data_collection_main.py', '--points-per-combo', '-1'])
    with pytest.raises(SystemExit) as error:
        dc._parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize('hand,finger', [(h, f) for h, fingers in dc.HAND_MOTOR_MAP.items() for f in fingers])
def test_preuniform_starts_with_motor_targets_without_extra_whole_hand_zero(monkeypatch, tmp_path, hand, finger):
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    positions = np.arange(1, 29, dtype=float) * 10.0
    node = SimpleNamespace(hand=hand, snapshot=lambda: {'pos': positions.copy()})
    combo = SimpleNamespace(combo_id=f'{finger}_test')
    finger_state = dict(finger=finger, combos=[combo], point_id=0,
                        finger_zero_calib_events=3, finger_zero_calib_failed_events=2)
    counters = dict(global_point=7, global_timeout=2)
    calibration = dict(zero_calib_state='failed_after_3_attempts', zero_calib_fail_streak=2,
                       zero_calib_total_events=3, zero_calib_failed_events=2)
    original_calibration = calibration.copy()
    targets = []

    def unexpected_zero(*args):
        pytest.fail('Preuniform entry must not run whole-hand zero calibration')

    def collect(**kwargs):
        target = kwargs['target_by_motor'].copy()
        targets.append(target)
        for mid, value in target.items():
            positions[mid - 1] = value
        finger_state['point_id'] += 1
        counters['global_point'] += 1

    monkeypatch.setattr(dc, 'PRE_RANDOM_UNIFORM_ENABLE', True)
    monkeypatch.setattr(dc, '_run_zero_calibration', unexpected_zero)
    monkeypatch.setattr(dc, '_motor_bounds_from_combos', lambda combos, mid: (-60.0, 100.0))
    monkeypatch.setattr(dc, '_pick_combo_for_targets_preferring_deficit', lambda state, target: combo)
    monkeypatch.setattr(dc, '_infer_manual_active_mode', lambda *args: 'x_only')
    monkeypatch.setattr(dc, '_infer_manual_coverage_key', lambda *args: None)
    monkeypatch.setattr(dc, '_collect_one_manual_target_point', collect)
    assert dc._run_pre_random_uniform_sampling(
        node, {}, dc._finger_motor_sets(hand), finger_state, counters, calibration, tmp_path)
    assert targets
    for role, mid in muscles.items():
        sweep = [target[mid] for target in targets if mid in target]
        assert sweep
        if role.startswith('ext'):
            assert sweep[-1] == 0.0
        else:
            assert sweep[0] < 0.0 and sweep[-1] == -60.0
    assert calibration == original_calibration
    assert finger_state['finger_zero_calib_events'] == 3
    assert finger_state['finger_zero_calib_failed_events'] == 2
    assert not (tmp_path / 'zero_calibration_events.csv').exists()


@pytest.mark.parametrize('hand,finger,role', [case for case in CASES if case[1] != 'thumb'])
@pytest.mark.parametrize('blocked', [False, True])
def test_preuniform_extensor_releases_one_second_then_tensions_sway(monkeypatch, hand, finger, role, blocked):
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    active = muscles[role]
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role,
        single=True, preuniform=True, blocked={active} if blocked else ())
    assert result['motion_policy'] == 'extensor_first'
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    assert len(node.commands) == 5
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
    assert first[active - 1] == neutral[active - 1] == [
        targets[active], defaults['vel'][active - 1], -0.08,
        defaults['kp'][active - 1], defaults['kd'][active - 1]]
    assert node.commands[-1][1] == [[0, 0, -0.08, 0, 0] for _ in range(28)]
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


@pytest.mark.parametrize('hand,finger,role', [case for case in PREUNIFORM_CASES if case[1] != 'thumb'])
def test_preuniform_assist_releases_half_second_then_zero_current_until_arrival(monkeypatch, hand, finger, role):
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role,
                                            single=True, preuniform=True)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    assert result['reached'] and result['motion_policy'] == 'preuniform_assist'
    assert len(node.commands) == 5  # Release, current0, tighten, settle, sampling hold.
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
    assert stop == [[0, 0, -0.08, 0, 0] for _ in range(28)]
    assert result['assist_release_tor_plan'] == expected
    assert result['assist_release_sec'] == 0.5
    assert result['assist_release_plan_sec'] == {mid: 0.5 if current else 0.0
                                               for mid, current in expected.items()}
    assert result['result_sample_count'] > 0


@pytest.mark.parametrize('hand,role', [(hand, role) for hand in ('left', 'right')
                                    for role in dc.HAND_MOTOR_MAP[hand]['thumb']])
@pytest.mark.parametrize('blocked', [False, True])
def test_thumb_preuniform_releases_all_passive_motors_then_tightens_only_thumb(monkeypatch, hand, role, blocked):
    original_snapshot = Motors.snapshot

    def snapshot(node):
        snap = original_snapshot(node)
        snap['tor'] = np.array([row[2] for row in node.matrix])
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']
    active = muscles[role]
    passive = set(muscles.values()) - {active}
    other_motors = set(range(1, 29)) - set(muscles.values())
    node, result, defaults, targets = execute(
        monkeypatch, hand, 'thumb', role, single=True, preuniform=True,
        blocked={active} if blocked else (), include_stable=True)
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    assert len(node.commands) == 5  # Release, current0, tighten, settle, sampling.
    _, release = node.commands[0]
    neutral_time, neutral = node.commands[1]
    assert 1.0 <= neutral_time < 1.0 + dc.CONTROL_PERIOD_SEC
    assert release[active - 1] == neutral[active - 1] == [
        targets[active], defaults['vel'][active - 1],
        -0.08 if role.startswith('ext') else defaults['tor'][active - 1],
        defaults['kp'][active - 1], defaults['kd'][active - 1]]
    for mid in passive:
        assert release[mid - 1] == [0, 0, 0.08, 0, 0]
        assert neutral[mid - 1] == [0, 0, 0, 0, 0]
    assert result['assist_release_tor_plan'] == dict.fromkeys(passive, 0.08)
    assert result['assist_release_plan_sec'] == dict.fromkeys(passive, 1.0)
    tighten_time, tighten = node.commands[-3]
    settle_time, settle = node.commands[-2]
    sampling_time, sampling = node.commands[-1]
    assert tighten_time == pytest.approx(result['move_elapsed_sec'])
    tighten_sec = 3.0 if role == 'flex_long' else 2.0
    assert settle_time - tighten_time == pytest.approx(tighten_sec)
    assert result['post_arrival_tighten_elapsed_sec'] == pytest.approx(tighten_sec)
    assert result['position_stable_confirmed']
    assert sampling_time - settle_time == pytest.approx(2.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)
    expected = [[0, 0, -0.1 if mid in muscles.values() else defaults['tor'][mid - 1], 0, 0]
                for mid in range(1, 29)]
    assert tighten == settle == sampling == expected
    for _, matrix in node.commands:
        for mid in other_motors:
            assert matrix[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    assert result['result_sample_count'] > 0
    assert all(row[3::5] == [command[2] for command in expected]
               for row in result['result_fb_rows'])
    assert result['result_window_label'] == '0.0-1.0'
    assert 29 <= result['result_sample_count'] <= 31
    assert min(row[0] for row in result['result_fb_rows']) == pytest.approx(0.0)
    assert all(0.0 <= row[0] <= 1.0 for row in result['result_fb_rows'])
    assert node.clock.now - sampling_time == pytest.approx(1.0, abs=0.0051)
    assert max(row[0] for row in result['process_fb_rows']) == pytest.approx(
        sampling_time + 1.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)


@pytest.mark.parametrize('role', ['ext_short', 'opposition'])
@pytest.mark.parametrize('stop_at', [0.5, 1.5, 3.5, 5.5])
def test_thumb_preuniform_interruption_keeps_emergency_command(monkeypatch, role, stop_at):
    def interrupt(node, clock):
        if clock.now >= stop_at:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}

    active = dc.HAND_MOTOR_MAP['left']['thumb'][role]
    node, result, _, _ = execute(
        monkeypatch, 'left', 'thumb', role, single=True, preuniform=True,
        blocked={active}, runtime=interrupt)
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert result['result_window_label'] == '0.0-1.0'
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert not result['result_fb_rows'] and not result['result_glove_rows']


@pytest.mark.parametrize('role', ['flex_deep', 'abduction', 'adduction'])
def test_preuniform_zero_current_holds_until_timeout_then_tightens(monkeypatch, role):
    active = dc.HAND_MOTOR_MAP['left']['index'][role]
    node, result, defaults, _ = execute(monkeypatch, role=role, single=True,
                                      preuniform=True, blocked={active})
    assert result['timeout'] and not result['reached']
    assert len(node.commands) == 5
    assert 0.5 <= node.commands[1][0] < 0.5 + dc.CONTROL_PERIOD_SEC
    assert result['assist_release_sec'] == 0.5
    assert node.commands[-1][0] >= dc.MOVE_TIMEOUT_SEC
    assert node.commands[-1][1] == [[0, 0, -0.08, 0, 0] for _ in range(28)]


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


@pytest.mark.parametrize('hand,finger,role', [case for case in CASES if case[1] != 'thumb'])
def test_release_and_extensor_arrival_precede_remaining_targets(monkeypatch, hand, finger, role):
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role)
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    ext = muscles[role]
    released = {mid for muscle, mid in muscles.items()
                if 'flex' in muscle or muscle in ('adduction', 'abduction')}
    assert result['reached'] and not result['timeout']
    assert result['motion_policy'] == 'extensor_first'
    assert result['extensor_stage_reached']
    assert len(node.commands) == 5  # Extensor, remaining targets, tighten, settle, sampling hold.
    first_time, first = node.commands[0]
    second_time, second = node.commands[1]
    assert first_time == 0.0  # Immediate motion, no fixed prerelease wait.
    assert second_time >= dc.ARRIVAL_CONFIRM_SEC
    assert first[ext - 1][0] == second[ext - 1][0] == targets[ext]
    assert first[ext - 1][3] == second[ext - 1][3] == defaults['kp'][ext - 1]
    assert first[ext - 1] == second[ext - 1] == [
        targets[ext], defaults['vel'][ext - 1], -0.08,
        defaults['kp'][ext - 1], defaults['kd'][ext - 1]]
    for mid in released:
        assert first[mid - 1] == [0, 0, 0.05, 0, 0]
        if mid in targets:
            assert second[mid - 1][0] == targets[mid] and second[mid - 1][3] > 0
            assert second[mid - 1][2] == defaults['tor'][mid - 1]
        else:
            assert second[mid - 1] == [0, 0, 0, 0, 0]
    if finger == 'thumb':
        opp = muscles['opposition']
        assert first[opp - 1][0] == 0 and first[opp - 1][3] > 0
        assert second[opp - 1][0] == targets[opp]
    for mid in set(range(1, 29)) - set(targets) - released:
        assert first[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
        assert second[mid - 1] == ([0, 0, 0, 0, 0] if mid in muscles.values()
                                  else [0, 0, defaults['tor'][mid - 1], 0, 0])
    last_time, last = node.commands[-1]
    assert last_time - second_time >= dc.ARRIVAL_CONFIRM_SEC
    assert last == expected_post_matrix(hand, finger, defaults)
    assert result['target_cmd_matrix'][muscles['adduction'] - 1][0] == targets[muscles['adduction']]
    assert result['assist_release_tor_plan'] == {mid: 0.05 for mid in released}
    assert result['assist_release_plan_sec'] == dict.fromkeys(released, result['extensor_stage_elapsed_sec'])
    assert result['result_sample_count'] > 0


def test_extensor_timeout_does_not_start_sway_targets(monkeypatch):
    ext = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, defaults, targets = execute(monkeypatch, blocked=[ext])
    assert result['timeout'] and not result['reached']
    assert not result['extensor_stage_reached']
    assert result['timeout_stage'] == 'extensor_first'
    assert len(node.commands) == 4
    assert node.commands[-1][0] >= dc.EXTENSOR_FIRST_TIMEOUT_SEC
    assert node.commands[-1][1] == [[0, 0, -0.08, 0, 0] for _ in range(28)]


@pytest.mark.parametrize('hand,finger,role', [case for case in CASES if case[1] != 'thumb'])
def test_remaining_targets_have_their_own_timeout_with_other_finger_motors_neutral(monkeypatch, hand, finger, role):
    muscles = dc.HAND_MOTOR_MAP[hand][finger]
    sway = muscles['adduction']
    node, result, defaults, targets = execute(monkeypatch, hand, finger, role, blocked=[sway])
    assert result['extensor_stage_reached']
    assert result['timeout'] and result['timeout_stage'] == 'remaining_targets'
    second_time, second = node.commands[1]
    assert node.commands[-3][0] - second_time == pytest.approx(3.0, abs=dc.CONTROL_PERIOD_SEC)
    for mid in set(muscles.values()) - set(targets):
        assert second[mid - 1] == [0, 0, 0, 0, 0]
    for mid, target in targets.items():
        assert second[mid - 1][0] == target and second[mid - 1][3] == defaults['kp'][mid - 1]


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_inactive_thumb_opposition_and_other_extensor_hold_release_after_sway_arrival(monkeypatch, hand):
    node, result, _, targets = execute(monkeypatch, hand, 'thumb', 'ext_short', opposition=False)
    assert result['reached']
    assert result['sway_stage_reached']
    second = node.commands[2][1]
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']
    assert muscles['opposition'] not in targets
    assert second[muscles['opposition'] - 1] == [0, 0, 0.02, 0, 0]
    assert second[muscles['ext_long'] - 1] == [0, 0, 0.02, 0, 0]


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('finger,role', [('index', 'extensor'), ('thumb', 'ext_short')])
def test_emergency_during_remaining_targets_preserves_stop_command(monkeypatch, hand, finger, role):
    muscles = dc.HAND_MOTOR_MAP[hand][finger]

    blocked_mid = muscles[role] if finger == 'thumb' else muscles['adduction']

    def emergency(node, clock):
        second_stage = node.matrix[blocked_mid - 1][3] > 0
        if second_stage and clock.now - node.commands[-1][0] >= 0.5:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}

    node, result, _, _ = execute(monkeypatch, hand, finger, role,
                                blocked={blocked_mid}, runtime=emergency)
    assert result['sway_stage_reached' if finger == 'thumb' else 'extensor_stage_reached']
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert not result['post_arrival_tighten_applied']
    assert not result['result_fb_rows']


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('role', ['flex_deep', 'flex_shallow', 'abduction', 'adduction'])
@pytest.mark.parametrize('blocked', [False, True])
def test_ordinary_release_restores_default_current_without_motion_rewind(monkeypatch, hand, role, blocked):
    active = dc.HAND_MOTOR_MAP[hand]['index'][role]
    node, result, defaults, _ = execute(monkeypatch, hand=hand, role=role, single=True,
                                       blocked={active} if blocked else ())
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    assert len(node.commands) == 5  # Release, defaults, tighten, settle, sampling hold.
    first_time, first = node.commands[0]
    restored_time, restored = node.commands[1]
    tighten_time, _ = node.commands[-3]
    for mid, release_sec in result['assist_release_plan_sec'].items():
        assert first[mid - 1] == [0, 0, 0.5, 0, 0]
        assert release_sec <= restored_time - first_time <= release_sec + dc.CONTROL_PERIOD_SEC
        assert restored[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    for timestamp, matrix in node.commands:
        if timestamp < tighten_time:
            assert all(row[2] != -0.1 for row in matrix)
    assert restored[active - 1][0] == -80.0 and restored[active - 1][3] > 0
    assert result['post_arrival_tighten_applied']
    assert result['result_sample_count'] > 0


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('sway_role', ['adduction', 'abduction'])
@pytest.mark.parametrize('single', [True])
@pytest.mark.parametrize('blocked', [False, True])
def test_thumb_flexor_releases_extensors_and_opposite_sway_one_second_then_zero_current(
        monkeypatch, hand, sway_role, single, blocked):
    muscles = dc.HAND_MOTOR_MAP[hand]['thumb']
    flexor = muscles['flex_long']
    extensors = {muscles['ext_short'], muscles['ext_long']}
    released_motors = set(extensors)
    if not single:
        released_motors.add(muscles['abduction' if sway_role == 'adduction' else 'adduction'])
    motion_samples = []
    original_snapshot = Motors.snapshot

    def snapshot(node):
        snap = original_snapshot(node)
        if node.matrix[flexor - 1][3] > 0:
            motion_samples.append((node.clock.now, [row[:] for row in node.matrix]))
            # Model a tendon that cannot reach its flexion target until its
            # antagonists have had time to release, and remains blocked if they
            # switch back to tension during the arrival confirmation window.
            released = all(node.matrix[mid - 1][3:] == [0, 0]
                           and node.matrix[mid - 1][2] in (0.0, 0.08)
                           for mid in released_motors)
            if node.clock.now < 0.65 or not released:
                node.positions[flexor - 1] = 0.0
                snap['pos'][flexor - 1] = 0.0
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    node, result, defaults, targets = execute(
        monkeypatch, hand, 'thumb', 'flex_long', single=single,
        blocked={flexor} if blocked else (), sway_role=sway_role)
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    assert result['move_elapsed_sec'] >= (3.0 if blocked else 1.65)
    assert motion_samples
    assert any(timestamp >= 1.0 for timestamp, _ in motion_samples)
    for timestamp, matrix in motion_samples:
        for mid in released_motors:
            assert matrix[mid - 1] == [0, 0, 0.08 if timestamp < 1.0 else 0.0, 0, 0]
        for mid, target in targets.items():
            assert matrix[mid - 1] == [target, defaults['vel'][mid - 1],
                                      defaults['tor'][mid - 1], defaults['kp'][mid - 1],
                                      defaults['kd'][mid - 1]]
        for mid in set(range(1, 29)) - set(muscles.values()):
            assert matrix[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    for mid in released_motors:
        assert result['assist_release_tor_plan'][mid] == 0.08
        assert result['assist_release_plan_sec'][mid] == pytest.approx(1.0)
    assert all(matrix == expected_post_matrix(hand, 'thumb', defaults, thumb_combo=not single)
               for _, matrix in node.commands[-3:])
    assert result['post_arrival_tighten_elapsed_sec'] == pytest.approx(3.0)
    assert result['position_stable_confirmed']
    assert result['result_window_label'] == '1.0-2.0'
    assert result['result_sample_count'] > 0


@pytest.mark.parametrize('stop_at', [0.5, 1.5])
def test_thumb_flexor_release_and_zero_current_preserve_emergency_command(monkeypatch, stop_at):
    def emergency(node, clock):
        if clock.now >= stop_at:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': 'emergency'}

    active = dc.HAND_MOTOR_MAP['left']['thumb']['flex_long']
    node, result, _, _ = execute(
        monkeypatch, 'left', 'thumb', 'flex_long', blocked={active}, runtime=emergency)
    assert result['interrupted'] and result['interrupt_reason'] == 'emergency'
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert not result['result_fb_rows'] and not result['result_glove_rows']


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
    original = Motors.snapshot

    def snapshot(node):
        snap = original(node)
        # This case tests arrival failure; feedback recovers once movement ends.
        if not any(row[3] > 0 for row in node.matrix):
            snap['feedback_age'] = 0.0
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    ext = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, _, _ = execute(monkeypatch, stale=bad_feedback == 'stale',
                                  nan_motor=ext if bad_feedback == 'nan' else None)
    assert result['timeout_stage'] == 'extensor_first'
    assert not result['extensor_stage_reached']
    assert len(node.commands) == 4


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
    assert len(node.commands) == 4  # Extensor, tighten, settle, sampling hold.


@pytest.mark.parametrize('disabled', [False, True])
def test_flexor_points_and_disabled_assist_use_original_simultaneous_motion(monkeypatch, disabled):
    role = 'extensor' if disabled else 'flex_deep'
    node, result, defaults, targets = execute(monkeypatch, role=role, disable_assist=disabled)
    assert result['motion_policy'] == 'simultaneous'
    assert result['reached']
    first = node.commands[0][1]
    assert all(first[mid - 1][0] == target and first[mid - 1][3] > 0
               for mid, target in targets.items())
    ext = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    for mid in targets:
        assert first[mid - 1][2] == (-0.08 if mid == ext else defaults['tor'][mid - 1])
        assert first[mid - 1][3:] == [defaults['kp'][mid - 1], defaults['kd'][mid - 1]]


@pytest.mark.parametrize('hand,finger,role', [case for case in CASES + PREUNIFORM_CASES
                                           if case[1] != 'thumb' or case[2] not in ('adduction', 'abduction')])
@pytest.mark.parametrize('blocked', [False, True])
def test_arrival_or_timeout_tightens_three_seconds_before_sampling(monkeypatch, hand, finger, role, blocked):
    original_snapshot = Motors.snapshot

    def snapshot(node):
        snap = original_snapshot(node)
        # Make phase contamination visible in both glove and feedback averages.
        snap['glove48'][:] = node.matrix[0][2]
        snap['tor'] = np.array([row[2] for row in node.matrix])
        return snap

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    active = dc.HAND_MOTOR_MAP[hand][finger][role]
    node, result, defaults, targets = execute(monkeypatch, hand=hand, finger=finger,
                                             role=role, include_stable=True,
                                             blocked={active} if blocked else ())
    tighten_sec = 2.0 if finger == 'thumb' and role != 'flex_long' else 3.0
    record_sec = 1.0 if finger == 'thumb' else 2.0
    assert result['reached'] == (not blocked)
    assert result['timeout'] == blocked
    tighten_start, tighten = node.commands[-3]
    settle_start, settle = node.commands[-2]
    stable_start, stable = node.commands[-1]
    expected = expected_post_matrix(hand, finger, defaults)
    assert tighten == expected
    # A final movement command can share the tighten command's timestamp.
    assert all(matrix == expected for _, matrix in node.commands[-3:])
    assert settle_start - tighten_start == pytest.approx(tighten_sec)
    assert tighten_start == pytest.approx(result['move_elapsed_sec'])
    assert stable == expected
    assert result['post_arrival_tighten_applied']
    assert result['post_arrival_tighten_elapsed_sec'] == pytest.approx(tighten_sec)
    assert result['position_stable_confirmed']
    assert result['position_stable_wait_sec'] == pytest.approx(2.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)
    assert result['position_stable_motor_ids'] == sorted(dc.HAND_MOTOR_MAP[hand][finger].values())
    assert stable_start - settle_start == pytest.approx(result['position_stable_wait_sec'])
    # No motor changes current at the settling or sampling boundaries.
    assert settle == stable == expected
    assert result['result_sample_count'] > 0
    assert result['result_glove_avg'] == pytest.approx([expected[0][2]] * 48)
    # The result clock starts after the stability gate while currents stay unchanged.
    assert all((0.0 if finger == 'thumb' else 1.0) <= row[0] <= record_sec
               for row in result['result_fb_rows'])
    assert all(row[3::5] == [command[2] for command in expected] for row in result['result_fb_rows'])
    # CSV-mode process logs include the 3 s phase and retain the total elapsed time.
    # A motion snapshot can share the command timestamp at the phase boundary.
    # Inspect samples strictly inside each phase for its commanded currents.
    tighten_rows = [row for row in result['process_fb_rows'] if tighten_start < row[0] < settle_start]
    assert tighten_rows
    assert all(row[3] == tighten[0][2] for row in tighten_rows)
    settle_rows = [row for row in result['process_fb_rows'] if settle_start < row[0] < stable_start]
    assert settle_rows
    assert all(row[3::5] == [command[2] for command in settle] for row in settle_rows)
    assert max(row[0] for row in result['process_fb_rows']) == pytest.approx(
        stable_start + record_sec, abs=1.0 / dc.PROCESS_SAMPLE_HZ)


@pytest.mark.parametrize('role', ['extensor', 'flex_deep'])
@pytest.mark.parametrize('hand', ['left', 'right'])
def test_timeout_tightens_before_sampling_and_keeps_timeout_flag(monkeypatch, role, hand):
    active = dc.HAND_MOTOR_MAP[hand]['index'][role]
    node, result, defaults, targets = execute(monkeypatch, hand=hand, role=role, blocked={active})
    assert result['timeout'] and not result['reached']
    assert result['post_arrival_tighten_applied']
    assert result['post_arrival_tighten_elapsed_sec'] == pytest.approx(3.0)
    expected = [[0, 0, -0.08, 0, 0] for _ in range(28)]
    assert node.commands[-3][1] == expected
    assert node.commands[-2][0] - node.commands[-3][0] == pytest.approx(3.0)
    assert node.commands[-1][1] == [[0, 0, -0.08, 0, 0] for _ in range(28)]
    assert result['result_sample_count'] > 0


@pytest.mark.parametrize('kind', ['emergency', 'pause_hold', 'stop'])
@pytest.mark.parametrize('stop_after', [0.5, 2.5])
@pytest.mark.parametrize('blocked', [False, True])
def test_interruption_during_tightening_preserves_command_and_skips_result(monkeypatch, kind, stop_after, blocked):
    def interrupt(node, clock):
        expected = [[0, 0, -0.08, 0, 0] for _ in range(28)]
        if node.matrix == expected and clock.now - node.commands[-1][0] >= stop_after:
            node.set_command_matrix([[0.0] * 5 for _ in range(28)])
            return {'kind': kind}

    active = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, _, _ = execute(monkeypatch, runtime=interrupt, blocked={active} if blocked else ())
    assert result['interrupted'] and result['interrupt_reason'] == kind
    assert not result['reached']
    assert stop_after <= result['post_arrival_tighten_elapsed_sec'] < 3.0
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]
    assert result['result_sample_count'] == 0
    assert not result['result_fb_rows'] and not result['result_glove_rows']


STABILITY_MOTORS = [(hand, finger, mid)
                    for hand, fingers in dc.HAND_MOTOR_MAP.items()
                    for finger, muscles in fingers.items()
                    for mid in muscles.values()]


@pytest.mark.parametrize('hand,finger,moving_motor', STABILITY_MOTORS)
def test_stability_rejects_movement_on_every_current_finger_motor(monkeypatch, hand, finger, moving_motor):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)

    def snapshot():
        pos = np.zeros(28)
        pos[moving_motor - 1] = clock.now * 10.0
        return dict(pos=pos, feedback_age=0.0)

    result = dc._wait_finger_position_stable(
        SimpleNamespace(snapshot=snapshot), dc._finger_motor_sets(hand)[finger],
        runtime_tick=lambda: {'kind': 'stop'} if clock.now >= 3.1 else None,
    )
    assert result['interrupted'] and not result['position_stable_confirmed']


@pytest.mark.parametrize('hand,finger', [(h, f) for h, fingers in dc.HAND_MOTOR_MAP.items() for f in fingers])
def test_stability_ignores_movement_on_other_fingers_and_wrist(monkeypatch, hand, finger):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    mids = dc._finger_motor_sets(hand)[finger]

    def snapshot():
        pos = np.full(28, clock.now * 100.0)
        pos[np.array(mids) - 1] = -50.0
        return dict(pos=pos, feedback_age=0.0)

    result = dc._wait_finger_position_stable(SimpleNamespace(snapshot=snapshot), mids)
    assert result['position_stable_confirmed']
    assert result['position_stable_wait_sec'] == pytest.approx(2.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)
    assert result['position_stable_max_span_deg'] == 0.0


@pytest.mark.parametrize('excursion', [3.0, 3.01])
def test_stability_uses_inclusive_three_degree_range_and_full_window(monkeypatch, excursion):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)

    def snapshot():
        pos = np.zeros(28)
        pos[0] = excursion if clock.now > 0.0 else 0.0
        return dict(pos=pos, feedback_age=0.0)

    result = dc._wait_finger_position_stable(SimpleNamespace(snapshot=snapshot), [1])
    assert result['position_stable_confirmed']
    if excursion == 3.0:
        assert result['position_stable_wait_sec'] == pytest.approx(2.0)
        assert result['position_stable_max_span_deg'] == 3.0
    else:
        assert result['position_stable_wait_sec'] > 2.0
        assert result['position_stable_max_span_deg'] == 0.0


def test_stability_rejects_back_and_forth_motion_even_with_identical_endpoints(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)

    def snapshot():
        pos = np.zeros(28)
        pos[0] = 10.0 * np.sin(np.pi * clock.now)
        return dict(pos=pos, feedback_age=0.0)

    result = dc._wait_finger_position_stable(
        SimpleNamespace(snapshot=snapshot), [1],
        runtime_tick=lambda: {'kind': 'stop'} if clock.now >= 2.1 else None,
    )
    assert not result['position_stable_confirmed']


@pytest.mark.parametrize('bad_feedback', ['missing', 'stale', 'nan', 'infinite_age'])
def test_stability_resets_two_second_window_on_invalid_feedback(monkeypatch, bad_feedback):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)

    def snapshot():
        pos = np.zeros(28)
        age = 0.0
        if 0.5 <= clock.now < 1.0:
            if bad_feedback == 'missing':
                return None
            if bad_feedback == 'stale':
                age = 2.0
            if bad_feedback == 'infinite_age':
                age = float('inf')
            if bad_feedback == 'nan':
                pos[0] = float('nan')
        return dict(pos=pos, feedback_age=age)

    result = dc._wait_finger_position_stable(SimpleNamespace(snapshot=snapshot), [1])
    assert result['position_stable_confirmed']
    assert 3.0 <= result['position_stable_wait_sec'] < 3.1


@pytest.mark.parametrize('kind', ['emergency', 'pause_hold', 'stop'])
@pytest.mark.parametrize('blocked', [False, True])
def test_interruption_during_stability_wait_preserves_command_and_skips_result(monkeypatch, kind, blocked):
    def interrupt(node, clock):
        if len(node.commands) == (3 if blocked else 4):
            if clock.now - node.commands[-1][0] >= 0.5:
                node.set_command_matrix([[0.0] * 5 for _ in range(28)])
                return {'kind': kind}

    active = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, _, _ = execute(monkeypatch, runtime=interrupt, blocked={active} if blocked else ())
    assert result['interrupted'] and result['interrupt_reason'] == kind
    assert not result['position_stable_confirmed']
    assert result['position_stable_wait_sec'] >= 0.5
    assert not result['result_fb_rows'] and not result['result_glove_rows']
    assert node.commands[-1][1] == [[0.0] * 5 for _ in range(28)]


@pytest.mark.parametrize('blocked', [False, True])
def test_sampling_waits_for_passive_motor_to_settle_and_excludes_wait_frames(monkeypatch, blocked):
    original_snapshot = Motors.snapshot
    original_collect = dc._collect_window_samples
    sampling_start = []

    def snapshot(node):
        snap = original_snapshot(node)
        if len(node.commands) == (3 if blocked else 4):
            elapsed = node.clock.now - node.commands[-1][0]
            # Passive left-index flexor M15 keeps retracting during the stability wait.
            snap['pos'][14] = 100.0 * min(elapsed, 3.0)
        snap['glove48'][:] = 7.0 if sampling_start else -99.0
        return snap

    def collect(**kwargs):
        sampling_start.append(dc.time.monotonic())
        return original_collect(**kwargs)

    monkeypatch.setattr(Motors, 'snapshot', snapshot)
    monkeypatch.setattr(dc, '_collect_window_samples', collect)
    active = dc.HAND_MOTOR_MAP['left']['index']['extensor']
    node, result, _, _ = execute(monkeypatch, include_stable=True, blocked={active} if blocked else ())
    assert result['position_stable_confirmed']
    assert result['timeout'] == blocked
    assert result['position_stable_wait_sec'] >= 4.9
    assert sampling_start[0] == pytest.approx(node.commands[-1][0])
    assert sampling_start[0] == pytest.approx(node.commands[-2][0] + result['position_stable_wait_sec'])
    assert result['result_glove_avg'] == pytest.approx([7.0] * 48)
    assert any(row[0] == -99.0 for row in result['process_glove_rows'])
    assert max(row[0] for row in result['process_fb_rows']) == pytest.approx(
        sampling_start[0] + 2.0, abs=1.0 / dc.PROCESS_SAMPLE_HZ)
    assert dc._point_motion_metadata(result)['position_stable_confirmed']


@pytest.mark.parametrize('hand', ['left', 'right'])
@pytest.mark.parametrize('feedback', ['ok', 'blocked', 'missing', 'stale'])
def test_thumb_zero_only_controls_thumb_and_preserves_four_finger_currents(monkeypatch, hand, feedback):
    clock = Clock()
    monkeypatch.setattr(dc, 'time', clock)
    thumb = set(dc.HAND_MOTOR_MAP[hand]['thumb'].values())
    node = Motors(hand, clock, blocked=thumb if feedback == 'blocked' else (),
                  stale=feedback == 'stale')
    initial = -np.arange(1, 29, dtype=float) * 20.0
    node.positions[:] = initial
    defaults = dc._load_defaults_from_py(PACKAGE / 'scripts' / 'muscle_adjustment_defaults.py', hand)
    if feedback == 'missing':
        node.snapshot = lambda: None

    def forbidden_prerelease(*args, **kwargs):
        pytest.fail('Thumb zero must never release the four fingers')

    monkeypatch.setattr(dc, '_run_zero_prerelease', forbidden_prerelease)
    result = dc._run_zero_calibration(node, defaults, finger='thumb')
    assert result['success'] == (feedback == 'ok')
    assert result['attempts'] == (1 if feedback == 'ok' else dc.ZERO_CALIBRATE_MAX_ATTEMPTS)
    assert node.commands
    opposition = dc.HAND_MOTOR_MAP[hand]['thumb']['opposition']
    for _, matrix in node.commands[:result['attempts']]:
        for mid in thumb:
            assert matrix[mid - 1] == [
                0, defaults['vel'][mid - 1],
                0.08 if mid == opposition else defaults['tor'][mid - 1],
                defaults['kp'][mid - 1], defaults['kd'][mid - 1]]
    for _, matrix in node.commands:
        for mid in set(range(1, 29)) - thumb:
            assert matrix[mid - 1] == [0, 0, defaults['tor'][mid - 1], 0, 0]
    for mid in set(range(1, 29)) - thumb:
        assert node.positions[mid - 1] == initial[mid - 1]
    if feedback == 'ok':
        assert result['last_max_abs_pos'] == 0.0
        assert all(node.positions[mid - 1] == 0.0 for mid in thumb)
    else:
        assert result['failed_stage'] == 'thumb'
        if feedback in ('missing', 'stale'):
            assert all(node.commands[-1][1][mid - 1] == [0] * 5 for mid in thumb)


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
def test_four_finger_quota_is_600_per_combo(finger, count):
    state = quota_state(finger, count)
    assert state['nominal_total'] == count * 600
    assert set(state['combo_target'].values()) == {600.0}
    assert set(state['combo_min'].values()) == {540}
    assert set(state['combo_max'].values()) == {660}
    assert state['mode_min']['xy'] == count * 540
    assert state['mode_max']['xy'] == count * 660


def test_thumb_quota_and_opposition_split_are_unchanged():
    state = quota_state('thumb', 12)
    assert state['nominal_total'] == 24000
    assert state['combo_target']['thumb_1'] == 2800.0
    assert state['combo_target']['thumb_0'] == 1200.0
    assert state['mode_min']['full'] == 21600
    assert state['mode_max']['full'] == 26400


def test_resume_with_fewer_thumb_combos_rebuilds_limits_and_keeps_history():
    previous = quota_state('thumb', 12)
    previous['combo_counts'] = dict.fromkeys(previous['combo_counts'], 500)
    previous['mode_counts']['full'] = 5944
    previous['mode_counts']['single'] = 56
    previous['point_id'] = 6000
    previous['preuniform_done'] = True
    previous['done'] = True
    payload = dc._finger_state_to_resume_payload(previous)
    restored = quota_state('thumb', 10)
    expected_min = restored['combo_min'].copy()
    expected_max = restored['combo_max'].copy()
    expected_mode_min = restored['mode_min'].copy()
    expected_mode_max = restored['mode_max'].copy()
    dc._apply_finger_resume_payload(restored, payload)
    assert restored['point_id'] == 6000 and restored['preuniform_done']
    assert set(restored['combo_counts']) == {f'thumb_{i}' for i in range(10)}
    assert set(restored['combo_counts'].values()) == {500}
    assert restored['mode_counts'] == previous['mode_counts']
    assert restored['combo_min'] == expected_min and restored['combo_max'] == expected_max
    assert restored['mode_min'] == expected_mode_min and restored['mode_max'] == expected_mode_max
    assert not restored['done']
    assert restored['nominal_total'] == 20000


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
    assert set(state['combo_min'].values()) == {540}
    assert set(state['combo_max'].values()) == {660}
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
    assert payload['quota_base_per_combo'] == (2000 if finger == 'thumb' else 600)
    if finger == 'thumb':
        payload.pop('quota_base_per_combo')  # Legacy thumb checkpoints also remain valid.
    restored = quota_state(finger, count)
    dc._apply_finger_resume_payload(restored, payload)
    assert restored['done']
    assert restored['combo_counts'][cid] == 700
    assert restored['combo_min'][cid] == restored['combo_max'][cid] == 700
