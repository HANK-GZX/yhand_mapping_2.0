#!/usr/bin/env python3

from pathlib import Path

MOTOR_COUNT = 28

# Hand and topics
DEFAULT_HAND = "left"
HAND_CHOICES = ("left", "right")
DEFAULT_GLOVE_TOPIC = "/glove_joint_data"
DEFAULT_GLOVE_RAW_SKELETON_TOPIC = "/glove_joint_data_raw_skeleton"

# Control and timing
CONTROL_PERIOD_SEC = 0.02
PROCESS_SAMPLE_HZ = 30.0
MOVE_TIMEOUT_SEC = 3.0
# Arrival must hold under error threshold long enough to reject transient crossings,
# while leaving most of MOVE_TIMEOUT_SEC available for the actual movement.
ARRIVAL_CONFIRM_SEC = 1.0
RETURN_ZERO_TIMEOUT_SEC = 3.0
# After arrival or timeout, keep all motors at this current through tightening,
# the position stability gate, and result sampling (all kp/kd=0).
POST_ARRIVAL_TIGHTEN_TOR_A = -0.08
POST_ARRIVAL_TIGHTEN_SEC = 3.0
# Require a continuous stable position window on every current-finger motor
# without switching currents before or after the gate.
POSITION_STABLE_WINDOW_SEC = 2.0
POSITION_STABLE_MAX_RANGE_DEG = 3.0
# Post-move stabilization duration (always executed for reached/timeout).
STABLE_HOLD_SEC = 2.0
RESULT_WINDOW_START_SEC = 1.0
RESULT_WINDOW_END_SEC = 2.0
POINT_ARRIVAL_TOL = 14.0
FEEDBACK_STALE_SEC = 1.0
READY_WAIT_TIMEOUT_SEC = 8.0

# Target direction constraint:
# when enabled, every active motor is only allowed to move toward
# smaller position targets (target < current), i.e., tighten direction.
ENFORCE_ACTIVE_TARGET_DECREASE = True
# Backward-compatible alias.
ENFORCE_EXTENSOR_TARGET_DECREASE = ENFORCE_ACTIVE_TARGET_DECREASE
# Numeric tolerance for direction checks.
TARGET_DIRECTION_EPS = 1e-6

# Antagonist release assist during multi-motor active movement:
# for each active motor, its mapped antagonist(s) are commanded to passive release.
ANTAGONIST_RELEASE_ENABLE = True
# Points with an active extensor: release this finger's flexors and both sway
# motors while the extensor moves, then move the remaining target motors.
EXTENSOR_FIRST_ENABLE = True
EXTENSOR_FIRST_RELEASE_CURRENT_A = 0.05
# Active extensor position control keeps the per-motor kp/kd and uses this
# fixed feedforward current during movement and target holding.
ACTIVE_EXTENSOR_FEEDFORWARD_TOR_A = -0.08
# Each movement stage gets its own timeout; there is no prerelease dwell.
EXTENSOR_FIRST_TIMEOUT_SEC = MOVE_TIMEOUT_SEC
# Deprecated compatibility knobs:
# main planner now reads DATA_COLLECTION_RELEASE_*_BY_ACTIVE_TARGET below.
# Keep these for backward compatibility; they are not used by planner logic.
ANTAGONIST_RELEASE_TOR = 0.5
ANTAGONIST_RELEASE_SEC = 0.4
# Data-collection zero return: release four-finger flexors/sway for 1 s,
# then tension them at -0.03 A while extensors return first, then all motors.
ZERO_SWAY_RELEASE_TOR_A = 0.08
ZERO_SWAY_RELEASE_SEC = 1.0
ZERO_FLEXOR_RELEASE_TOR_A = 0.08
ZERO_FLEXOR_RELEASE_SEC = 1.0
ZERO_FLEX_SWAY_TIGHTEN_TOR_A = -0.03
ZERO_EXTENSOR_TOR_A = -0.05

# Data-collection-only antagonist release current profile (A).
# Intentionally decoupled from motor_position_test.
DATA_COLLECTION_RELEASE_TOR_DEFAULT_A = 0.5

# Per-hand, per-active-motor, per-target-motor release current override.
# Structure:
#   hand -> active_motor_id -> {target_motor_id: release_tor_A}
# Any missing pair falls back to DATA_COLLECTION_RELEASE_TOR_DEFAULT_A.
DATA_COLLECTION_RELEASE_TOR_BY_ACTIVE_TARGET = {'left': {1: {28: 0.5},
          2: {23: 0.5},
          15: {25: 0.5},
          27: {8: 0.5},
          7: {4: 0.5},
          8: {27: 0.5},
          24: {9: 0.5},
          9: {24: 0.5},
          11: {13: 0.5},
          20: {18: 0.5},
          18: {20: 0.5},
          3: {25: 0.5},
          16: {23: 0.5},
          17: {10: 0.5},
          13: {11: 0.5},
          21: {22: 0.5},
          10: {17: 0.5},
          22: {21: 0.5},
          23: {2: 0.45, 16: 0.45},
          25: {15: 0.5, 3: 0.5},
          6: {4: 0.5},
          28: {1: 0.5},
          4: {7: 0.5, 6: 0.5}},
 'right': {17: {11: 0.5},
           2: {9: 0.5},
           3: {24: 0.5},
           27: {25: 0.5},
           7: {8: 0.5},
           24: {3: 0.5},
           8: {7: 0.5},
           9: {2: 0.5, 16: 0.5},
           11: {17: 0.5, 1: 0.5},
           20: {18: 0.5},
           14: {15: 0.5},
           18: {20: 0.5, 21: 0.5},
           15: {14: 0.5},
           16: {9: 0.5},
           1: {11: 0.5},
           13: {},
           21: {18: 0.5},
           22: {},
           10: {23: 0.5},
           23: {10: 0.5},
           25: {27: 0.5},
           6: {4: 0.5},
           4: {6: 0.5}}}

# Data-collection-only antagonist release duration profile (sec / degree).
# Release duration is computed as:
#   release_sec = abs(delta_deg_of_active_motor) * per_deg_sec
# where per_deg_sec is configurable by active->target motor pair below.
DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT = 0.001

# Per-hand, per-active-motor, per-target-motor release duration gain.
# Structure:
#   hand -> active_motor_id -> {target_motor_id: release_sec_per_deg}
# Any missing pair falls back to DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT.
#
# Left-hand entries below follow the requested mapping list, and can be tuned
# independently from motor_position_test.
DATA_COLLECTION_RELEASE_SEC_PER_DEG_BY_ACTIVE_TARGET = {'left': {1: {28: 0.00018},
          2: {23: 0.0008},
          15: {25: 0.001},
          27: {8: 0.003},
          7: {4: 0.0015},
          8: {27: 0.003},
          24: {9: 0.0001},
          9: {24: 0.00015},
          11: {13: 0.0015},
          20: {18: 0.00015},
          18: {20: 0.0001},
          3: {25: 0.001},
          16: {23: 0.0008},
          17: {10: 0.001},
          13: {11: 0.001},
          21: {22: 0.00013},
          10: {17: 0.0015},
          22: {21: 0.00015},
          23: {2: 0.0019, 16: 0.0015},
          25: {15: 0.0027, 3: 0.0015},
          6: {4: 0.0015},
          28: {1: 0.00018},
          4: {7: 0.0005, 6: 0.002}},
 'right': {17: {11: 0.001},
           2: {9: 0.001},
           3: {24: 0.001},
           27: {25: 0.001},
           7: {8: 0.001},
           24: {3: 0.001},
           8: {7: 0.001},
           9: {2: 0.001, 16: 0.001},
           11: {17: 0.001, 1: 0.001},
           20: {18: 0.001},
           14: {15: 0.001},
           18: {20: 0.001, 21: 0.001},
           15: {14: 0.001},
           16: {9: 0.001},
           1: {11: 0.001},
           13: {},
           21: {18: 0.001},
           22: {},
           10: {23: 0.001},
           23: {10: 0.001},
           25: {27: 0.001},
           6: {4: 0.001},
           4: {6: 0.001}}}

# Planning / random walk
QUOTA_BASE_PER_COMBO = 2000
FOUR_FINGER_QUOTA_BASE_PER_COMBO = 600
QUOTA_FLOAT_RATIO = 0.10
# Thumb opposition sampling preference:
# when thumb has both opposition_mode on/off combos, allocate about this ratio to "on".
THUMB_OPPOSITION_ON_RATIO = 0.70
# Active-motor mode quota for non-thumb fingers (2D combos).
# x axis = flex/ext, y axis = adduction/abduction.
FOUR_FINGER_ACTIVE_MODE_RATIO = {
    "xy": 0.80,
    "x_only": 0.15,
    "y_only": 0.05,
}
# Active-motor mode quota for thumb (3D combos).
THUMB_ACTIVE_MODE_RATIO = {
    "full": 0.80,
    "single": 0.10,
    "double": 0.10,
}
# Random stage single-motor active modes:
# when enabled, random-stage mode selection will not choose single-motor modes
# (thumb: single, four-finger: x_only/y_only), because preuniform has already
# covered deterministic single-motor sampling.
# In current policy, thumb random stage is further restricted to full-only
# (double is also disabled in random stage).
DISABLE_RANDOM_SINGLE_MOTOR_MODES = True
# Thumb abduction boundary enhancement:
# on top of normal thumb quota, add extra points and bias these samples
# near thumb "out" boundary region.
THUMB_ABDUCTION_EXTRA_POINTS = 500
# Thumb flexion boundary enhancement:
# add extra points to thumb flexion combos and bias x sampling near flexion-side edge.
THUMB_FLEXION_EXTRA_POINTS = 500
# Four-finger flexion boundary enhancement:
# for index/middle/ring/little, add extra points per finger to flexion combos.
FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER = 500
# Near-boundary region width ratio on y-axis for thumb out combos.
# 0.20 means "closest 20% to boundary side".
THUMB_ABDUCTION_EDGE_REGION_RATIO = 0.20
# Probability of sampling from near-boundary region for thumb out combos
# when y motor is active.
THUMB_ABDUCTION_EDGE_BIAS_PROB = 0.90
# Near-boundary region + bias for flexion-side x-axis sampling.
FLEXION_EDGE_REGION_RATIO = 0.20
FLEXION_EDGE_BIAS_PROB = 0.90

# Random collection finish guard:
# a finger is allowed to finish when total points reaches at least this ratio
# of nominal_total, even if some combo quotas are still hard to satisfy.
FINGER_COMPLETE_MIN_RATIO = 0.80
# If finger cannot continue because all combos are frozen/capped and ratio is
# still below FINGER_COMPLETE_MIN_RATIO, unfreeze all combo caps and retry from
# zero-calibrated state. This limits infinite loops.
FINGER_UNFREEZE_MAX_ROUNDS = 10
FINGER_UNFREEZE_MAX_NO_PROGRESS_ROUNDS = 3

# Pre-random deterministic single-motor sampling.
PRE_RANDOM_UNIFORM_ENABLE = True
PRE_RANDOM_FLEX_STEP = 15.0
PRE_RANDOM_SWAY_STEP = 7.0
PRE_RANDOM_OTHER_STEP = 15.0
PRE_RANDOM_EXTENSOR_STEP = 15.0
PRE_RANDOM_FLEX_EXTENSOR_RELEASE_TOR_A = 0.05
PRE_RANDOM_SWAY_OPPOSITE_RELEASE_TOR_A = 0.05
PRE_RANDOM_PASSIVE_RELEASE_SEC = 0.5
PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_TOR_A = 0.05
PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_SEC = 1.0
PRE_RANDOM_EXTENSOR_SWAY_TIGHTEN_TOR_A = -0.03
ZERO_CALIBRATE_EVERY_POINTS = 50
ZERO_CALIBRATE_MAX_ATTEMPTS = 3
# After zero position is reached, keep within tolerance for this stable hold
# duration before continuing data collection.
ZERO_CALIBRATE_STABLE_SEC = 2.0
TARGET_GENERATION_RETRY = 40
MIN_STEP_ABS = 1.0
BOUNDARY_EPS = 1e-6
# If available range on one axis is smaller than this value,
# treat that axis as degenerate (fixed value) and continue sampling on other axes.
DEGENERATE_AXIS_SPAN_EPS = 1.0
# Direction sampling preference at boundary:
# choose direction by available room (farther side is more likely).
DIRECTION_BY_ROOM_WEIGHTED = False
# If a combo repeatedly cannot generate target points, put it in temporary cooldown
# to avoid local dead-loop; it will be retried automatically after cooldown expires.
TARGET_NONE_MAX_CONSEC_PER_COMBO = 120
TARGET_NONE_WARN_EVERY = 50
TARGET_NONE_COOLDOWN_CYCLES = 60
# Combo sampling weights (quota gap + remaining room + coverage gap).
COMBO_WEIGHT_QUOTA_GAP_GAIN = 4.0
COMBO_WEIGHT_REMAIN_GAIN = 0.5
COMBO_WEIGHT_COVERAGE_GAIN = 1.0
# Combo stall handling:
# if one combo gets no new target for N consecutive attempts:
# - when count >= target*ratio: mark combo finished (freeze min/max at current count)
# - when count < target*ratio: run one zero-calibration retry and continue sampling.
#   After retry count is exhausted, mark combo finished.
#
# Current policy (requested):
# - stall window = 50 consecutive target-none attempts
# - completion threshold ratio = 70%
# - for combos already >= threshold ratio, require this many stall windows
#   before allowing combo freeze.
# - zero-calibration retries before freezing low-ratio combo = 10
COMBO_STALL_FAIL_TRIGGER = 50
COMBO_STALL_COMPLETE_RATIO = 0.70
COMBO_STALL_COMPLETE_HIT_WINDOWS = 3
COMBO_STALL_MAX_ZERO_RETRY = 10
# Coverage-driven region sampling: weight = 1 / (1 + count)^power.
BIN_COVERAGE_INV_WEIGHT_POWER = 1.0

# Runtime / logging
STATUS_PRINT_EVERY_POINTS = 20
RNG_SEED = None

# Fixed finger execution order required by user.
FINGER_ORDER_FIXED = ["thumb", "index", "middle", "ring", "little"]

def default_output_root(script_dir: Path) -> Path:
    return script_dir / "motor_test_logs" / "data_collection"
