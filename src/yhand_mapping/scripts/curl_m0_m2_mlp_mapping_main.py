#!/usr/bin/env python3
import argparse
from collections import deque
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

from data_collection_config import CONTROL_PERIOD_SEC, DEFAULT_GLOVE_TOPIC
from workspace_boundary_config import get_hand_topics

try:
    from replace_right_non_thumb_q_from_skeleton import compute_non_thumb_q16
except Exception as exc:  # pragma: no cover
    compute_non_thumb_q16 = None
    RIGHT_NON_THUMB_Q_IMPORT_ERROR = exc
else:
    RIGHT_NON_THUMB_Q_IMPORT_ERROR = None


from hand_motor_map import (
    HAND_MOTOR_MAP, LEFT_TO_RIGHT_MOTOR_MAP, RIGHT_TO_LEFT_MOTOR_MAP,
)

from hand_motor_map import (
    model_motor_ids,
)

LEFT_HAND_GLOVE_SLICE = slice(8, 28)
RIGHT_HAND_GLOVE_SLICE = slice(28, 48)
MOTOR_COUNT = 28
RAW_SKELETON_ONE_HAND_DIM = 176
RAW_SKELETON_TWO_HAND_DIM = 352
RAW_Q_STATUS_PERIOD_SEC = 1.0

FIVE_FINGER_Q_GROUPS = {
    "thumb": [0, 1, 2, 3],
    "index": [4, 5, 6, 7],
    "middle": [8, 9, 10, 11],
    "ring": [12, 13, 14, 15],
    "little": [16, 17, 18, 19],
}

FINGER_RUNTIME_PRESETS = {
    "index": {
        "q_indices": [4, 5, 6, 7],
        "right_native_motor_ids": model_motor_ids('right', 'index'),
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "index",
            "best_index_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "index",
            "index_target_q0_supervised_curl_1570.meta.json",
        ),
    },
    "middle": {
        "q_indices": [8, 9, 10, 11],
        "right_native_motor_ids": model_motor_ids('right', 'middle'),
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "middle",
            "best_middle_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "middle",
            "middle_target_q0_supervised_curl_1357.meta.json",
        ),
    },
    "ring": {
        "q_indices": [12, 13, 14, 15],
        "right_native_motor_ids": model_motor_ids('right', 'ring'),
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "ring",
            "best_ring_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "ring",
            "ring_target_q0_supervised_curl_1704.meta.json",
        ),
    },
    "little": {
        "q_indices": [16, 17, 18, 19],
        "right_native_motor_ids": model_motor_ids('right', 'little'),
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "little",
            "best_little_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "little",
            "little_target_q0_supervised_curl_1320.meta.json",
        ),
    },
}
DEFAULT_FINGER_ORDER = ("index", "middle", "ring", "little")

TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG = 3.0
TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG = 5.0
SIDE_SWING_TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG = 5.0
SIDE_SWING_TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG = 7.0
CMD_POSITION_HOLD_DEADBAND_DEG = 1.5

# Shared anatomical correspondence is imported above.

SUPPORTED_MODEL_FEATURE_COLS = {
    "target_q_0",
    "target_curl",
}
# Shared anatomical correspondence is imported above.

RIGHT_SIDE_SWING_MOTOR_IDS = {mid for finger, muscles in HAND_MOTOR_MAP['right'].items()
    if finger != "thumb" for role, mid in muscles.items() if role in ("abduction", "adduction")} 
LEFT_SIDE_SWING_MOTOR_IDS = {mid for finger, muscles in HAND_MOTOR_MAP['left'].items()
    if finger != "thumb" for role, mid in muscles.items() if role in ("abduction", "adduction")} 
SIDE_SWING_MOTOR_IDS = RIGHT_SIDE_SWING_MOTOR_IDS | LEFT_SIDE_SWING_MOTOR_IDS


MOTOR_FEEDBACK_LIMITS_LEFT = {6: (-50.0, 500.0),
 4: (-600.0, 50.0),
 27: (-150.0, 100.0),
 7: (-50.0, 500.0),
 8: (-100.0, 150.0),
 14: (-600.0, 50.0),
 3: (-600.0, 50.0),
 25: (-50.0, 500.0),
 28: (-50.0, 50.0),
 1: (-50.0, 50.0),
 15: (-600.0, 50.0),
 16: (-600.0, 50.0),
 23: (-50.0, 500.0),
 2: (-600.0, 50.0),
 24: (-50.0, 50.0),
 9: (-50.0, 50.0),
 17: (-600.0, 50.0),
 21: (-50.0, 50.0),
 10: (-50.0, 500.0),
 22: (-50.0, 50.0),
 13: (-600.0, 50.0),
 11: (-50.0, 500.0),
 20: (-50.0, 50.0),
 18: (-50.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_RIGHT = {20: (-50.0, 500.0),
 18: (-400.0, 50.0),
 13: (-200.0, 400.0),
 21: (-50.0, 500.0),
 22: (-100.0, 400.0),
 28: (-400.0, 50.0),
 17: (-500.0, 50.0),
 11: (-50.0, 500.0),
 14: (-100.0, 100.0),
 15: (-100.0, 100.0),
 1: (-500.0, 50.0),
 2: (-500.0, 50.0),
 9: (-50.0, 500.0),
 16: (-500.0, 50.0),
 10: (-100.0, 100.0),
 23: (-100.0, 100.0),
 3: (-500.0, 50.0),
 7: (-50.0, 50.0),
 24: (-50.0, 500.0),
 8: (-50.0, 50.0),
 27: (-500.0, 50.0),
 25: (-50.0, 500.0),
 6: (-50.0, 50.0),
 4: (-50.0, 50.0)}


def _directional_tor_active(
    base_tor_active: float,
    default_tor: float,
    target_pos: float,
    current_pos: float,
) -> float:
    magnitude = abs(float(base_tor_active))
    position_error = float(target_pos) - float(current_pos)
    if position_error > 0.0:
        return magnitude
    if position_error < 0.0:
        return -magnitude
    return float(default_tor)


def _clip_scale(value, source_range, output_range):
    src_lo, src_hi = source_range
    out_lo, out_hi = output_range
    x = (float(value) - float(src_lo)) / (float(src_hi) - float(src_lo) + 1e-12)
    x = float(np.clip(x, 0.0, 1.0))
    return x * (float(out_hi) - float(out_lo)) + float(out_lo)


def compute_curl(q1: float, q2: float, q3: float, meta: dict):
    q_cal_meta = meta["q_calibration"]
    q1_cal = _clip_scale(
        q1,
        q_cal_meta["target_q_1_to_q1_cal"]["source_percentile_range"],
        q_cal_meta["target_q_1_to_q1_cal"]["output_range"],
    )
    q2_cal = _clip_scale(
        q2,
        q_cal_meta["target_q_2_to_q2_cal"]["source_percentile_range"],
        q_cal_meta["target_q_2_to_q2_cal"]["output_range"],
    )
    q3_cal = _clip_scale(
        q3,
        q_cal_meta["target_q_3_to_q3_cal"]["source_percentile_range"],
        q_cal_meta["target_q_3_to_q3_cal"]["output_range"],
    )

    fit = meta["linear_curl_fit"]
    weights = fit["weights"]
    curl = (
        float(fit["bias"])
        + float(weights["q1_cal"]) * q1_cal
        + float(weights["q2_cal"]) * q2_cal
        + float(weights["q3_cal"]) * q3_cal
    )
    curl = float(np.clip(curl, 0.0, 1.0))
    return float(curl), (float(q1_cal), float(q2_cal), float(q3_cal))


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std).astype(np.float32)

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.asarray(d["mean"], dtype=np.float32), std=np.asarray(d["std"], dtype=np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


if nn is not None:
    class HeadMLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_dims, dropout: float = 0.0):
            super().__init__()
            layers = []
            prev = int(in_dim)
            for h_raw in hidden_dims:
                h = int(h_raw)
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(float(dropout)))
                prev = h
            layers.append(nn.Linear(prev, int(out_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Q0CurlMultihead(nn.Module):
        def __init__(
            self,
            curl_in_dim: int,
            curl_out_dim: int,
            q0curl_in_dim: int,
            q0curl_out_dim: int,
            hidden_dims,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.curl_head = HeadMLP(curl_in_dim, curl_out_dim, hidden_dims, dropout)
            self.q0curl_head = HeadMLP(q0curl_in_dim, q0curl_out_dim, hidden_dims, dropout)

        def forward(self, x_curl, x_q0curl):
            return self.curl_head(x_curl), self.q0curl_head(x_q0curl)
else:
    class HeadMLP:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Cannot build model because torch/torch.nn is unavailable. "
                f"Import error: {TORCH_IMPORT_ERROR}"
            )

    class Q0CurlMultihead:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Cannot build model because torch/torch.nn is unavailable. "
                f"Import error: {TORCH_IMPORT_ERROR}"
            )


def _parse_csv_ints(raw: str):
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return [int(v) for v in vals]


def _parse_bool_arg(raw: str, arg_name: str):
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid value for {arg_name}: {raw!r}, expected true/false")


def _validate_finger_name(finger: str):
    value = str(finger).strip().lower()
    if value not in FINGER_RUNTIME_PRESETS:
        raise ValueError(
            f"Unsupported finger={finger!r}, expected one of {sorted(FINGER_RUNTIME_PRESETS.keys())}"
        )
    return value


def _validate_motor_ids(ids, arg_name="--model-motor-ids"):
    if not ids:
        raise ValueError(f"{arg_name} must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated motor ids in {arg_name}: {ids}")
    for mid in ids:
        if mid < 1 or mid > MOTOR_COUNT:
            raise ValueError(f"Invalid motor id {mid}, expected in [1, {MOTOR_COUNT}]")


def _validate_q_indices(ids, arg_name="--q-indices"):
    if not ids:
        raise ValueError(f"{arg_name} must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated q indices in {arg_name}: {ids}")
    for qi in ids:
        if qi < 0 or qi >= 20:
            raise ValueError(f"Invalid glove20 index {qi}, expected in [0, 19]")


def _mirror_motor_ids_right_to_left(motor_ids):
    return [int(RIGHT_TO_LEFT_MOTOR_MAP.get(int(mid), int(mid))) for mid in motor_ids]


def _is_side_swing_motor_id(motor_id: int, hand: str) -> bool:
    motors = LEFT_SIDE_SWING_MOTOR_IDS if hand == "left" else RIGHT_SIDE_SWING_MOTOR_IDS
    return int(motor_id) in motors


def _resolve_device(device: str) -> str:
    raw = str(device).strip().lower()
    if raw == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return raw


def _load_defaults_from_py(defaults_path: Path, hand: str):
    spec = importlib.util.spec_from_file_location("ring_curl_mlp_defaults_dynamic", str(defaults_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load defaults file: {defaults_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if defaults is None or hand not in defaults:
        raise RuntimeError(f"DEFAULTS or hand '{hand}' not found in {defaults_path}")

    hand_defaults = defaults[hand]
    out = {}
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.03)):
        values = list(hand_defaults.get(key, []))
        if len(values) < MOTOR_COUNT:
            values += [fallback] * (MOTOR_COUNT - len(values))
        elif len(values) > MOTOR_COUNT:
            values = values[:MOTOR_COUNT]
        out[key] = [float(v) for v in values]

    tor_active_values = list(hand_defaults.get("tor_active", out["tor"]))
    if len(tor_active_values) < MOTOR_COUNT:
        tor_active_values += [out["tor"][i] for i in range(len(tor_active_values), MOTOR_COUNT)]
    elif len(tor_active_values) > MOTOR_COUNT:
        tor_active_values = tor_active_values[:MOTOR_COUNT]
    out["tor_active"] = [float(v) for v in tor_active_values]
    return out


def _default_model_ckpt_for_finger(script_dir: Path, finger: str) -> Path:
    preset = FINGER_RUNTIME_PRESETS[_validate_finger_name(finger)]
    return script_dir.joinpath(*preset["model_ckpt_rel_parts"])


def _default_curl_meta_for_finger(script_dir: Path, finger: str) -> Path:
    preset = FINGER_RUNTIME_PRESETS[_validate_finger_name(finger)]
    return script_dir.joinpath(*preset["curl_meta_rel_parts"])


def _default_model_motor_ids_for_finger(finger: str):
    preset = FINGER_RUNTIME_PRESETS[_validate_finger_name(finger)]
    return list(preset["right_native_motor_ids"])


def _default_q_indices_for_finger(finger: str):
    preset = FINGER_RUNTIME_PRESETS[_validate_finger_name(finger)]
    return list(preset["q_indices"])


def _finger_arg(args, finger: str, suffix: str):
    raw = getattr(args, f"{finger}_{suffix}", "")
    if raw is None:
        return ""
    return str(raw).strip()


def _add_finger_cli_args(parser: argparse.ArgumentParser, finger: str):
    parser.add_argument(f"--enable-{finger}", dest=f"enable_{finger}", default=None)
    parser.add_argument(f"--{finger}-model-ckpt", dest=f"{finger}_model_ckpt", default=None)
    parser.add_argument(f"--{finger}-curl-meta", dest=f"{finger}_curl_meta", default=None)
    parser.add_argument(f"--{finger}-model-motor-ids", dest=f"{finger}_model_motor_ids", default=None)
    parser.add_argument(f"--{finger}-q-indices", dest=f"{finger}_q_indices", default=None)
    parser.add_argument(
        f"--{finger}-mirror-left-from-right",
        dest=f"{finger}_mirror_left_from_right",
        default=None,
    )


def _build_finger_specs(args, script_dir: Path):
    selected_finger = _validate_finger_name(args.finger)
    explicit_enable = any(getattr(args, f"enable_{finger}") is not None for finger in DEFAULT_FINGER_ORDER)
    specs = {}
    for finger in DEFAULT_FINGER_ORDER:
        enable_raw = getattr(args, f"enable_{finger}")
        if enable_raw is None:
            enable_raw = "true" if ((not explicit_enable) and finger == selected_finger) else "false"

        ckpt_raw = _finger_arg(args, finger, "model_ckpt")
        curl_meta_raw = _finger_arg(args, finger, "curl_meta")
        motor_ids_raw = _finger_arg(args, finger, "model_motor_ids")
        q_indices_raw = _finger_arg(args, finger, "q_indices")
        mirror_raw = _finger_arg(args, finger, "mirror_left_from_right")

        if finger == selected_finger:
            ckpt_raw = ckpt_raw or str(args.model_ckpt).strip()
            curl_meta_raw = curl_meta_raw or str(args.curl_meta).strip()
            motor_ids_raw = motor_ids_raw or str(args.model_motor_ids).strip()
            q_indices_raw = q_indices_raw or str(args.q_indices).strip()
            mirror_raw = mirror_raw or str(args.mirror_left_from_right).strip()

        enabled = _parse_bool_arg(enable_raw, f"--enable-{finger}")
        if not enabled:
            specs[finger] = {"enabled": False}
            continue

        if not ckpt_raw:
            ckpt_raw = str(_default_model_ckpt_for_finger(script_dir, finger))
        if not curl_meta_raw:
            curl_meta_raw = str(_default_curl_meta_for_finger(script_dir, finger))
        if not motor_ids_raw:
            motor_ids_raw = ",".join(str(v) for v in _default_model_motor_ids_for_finger(finger))
        if not q_indices_raw:
            q_indices_raw = ",".join(str(v) for v in _default_q_indices_for_finger(finger))
        if not mirror_raw:
            mirror_raw = "true"

        motor_ids = _parse_csv_ints(motor_ids_raw)
        q_indices = _parse_csv_ints(q_indices_raw)
        _validate_motor_ids(motor_ids, f"--{finger}-model-motor-ids")
        _validate_q_indices(q_indices, f"--{finger}-q-indices")
        mirror_left_from_right = _parse_bool_arg(
            mirror_raw,
            f"--{finger}-mirror-left-from-right",
        )

        specs[finger] = {
            "enabled": True,
            "model_ckpt": Path(ckpt_raw),
            "curl_meta_path": Path(curl_meta_raw),
            "model_motor_ids": motor_ids,
            "q_indices": q_indices,
            "mirror_left_from_right": mirror_left_from_right,
        }
    return specs


def _infer_mlp_dims_from_state_dict(state_dict):
    linear_items = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", str(key))
        shape = getattr(value, "shape", None)
        if match is None or shape is None or len(shape) != 2:
            continue
        linear_items.append((int(match.group(1)), int(shape[0]), int(shape[1])))
    linear_items.sort(key=lambda item: item[0])
    if not linear_items:
        raise RuntimeError("Cannot infer MLP structure from checkpoint: no net.<idx>.weight tensors found")
    in_dim = int(linear_items[0][2])
    hidden_dims = [int(item[1]) for item in linear_items[:-1]]
    out_dim = int(linear_items[-1][1])
    return in_dim, hidden_dims, out_dim


def _infer_mlp_dims_from_state_dict_prefix(state_dict, prefix: str):
    linear_items = []
    pattern = re.compile(rf"{re.escape(prefix)}net\.(\d+)\.weight")
    for key, value in state_dict.items():
        match = pattern.fullmatch(str(key))
        shape = getattr(value, "shape", None)
        if match is None or shape is None or len(shape) != 2:
            continue
        linear_items.append((int(match.group(1)), int(shape[0]), int(shape[1])))
    linear_items.sort(key=lambda item: item[0])
    if not linear_items:
        raise RuntimeError(
            f"Cannot infer MLP structure from checkpoint: no {prefix}net.<idx>.weight tensors found"
        )
    in_dim = int(linear_items[0][2])
    hidden_dims = [int(item[1]) for item in linear_items[:-1]]
    out_dim = int(linear_items[-1][1])
    return in_dim, hidden_dims, out_dim


def _parse_index_col(col: str, prefix: str, kind: str):
    if not isinstance(col, str) or not col.startswith(prefix):
        return None
    suffix = col[len(prefix):]
    try:
        return int(suffix)
    except Exception as exc:
        raise RuntimeError(f"Cannot parse {kind} column index from {col!r}") from exc


def _load_curl_meta(meta_path: Path):
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Curl meta file not found: {meta_path}. "
            "Please provide --curl-meta pointing to the supervised curl meta json."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "q_calibration" not in meta or "linear_curl_fit" not in meta:
        raise RuntimeError(
            f"Unsupported curl meta format: {meta_path}. "
            "Expected keys: q_calibration, linear_curl_fit."
        )
    return meta


def _infer_output_scaler_name(target_cols, available_scalers):
    target_indices = []
    for col in target_cols:
        idx = _parse_index_col(col, "target_m_", "target motor")
        if idx is None:
            raise RuntimeError(f"Unsupported target column {col!r}. Expected target_m_<i> naming.")
        target_indices.append(int(idx))
    scaler_name = "y_m" + "".join(str(idx) for idx in target_indices)
    if scaler_name not in available_scalers:
        raise RuntimeError(
            f"Cannot find output scaler {scaler_name!r} for target cols {target_cols}. "
            f"Available scalers: {sorted(available_scalers.keys())}"
        )
    return scaler_name


def _load_curl_model(ckpt_path: Path, device: str):
    if torch is None:
        raise RuntimeError(
            "Cannot import torch. Please install torch in the same Python used by launch.\n"
            f"Import error: {TORCH_IMPORT_ERROR}"
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    required_keys = ["config", "model_state"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")

    cfg = dict(ckpt["config"])
    state_dict = ckpt["model_state"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint model_state must be dict, got {type(state_dict)}")
    architecture = str(ckpt.get("architecture", ""))
    dropout = float(cfg.get("dropout", 0.0))

    if "scalers" in ckpt and "curl_head_input_cols" in ckpt and "q0curl_head_input_cols" in ckpt:
        scalers_raw = ckpt["scalers"]
        if not isinstance(scalers_raw, dict):
            raise RuntimeError(f"Checkpoint scalers must be dict, got {type(scalers_raw)}")

        curl_head_input_cols = list(ckpt.get("curl_head_input_cols", []))
        q0curl_head_input_cols = list(ckpt.get("q0curl_head_input_cols", []))
        curl_head_target_m_cols = list(ckpt.get("curl_head_target_m_cols", []))
        q0curl_head_target_m_cols = list(ckpt.get("q0curl_head_target_m_cols", []))
        target_m_cols = list(ckpt.get("target_m_cols", []))

        curl_in_dim, hidden_dims_curl, curl_out_dim = _infer_mlp_dims_from_state_dict_prefix(state_dict, "curl_head.")
        q0curl_in_dim, hidden_dims_q0curl, q0curl_out_dim = _infer_mlp_dims_from_state_dict_prefix(state_dict, "q0curl_head.")
        if hidden_dims_curl != hidden_dims_q0curl:
            raise RuntimeError(
                f"Head hidden dims mismatch in checkpoint: curl={hidden_dims_curl}, q0curl={hidden_dims_q0curl}"
            )
        hidden_dims = hidden_dims_curl

        if len(curl_head_input_cols) != curl_in_dim:
            raise RuntimeError(
                f"curl head input dim mismatch: len(curl_head_input_cols)={len(curl_head_input_cols)} vs model={curl_in_dim}"
            )
        if len(q0curl_head_input_cols) != q0curl_in_dim:
            raise RuntimeError(
                f"q0curl head input dim mismatch: len(q0curl_head_input_cols)={len(q0curl_head_input_cols)} vs model={q0curl_in_dim}"
            )
        unsupported_feature_cols = sorted(
            {
                str(col)
                for col in list(curl_head_input_cols) + list(q0curl_head_input_cols)
                if str(col) not in SUPPORTED_MODEL_FEATURE_COLS
            }
        )
        if unsupported_feature_cols:
            raise RuntimeError(
                f"Unsupported model input feature columns: {unsupported_feature_cols}. "
                f"Supported cols: {sorted(SUPPORTED_MODEL_FEATURE_COLS)}"
            )
        if len(curl_head_target_m_cols) != curl_out_dim:
            raise RuntimeError(
                f"curl head output dim mismatch: len(curl_head_target_m_cols)={len(curl_head_target_m_cols)} vs model={curl_out_dim}"
            )
        if len(q0curl_head_target_m_cols) != q0curl_out_dim:
            raise RuntimeError(
                f"q0curl head output dim mismatch: len(q0curl_head_target_m_cols)={len(q0curl_head_target_m_cols)} vs model={q0curl_out_dim}"
            )
        if not target_m_cols:
            target_m_cols = list(curl_head_target_m_cols) + list(q0curl_head_target_m_cols)

        scalers = {name: Standardizer.from_dict(val) for name, val in scalers_raw.items()}
        required_scalers = ("x_curl", "x_q0curl")
        missing_scalers = [name for name in required_scalers if name not in scalers]
        if missing_scalers:
            raise RuntimeError(f"Checkpoint missing scalers: {missing_scalers}")

        curl_output_scaler_name = _infer_output_scaler_name(curl_head_target_m_cols, scalers)
        q0curl_output_scaler_name = _infer_output_scaler_name(q0curl_head_target_m_cols, scalers)

        if len(scalers["x_curl"].mean) != curl_in_dim:
            raise RuntimeError(f"x_curl scaler dim mismatch: {len(scalers['x_curl'].mean)} vs model={curl_in_dim}")
        if len(scalers["x_q0curl"].mean) != q0curl_in_dim:
            raise RuntimeError(f"x_q0curl scaler dim mismatch: {len(scalers['x_q0curl'].mean)} vs model={q0curl_in_dim}")
        if len(scalers[curl_output_scaler_name].mean) != curl_out_dim:
            raise RuntimeError(
                f"{curl_output_scaler_name} scaler dim mismatch: "
                f"{len(scalers[curl_output_scaler_name].mean)} vs model={curl_out_dim}"
            )
        if len(scalers[q0curl_output_scaler_name].mean) != q0curl_out_dim:
            raise RuntimeError(
                f"{q0curl_output_scaler_name} scaler dim mismatch: "
                f"{len(scalers[q0curl_output_scaler_name].mean)} vs model={q0curl_out_dim}"
            )

        model = Q0CurlMultihead(
            curl_in_dim=curl_in_dim,
            curl_out_dim=curl_out_dim,
            q0curl_in_dim=q0curl_in_dim,
            q0curl_out_dim=q0curl_out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        target_plan = []
        for col in target_m_cols:
            target_idx = _parse_index_col(col, "target_m_", "target motor")
            if target_idx is None:
                raise RuntimeError(
                    f"Unsupported target column {col!r}. Expected target_m_<i> naming."
                )
            target_plan.append((target_idx, col))

        return {
            "model_kind": "q0_curl_multihead",
            "model": model,
            "scalers": scalers,
            "curl_head_input_cols": curl_head_input_cols,
            "curl_head_target_m_cols": curl_head_target_m_cols,
            "curl_output_scaler_name": curl_output_scaler_name,
            "q0curl_head_input_cols": q0curl_head_input_cols,
            "q0curl_head_target_m_cols": q0curl_head_target_m_cols,
            "q0curl_output_scaler_name": q0curl_output_scaler_name,
            "target_m_cols": target_m_cols,
            "target_plan": target_plan,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "architecture": architecture or "q0+curl multihead MLP",
            "config": cfg,
        }

    input_cols = list(ckpt.get("input_cols", []))
    target_m_cols = list(ckpt.get("target_m_cols", []))
    in_dim, hidden_dims, out_dim = _infer_mlp_dims_from_state_dict(state_dict)
    if not input_cols:
        input_cols = [str(cfg.get("input_col", "target_curl"))]
    if not target_m_cols:
        raw_target_cols = str(cfg.get("target_m_cols", "")).strip()
        if raw_target_cols:
            target_m_cols = [v.strip() for v in raw_target_cols.split(",") if v.strip()]

    if len(input_cols) != in_dim:
        raise RuntimeError(f"Input column count mismatch: len(input_cols)={len(input_cols)} vs model in_dim={in_dim}")
    if len(target_m_cols) != out_dim:
        raise RuntimeError(
            f"Target column count mismatch: len(target_m_cols)={len(target_m_cols)} vs model out_dim={out_dim}"
        )

    x_scaler = Standardizer.from_dict(ckpt["x_scaler"])
    y_scaler = Standardizer.from_dict(ckpt["y_scaler"])
    if len(x_scaler.mean) != in_dim:
        raise RuntimeError(f"x_scaler dim mismatch: {len(x_scaler.mean)} vs model in_dim={in_dim}")
    if len(y_scaler.mean) != out_dim:
        raise RuntimeError(f"y_scaler dim mismatch: {len(y_scaler.mean)} vs model out_dim={out_dim}")

    if input_cols != ["target_curl"]:
        raise RuntimeError(
            f"Unsupported input_cols in curl checkpoint: {input_cols}. "
            "Current runtime expects ['target_curl']."
        )

    target_plan = []
    for col in target_m_cols:
        target_idx = _parse_index_col(col, "target_m_", "target motor")
        if target_idx is None:
            raise RuntimeError(
                f"Unsupported target column {col!r}. Expected target_m_<i> naming."
            )
        target_plan.append((target_idx, col))

    model = HeadMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return {
        "model_kind": "curl_only_singlehead",
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": input_cols,
        "target_m_cols": target_m_cols,
        "target_plan": target_plan,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "architecture": architecture or "MLP over target_curl only",
        "config": cfg,
    }


class MultiFingerCurlMultiheadMLPMappingNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        glove_topic: str,
        glove_raw_skeleton_topic: str,
        right_non_thumb_q_from_raw_skeleton: bool,
        defaults_hand,
        finger_specs,
        device: str,
        control_period_sec: float,
        status_period_sec: float,
        glove_max_hz: float,
        glove_sub_queue_size: int,
        glove_pending_maxlen: int,
    ):
        super().__init__(f"non_thumb_curl_multihead_mlp_mapping_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError(f"Cannot import devices_pkg.msg: {DEVICES_MSG_IMPORT_ERROR}")
        if torch is None:
            raise RuntimeError(f"Cannot import torch: {TORCH_IMPORT_ERROR}")

        self.hand = hand
        self.hand_glove_slice = LEFT_HAND_GLOVE_SLICE if hand == "left" else RIGHT_HAND_GLOVE_SLICE
        self.defaults_hand = defaults_hand
        self.device = _resolve_device(device)
        self.control_period_sec = float(control_period_sec)
        self.status_period_sec = float(status_period_sec)
        self.glove_max_hz = float(glove_max_hz)
        self.glove_min_interval_sec = (1.0 / self.glove_max_hz) if self.glove_max_hz > 0.0 else 0.0
        self.glove_raw_skeleton_topic = glove_raw_skeleton_topic
        # Although the helper comes from replace_right_non_thumb_q_from_skeleton.py,
        # the actual q computation uses only local node transforms and can be reused
        # for either hand once the correct 176D hand chunk is selected.
        self.use_non_thumb_raw_skeleton_q = bool(right_non_thumb_q_from_raw_skeleton)
        if self.use_non_thumb_raw_skeleton_q and (not callable(compute_non_thumb_q16)):
            raise RuntimeError(
                "Cannot import replace_right_non_thumb_q_from_skeleton.py while "
                "raw-skeleton q replacement is enabled.\n"
                f"Import error: {RIGHT_NON_THUMB_Q_IMPORT_ERROR}"
            )

        self.motor_pos_limits = MOTOR_FEEDBACK_LIMITS_LEFT if self.hand == "left" else MOTOR_FEEDBACK_LIMITS_RIGHT
        self.finger_models = {}
        for finger in DEFAULT_FINGER_ORDER:
            spec = finger_specs.get(finger, {"enabled": False})
            if not spec.get("enabled", False):
                continue
            self.finger_models[finger] = self._load_finger_model(finger=finger, spec=spec)
        if not self.finger_models:
            raise RuntimeError("No finger model enabled. Set at least one --enable-ring=true or --enable-little=true")

        used_motors = {}
        for finger, bundle in self.finger_models.items():
            for motor_id in bundle["controlled_motor_ids"]:
                if motor_id in used_motors:
                    self.get_logger().warn(
                        f"Motor overlap detected: motor {motor_id} is used by both '{used_motors[motor_id]}' and '{finger}'. "
                        "Later finger in order will overwrite previous command."
                    )
                used_motors[motor_id] = finger
        self.controlled_motor_ids = sorted(used_motors.keys())

        self.publisher = self.create_publisher(YHandMsg, cmd_topic, 10)
        self.feedback_sub = self.create_subscription(YHandMsg, feedback_topic, self._feedback_cb, 10)
        self.glove_sub = self.create_subscription(
            Float64MultiArray,
            glove_topic,
            self._glove_cb,
            max(int(glove_sub_queue_size), 1),
        )
        self.glove_raw_sub = None
        if self.use_non_thumb_raw_skeleton_q:
            self.glove_raw_sub = self.create_subscription(
                Float64MultiArray,
                self.glove_raw_skeleton_topic,
                self._glove_raw_cb,
                max(int(glove_sub_queue_size), 1),
            )
        self.timer = self.create_timer(self.control_period_sec, self._publish_cb)
        self.status_timer = self.create_timer(self.status_period_sec, self._status_cb)
        self.raw_q_status_timer = None
        if self.use_non_thumb_raw_skeleton_q:
            self.raw_q_status_timer = self.create_timer(float(RAW_Q_STATUS_PERIOD_SEC), self._raw_q_status_cb)

        self.feedback_received = False
        self.latest_feedback_matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        self.latest_glove20 = None
        self.latest_raw_q_by_finger = {}
        self.glove_pending = deque(maxlen=max(int(glove_pending_maxlen), 1))
        self.last_feedback_ts = 0.0
        self.last_glove_ts = 0.0
        self.last_glove_accept_ts = 0.0
        self.last_glove_raw_ts = 0.0
        self.last_glove_raw_accept_ts = 0.0

        self.last_target_matrix = self.build_idle_tor_matrix()
        self.last_cmd_matrix = self._copy_matrix(self.last_target_matrix)
        self.tor_idle_latched = [False] * MOTOR_COUNT
        self.infer_last_ms = 0.0
        self.infer_ema_ms = 0.0
        self.clip_last = 0
        self.clip_total = 0
        self.glove_rx_total = 0
        self.glove_accept_total = 0
        self.glove_drop_rate_limited = 0
        self.glove_raw_rx_total = 0
        self.glove_raw_accept_total = 0
        self.glove_raw_drop_rate_limited = 0
        self._raw_side_mismatch_warned = False
        self.get_logger().info(
            "[non_thumb_curl_multihead_mlp_mapping] started "
            f"hand={self.hand}, glove_topic={glove_topic}, raw_topic={self.glove_raw_skeleton_topic if self.use_non_thumb_raw_skeleton_q else 'disabled'}, "
            f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}, "
            f"device={self.device}, enabled_fingers={list(self.finger_models.keys())}, "
            f"controlled_motor_ids={self.controlled_motor_ids}, "
            f"non_thumb_raw_skeleton={self.use_non_thumb_raw_skeleton_q}"
        )
        for finger in DEFAULT_FINGER_ORDER:
            if finger not in self.finger_models:
                continue
            b = self.finger_models[finger]
            self.get_logger().info(
                "[non_thumb_curl_multihead_mlp_mapping] "
                f"finger={finger}, ckpt={b['ckpt_path']}, curl_meta={b['curl_meta_path']}, "
                f"q_indices={b['q_indices']}, model_motor_ids={b['model_motor_ids']}, "
                f"controlled_motor_ids={b['controlled_motor_ids']}, "
                f"mirror_left_from_right={b['mirror_left_from_right']}, "
                f"target_m_cols={b['model_bundle']['target_m_cols']}"
            )

    def _load_finger_model(self, finger: str, spec: dict):
        model_motor_ids = list(spec["model_motor_ids"])
        if self.hand == "left" and spec["mirror_left_from_right"]:
            model_motor_ids = _mirror_motor_ids_right_to_left(model_motor_ids)

        model_bundle = _load_curl_model(spec["model_ckpt"], self.device)
        curl_meta = _load_curl_meta(spec["curl_meta_path"])
        q_indices = list(spec["q_indices"])
        if len(q_indices) != 4:
            raise ValueError(f"{finger} curl runtime expects exactly 4 q indices, got {q_indices}")

        max_target_idx = max(int(target_idx) for target_idx, _ in model_bundle["target_plan"])
        if len(model_motor_ids) <= max_target_idx:
            raise ValueError(
                f"{finger}: model_motor_ids is too short for checkpoint target_m indices: "
                f"len(model_motor_ids)={len(model_motor_ids)}, need > {max_target_idx}"
            )

        controlled_motor_ids = [
            int(model_motor_ids[int(target_idx)])
            for target_idx, _ in model_bundle["target_plan"]
        ]
        return {
            "finger": finger,
            "ckpt_path": Path(spec["model_ckpt"]),
            "curl_meta_path": Path(spec["curl_meta_path"]),
            "model_bundle": model_bundle,
            "curl_meta": curl_meta,
            "q_indices": q_indices,
            "model_motor_ids": model_motor_ids,
            "controlled_motor_ids": controlled_motor_ids,
            "mirror_left_from_right": bool(spec["mirror_left_from_right"]),
            "raw_not_ready_warned": False,
            "last_raw_q": np.zeros((4,), dtype=np.float32),
            "last_q_cal": np.zeros((3,), dtype=np.float32),
            "last_curl": 0.0,
            "last_pred_m": np.zeros((len(model_bundle["target_m_cols"]),), dtype=np.float32),
            "last_cmd_pos": np.zeros((len(model_bundle["target_m_cols"]),), dtype=np.float32),
        }

    def build_idle_tor_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append([0.0, 0.0, float(self.defaults_hand["tor"][i]), 0.0, 0.0])
        return matrix

    @staticmethod
    def _copy_matrix(matrix):
        return [
            [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            for row in matrix
        ]

    def _feedback_cb(self, msg):
        self.feedback_received = True
        self.last_feedback_ts = time.monotonic()
        matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        limit = min(len(motors), MOTOR_COUNT)
        for i in range(limit):
            m = motors[i]
            matrix[i, 0] = float(m.pos)
            matrix[i, 1] = float(m.vel)
            matrix[i, 2] = float(m.tor)
            matrix[i, 3] = float(m.kp)
            matrix[i, 4] = float(m.kd)
        self.latest_feedback_matrix = matrix

    def _glove_cb(self, msg):
        self.glove_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_accept_ts) < self.glove_min_interval_sec:
            self.glove_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        if len(data) == 4:
            if len(self.finger_models) != 1:
                return
            glove20 = np.zeros((20,), dtype=np.float32)
            only_bundle = next(iter(self.finger_models.values()))
            for value, q_index in zip(data, only_bundle["q_indices"]):
                glove20[int(q_index)] = float(value)
        elif len(data) == 20:
            glove20 = np.asarray(data, dtype=np.float32)
        elif len(data) >= 48:
            glove20 = np.asarray(data[self.hand_glove_slice], dtype=np.float32)
        else:
            return

        if glove20.shape[0] != 20:
            return

        self.latest_glove20 = glove20
        self.glove_pending.append(glove20)
        self.last_glove_ts = now
        self.last_glove_accept_ts = now
        self.glove_accept_total += 1

    def _extract_raw_skeleton_chunk(self, data):
        target_code = 1.0 if self.hand == "right" else 0.0
        if len(data) == RAW_SKELETON_ONE_HAND_DIM:
            try:
                side_code = float(data[0])
            except Exception:
                return None
            if abs(side_code - target_code) < 0.5:
                return data
            return None
        if len(data) < RAW_SKELETON_TWO_HAND_DIM:
            return None

        chunks = [
            data[0:RAW_SKELETON_ONE_HAND_DIM],
            data[RAW_SKELETON_ONE_HAND_DIM:RAW_SKELETON_TWO_HAND_DIM],
        ]
        for chunk in chunks:
            try:
                side_code = float(chunk[0])
            except Exception:
                continue
            if abs(side_code - target_code) < 0.5:
                return chunk
        return None

    def _glove_raw_cb(self, msg):
        self.glove_raw_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_raw_accept_ts) < self.glove_min_interval_sec:
            self.glove_raw_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        chunk = self._extract_raw_skeleton_chunk(data)
        if chunk is None:
            expected_side = "right" if self.hand == "right" else "left"
            if (not self._raw_side_mismatch_warned) and len(data) in (RAW_SKELETON_ONE_HAND_DIM, RAW_SKELETON_TWO_HAND_DIM):
                self.get_logger().warn(
                    f"[non_thumb_curl_multihead_mlp_mapping] ignore raw skeleton packet because it does not contain the expected {expected_side}-hand chunk."
                )
                self._raw_side_mismatch_warned = True
            return
        self._raw_side_mismatch_warned = False

        try:
            q16_map = compute_non_thumb_q16(chunk)
            raw_by_finger = {}
            for finger in self.finger_models.keys():
                q_vals = []
                for i in range(4):
                    key = f"{finger}_q_{i}"
                    if key not in q16_map:
                        raise KeyError(f"Missing key from compute_non_thumb_q16: {key}")
                    q_vals.append(float(q16_map[key]))
                raw_by_finger[finger] = np.asarray(q_vals, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(
                f"[non_thumb_curl_multihead_mlp_mapping] failed to compute {self.hand} non-thumb q from raw skeleton: {exc}"
            )
            return

        self.latest_raw_q_by_finger = raw_by_finger
        self.last_glove_raw_ts = now
        self.last_glove_raw_accept_ts = now
        self.glove_raw_accept_total += 1

    def _try_get_raw_q(self, finger: str):
        raw_q = self.latest_raw_q_by_finger.get(finger)
        if raw_q is None:
            return None
        return np.asarray(raw_q, dtype=np.float32)

    def _hold_last_cmd_for_controlled_motors(self, matrix, controlled_motor_ids):
        for mid in controlled_motor_ids:
            idx = int(mid) - 1
            prev = self.last_target_matrix[idx]
            matrix[idx] = [
                float(prev[0]),
                float(prev[1]),
                float(prev[2]),
                float(prev[3]),
                float(prev[4]),
            ]

    def _build_raw_q_from_glove20(self, glove20: np.ndarray, q_indices):
        return np.asarray([float(glove20[i]) for i in q_indices], dtype=np.float32)

    def _build_cmd_from_glove20(self, glove20: np.ndarray):
        matrix = self.build_idle_tor_matrix()
        clip_last = 0
        infer_t0 = time.monotonic()
        for finger in DEFAULT_FINGER_ORDER:
            if finger not in self.finger_models:
                continue
            b = self.finger_models[finger]
            if self.use_non_thumb_raw_skeleton_q:
                raw_q = self._try_get_raw_q(finger)
                if raw_q is None:
                    if not b["raw_not_ready_warned"]:
                        self.get_logger().warn(
                            f"[non_thumb_curl_multihead_mlp_mapping] {self.hand} {finger} raw skeleton q is not ready yet; "
                            "hold previous command for this finger this frame."
                        )
                        b["raw_not_ready_warned"] = True
                    self._hold_last_cmd_for_controlled_motors(matrix, b["controlled_motor_ids"])
                    continue
                b["raw_not_ready_warned"] = False
            else:
                raw_q = self._build_raw_q_from_glove20(glove20, b["q_indices"])

            q0 = float(raw_q[0])
            if self.hand == "left" and b["mirror_left_from_right"]:
                q0 = -q0
            curl, q_cal = compute_curl(raw_q[1], raw_q[2], raw_q[3], b["curl_meta"])
            model_bundle = b["model_bundle"]
            if model_bundle["model_kind"] == "q0_curl_multihead":
                x_curl_raw = np.asarray(
                    [[float(curl)]],
                    dtype=np.float32,
                )
                x_q0curl_raw = np.asarray(
                    [[float(q0), float(curl)]],
                    dtype=np.float32,
                )
                x_curl = model_bundle["scalers"]["x_curl"].transform(
                    x_curl_raw
                )
                x_q0curl = model_bundle["scalers"]["x_q0curl"].transform(
                    x_q0curl_raw
                )
                with torch.no_grad():
                    xb_curl = torch.tensor(x_curl, dtype=torch.float32, device=self.device)
                    xb_q0curl = torch.tensor(x_q0curl, dtype=torch.float32, device=self.device)
                    pred_curl_norm, pred_q0curl_norm = model_bundle["model"](xb_curl, xb_q0curl)
                pred_curl = model_bundle["scalers"][model_bundle["curl_output_scaler_name"]].inverse(
                    pred_curl_norm.cpu().numpy()
                )[0]
                pred_q0curl = model_bundle["scalers"][model_bundle["q0curl_output_scaler_name"]].inverse(
                    pred_q0curl_norm.cpu().numpy()
                )[0]
                pred_by_col = {}
                for col, value in zip(model_bundle["curl_head_target_m_cols"], pred_curl):
                    pred_by_col[str(col)] = float(value)
                for col, value in zip(model_bundle["q0curl_head_target_m_cols"], pred_q0curl):
                    pred_by_col[str(col)] = float(value)
                pred_m = np.asarray(
                    [float(pred_by_col[col]) for col in model_bundle["target_m_cols"]],
                    dtype=np.float32,
                )
            else:
                x = np.asarray([[curl]], dtype=np.float32)
                x_norm = model_bundle["x_scaler"].transform(x)
                with torch.no_grad():
                    xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                    pred_m_norm = model_bundle["model"](xb).cpu().numpy()
                pred_m = model_bundle["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)

            cmd_pos = []
            for out_col_idx, (target_idx, _target_col) in enumerate(model_bundle["target_plan"]):
                motor_id = int(b["model_motor_ids"][int(target_idx)])
                idx = motor_id - 1
                pred_pos = float(pred_m[out_col_idx])
                prev_cmd_pos = float(self.last_target_matrix[idx][0])
                if abs(pred_pos - prev_cmd_pos) <= CMD_POSITION_HOLD_DEADBAND_DEG:
                    pred_pos = prev_cmd_pos
                lim = self.motor_pos_limits.get(motor_id)
                if lim is not None:
                    pred_pos_clipped = float(np.clip(pred_pos, lim[0], lim[1]))
                    if abs(pred_pos_clipped - pred_pos) > 1e-6:
                        clip_last += 1
                    pred_pos = pred_pos_clipped

                matrix[idx] = [
                    pred_pos,
                    float(self.defaults_hand["vel"][idx]),
                    float(self.defaults_hand["tor_active"][idx]),
                    float(self.defaults_hand["kp"][idx]),
                    float(self.defaults_hand["kd"][idx]),
                ]
                cmd_pos.append(float(pred_pos))

            b["last_raw_q"] = raw_q.astype(np.float32)
            b["last_q_cal"] = np.asarray(q_cal, dtype=np.float32)
            b["last_curl"] = float(curl)
            b["last_pred_m"] = pred_m.astype(np.float32)
            b["last_cmd_pos"] = np.asarray(cmd_pos, dtype=np.float32)

        infer_ms = (time.monotonic() - infer_t0) * 1000.0
        self.infer_last_ms = infer_ms
        self.infer_ema_ms = infer_ms if self.infer_ema_ms <= 0.0 else (0.9 * self.infer_ema_ms + 0.1 * infer_ms)
        self.clip_last = int(clip_last)
        self.clip_total += int(clip_last)
        return matrix

    def _publish_matrix(self, matrix):
        msg = YHandMsg()
        msg.motors = [
            Motor(
                pos=float(r[0]),
                vel=float(r[1]),
                tor=float(r[2]),
                kp=float(r[3]),
                kd=float(r[4]),
            )
            for r in matrix
        ]
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.publisher.publish(msg)

    def publish_idle_once(self):
        self._publish_matrix(self.build_idle_tor_matrix())

    def _apply_directional_tor_active(self, matrix):
        for idx, row in enumerate(matrix):
            kp = float(row[3])
            kd = float(row[4])
            if abs(kp) <= 1e-9 and abs(kd) <= 1e-9:
                self.tor_idle_latched[idx] = False
                continue

            motor_id = idx + 1
            if _is_side_swing_motor_id(motor_id, self.hand):
                enter_deadband_deg = float(SIDE_SWING_TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(SIDE_SWING_TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)
            else:
                enter_deadband_deg = float(TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)

            current_pos = float(self.latest_feedback_matrix[idx, 0])
            if not np.isfinite(current_pos):
                continue

            position_error = float(row[0]) - current_pos
            abs_error = abs(position_error)
            if bool(self.tor_idle_latched[idx]):
                if abs_error > exit_deadband_deg:
                    self.tor_idle_latched[idx] = False
                else:
                    if _is_side_swing_motor_id(motor_id, self.hand):
                        row[1] = 0.0
                        row[2] = float(self.defaults_hand["tor"][idx])
                    else:
                        row[:] = [
                            0.0,
                            0.0,
                            float(self.defaults_hand["tor"][idx]),
                            0.0,
                            0.0,
                        ]
                    continue

            if abs_error <= enter_deadband_deg:
                self.tor_idle_latched[idx] = True
                if _is_side_swing_motor_id(motor_id, self.hand):
                    row[1] = 0.0
                    row[2] = float(self.defaults_hand["tor"][idx])
                else:
                    row[:] = [
                        0.0,
                        0.0,
                        float(self.defaults_hand["tor"][idx]),
                        0.0,
                        0.0,
                    ]
                continue

            row[2] = _directional_tor_active(
                self.defaults_hand["tor_active"][idx],
                self.defaults_hand["tor"][idx],
                row[0],
                current_pos,
            )
        return matrix

    def _publish_cb(self):
        if not self.feedback_received:
            target_matrix = self.build_idle_tor_matrix()
        else:
            if len(self.glove_pending) > 0:
                glove20_last = self.glove_pending[-1]
                self.glove_pending.clear()
                target_matrix = self._build_cmd_from_glove20(glove20_last)
            elif self.latest_glove20 is not None:
                target_matrix = self._copy_matrix(self.last_target_matrix)
            else:
                target_matrix = self.build_idle_tor_matrix()
        self.last_target_matrix = self._copy_matrix(target_matrix)
        matrix = self._copy_matrix(target_matrix)
        matrix = self._apply_directional_tor_active(matrix)
        self.last_cmd_matrix = self._copy_matrix(matrix)
        self._publish_matrix(matrix)

    def _format_five_finger_angles(self):
        if self.latest_glove20 is None:
            return "None"
        out = {}
        for finger, idxs in FIVE_FINGER_Q_GROUPS.items():
            out[finger] = [round(float(self.latest_glove20[i]), 3) for i in idxs]
        return out

    @staticmethod
    def _round_or_none(value):
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return round(value, 3)

    def _format_active_motor_log(self):
        out = {}
        for motor_id in self.controlled_motor_ids:
            idx = int(motor_id) - 1
            target_pos = self._round_or_none(self.last_target_matrix[idx][0])
            feedback_pos = self._round_or_none(self.latest_feedback_matrix[idx, 0])
            torque_fb = float(self.latest_feedback_matrix[idx, 2])
            torque_fb = self._round_or_none(torque_fb) if np.isfinite(torque_fb) else None
            out[int(motor_id)] = {
                "target_pos": target_pos,
                "feedback_pos": feedback_pos,
                "torque_fb": torque_fb,
            }
        return out

    def _raw_q_status_cb(self):
        if not self.use_non_thumb_raw_skeleton_q:
            return
        self.get_logger().info(
            "[non_thumb_curl_multihead_mlp_mapping] "
            f"raw_angles={self._format_five_finger_angles()}, "
            f"active_motors={self._format_active_motor_log()}"
        )

    def _status_cb(self):
        self.get_logger().info(
            "[non_thumb_curl_multihead_mlp_mapping] "
            f"raw_angles={self._format_five_finger_angles()}, "
            f"active_motors={self._format_active_motor_log()}"
        )

    def _raw_q_ready_all(self):
        if not self.use_non_thumb_raw_skeleton_q:
            return self.latest_glove20 is not None
        for finger in self.finger_models.keys():
            if finger not in self.latest_raw_q_by_finger:
                return False
        return True

    def wait_ready(self, timeout_sec: float):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            glove_ready = self._raw_q_ready_all()
            if self.feedback_received and glove_ready:
                return True
        return False


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Non-thumb finger q0+curl multihead mapping for ring/little models."
    )
    parser.add_argument("--finger", default="ring", choices=sorted(FINGER_RUNTIME_PRESETS.keys()))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--cmd-topic", default="", help="Override command topic")
    parser.add_argument("--feedback-topic", default="", help="Override feedback topic")
    parser.add_argument(
        "--glove-topic",
        default=DEFAULT_GLOVE_TOPIC,
        help="48D glove topic, 20D glove20 topic, or 4D finger-q topic",
    )
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default="/glove_joint_data_raw_skeleton",
        help="176D raw-skeleton topic (Float64MultiArray).",
    )
    raw_non_thumb_group = parser.add_mutually_exclusive_group()
    raw_non_thumb_group.add_argument(
        "--right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_true",
        default=True,
        help="Solve the selected finger q_0..q_3 from the selected hand's 176D raw skeleton instead of /glove_joint_data indices (default: enabled).",
    )
    raw_non_thumb_group.add_argument(
        "--no-right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_false",
        help="Disable raw-skeleton q replacement and use glove48/glove20 q indices.",
    )
    parser.add_argument(
        "--model-ckpt",
        default="",
        help="Path to q0+curl multihead checkpoint. If empty, use the selected finger default.",
    )
    parser.add_argument(
        "--curl-meta",
        default="",
        help="Path to supervised curl meta json used to convert q1/q2/q3 into target_curl. If empty, use the selected finger default.",
    )
    parser.add_argument(
        "--model-motor-ids",
        default="",
        help="Right-hand-native motor order used by target_m_i indices. If empty, use the selected finger default.",
    )
    parser.add_argument(
        "--q-indices",
        default="",
        help="Finger glove20 indices (hand-local q_0..q_3). If empty, use the selected finger default.",
    )
    parser.add_argument(
        "--mirror-left-from-right",
        default="true",
        help="Legacy alias for the selected finger mirror mode when hand=left",
    )
    for finger in DEFAULT_FINGER_ORDER:
        _add_finger_cli_args(parser, finger)
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
        help="Path to muscle_adjustment_defaults.py",
    )
    parser.add_argument(
        "--control-period-sec",
        type=float,
        default=float(CONTROL_PERIOD_SEC),
        help="Command publish period (sec)",
    )
    parser.add_argument(
        "--status-period-sec",
        type=float,
        default=2.0,
        help="Status print period (sec)",
    )
    parser.add_argument(
        "--glove-max-hz",
        type=float,
        default=100.0,
        help="Max accepted glove callback rate. <=0 disables rate limiting",
    )
    parser.add_argument("--glove-sub-queue-size", type=int, default=60)
    parser.add_argument("--glove-pending-maxlen", type=int, default=60)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--require-ready", action="store_true", help="Exit if startup is not ready before timeout")
    parser.add_argument("--ready-timeout-sec", type=float, default=12.0)
    return parser.parse_args()


def main():
    args = parse_args()
    hand = args.hand.strip().lower()
    cmd_topic, feedback_topic = get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic.strip()
    if args.feedback_topic:
        feedback_topic = args.feedback_topic.strip()

    script_dir = Path(__file__).resolve().parent
    finger_specs = _build_finger_specs(args, script_dir)
    defaults_hand = _load_defaults_from_py(Path(args.defaults_path), hand)

    rclpy.init()
    node = None
    try:
        node = MultiFingerCurlMultiheadMLPMappingNode(
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            glove_topic=args.glove_topic.strip(),
            glove_raw_skeleton_topic=args.glove_raw_skeleton_topic.strip(),
            right_non_thumb_q_from_raw_skeleton=bool(args.right_non_thumb_q_from_raw_skeleton),
            defaults_hand=defaults_hand,
            finger_specs=finger_specs,
            device=args.device.strip().lower(),
            control_period_sec=float(args.control_period_sec),
            status_period_sec=float(args.status_period_sec),
            glove_max_hz=float(args.glove_max_hz),
            glove_sub_queue_size=int(args.glove_sub_queue_size),
            glove_pending_maxlen=int(args.glove_pending_maxlen),
        )

        if args.require_ready:
            ok = node.wait_ready(timeout_sec=float(args.ready_timeout_sec))
            if not ok:
                raise RuntimeError(
                    "Startup not ready within timeout: "
                    f"feedback_ready={node.feedback_received}, "
                    f"glove_ready={node.latest_glove20 is not None}, "
                    f"raw_q_ready={node._raw_q_ready_all()}"
                )

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[non_thumb_curl_multihead_mlp_mapping] KeyboardInterrupt received, exiting...")
    finally:
        if node is not None:
            try:
                node.publish_idle_once()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
