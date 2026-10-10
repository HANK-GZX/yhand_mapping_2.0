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
    LEFT_TO_RIGHT_MOTOR_MAP, RIGHT_TO_LEFT_MOTOR_MAP,
)

from hand_motor_map import (
    model_motor_ids_csv,
)

LEFT_HAND_GLOVE_SLICE = slice(8, 28)
RIGHT_HAND_GLOVE_SLICE = slice(28, 48)
MOTOR_COUNT = 28
RAW_SKELETON_ONE_HAND_DIM = 176
RAW_SKELETON_TWO_HAND_DIM = 352
RAW_Q_STATUS_PERIOD_SEC = 1.0
TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG = 1.5
TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG = 3.0

DEFAULT_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
RIGHT_NON_THUMB_FINGERS = ("index", "middle", "ring", "little")


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


# Shared anatomical correspondence is imported above.
# Shared anatomical correspondence is imported above.

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

FINGER_DEFAULTS_BY_HAND = {
    "left": {
        "thumb": {
            "enable_default": "false",
            "model_ckpt_rel_parts": None,
            "motor_ids_csv": model_motor_ids_csv('left', 'thumb'),
            "q_indices_csv": "0,1,2,3",
            "mirror_right_from_left_default": "false",
        },
        "index": {
            "enable_default": "false",
            "model_ckpt_rel_parts": None,
            "motor_ids_csv": model_motor_ids_csv('left', 'index'),
            "q_indices_csv": "4,5,6,7",
            "mirror_right_from_left_default": "false",
        },
        "middle": {
            "enable_default": "false",
            "model_ckpt_rel_parts": None,
            "motor_ids_csv": model_motor_ids_csv('left', 'middle'),
            "q_indices_csv": "8,9,10,11",
            "mirror_right_from_left_default": "false",
        },
        "ring": {
            "enable_default": "false",
            "model_ckpt_rel_parts": None,
            "motor_ids_csv": model_motor_ids_csv('left', 'ring'),
            "q_indices_csv": "12,13,14,15",
            "mirror_right_from_left_default": "false",
        },
        "little": {
            "enable_default": "false",
            "model_ckpt_rel_parts": None,
            "motor_ids_csv": model_motor_ids_csv('left', 'little'),
            "q_indices_csv": "16,17,18,19",
            "mirror_right_from_left_default": "false",
        },
    },
    "right": {
        "thumb": {
            "enable_default": "true",
            "model_ckpt_rel_parts": (
                "mp_right",
                "thumb_delta_multihead_masked",
                "best_thumb_prev_m_target_q_mlp.pt",
            ),
            "motor_ids_csv": model_motor_ids_csv('right', 'thumb'),
            "q_indices_csv": "0,1,2,3",
            "mirror_right_from_left_default": "false",
        },
        "index": {
            "enable_default": "false",
            "model_ckpt_rel_parts": (
                "mp_right",
                "index_delta_multihead_masked",
                "best_index_prev_m_target_q_mlp.pt",
            ),
            "motor_ids_csv": model_motor_ids_csv('right', 'index'),
            "q_indices_csv": "4,5,6,7",
            "mirror_right_from_left_default": "false",
        },
        "middle": {
            "enable_default": "false",
            "model_ckpt_rel_parts": (
                "mp_right",
                "middle_delta_multihead_masked",
                "best_middle_prev_m_target_q_mlp.pt",
            ),
            "motor_ids_csv": model_motor_ids_csv('right', 'middle'),
            "q_indices_csv": "8,9,10,11",
            "mirror_right_from_left_default": "false",
        },
        "ring": {
            "enable_default": "false",
            "model_ckpt_rel_parts": (
                "mp_right",
                "ring_delta_multihead_masked",
                "best_ring_prev_m_target_q_mlp.pt",
            ),
            "motor_ids_csv": model_motor_ids_csv('right', 'ring'),
            "q_indices_csv": "12,13,14,15",
            "mirror_right_from_left_default": "false",
        },
        "little": {
            "enable_default": "false",
            "model_ckpt_rel_parts": (
                "mp_right",
                "little_delta_multihead_masked",
                "best_little_prev_m_target_q_mlp.pt",
            ),
            "motor_ids_csv": model_motor_ids_csv('right', 'little'),
            "q_indices_csv": "16,17,18,19",
            "mirror_right_from_left_default": "false",
        },
    },
}


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std).astype(np.float32)

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.array(d["mean"], dtype=np.float32), std=np.array(d["std"], dtype=np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


if nn is not None:
    class PrevMTargetQMLP(nn.Module):
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
else:
    class PrevMTargetQMLP:  # pragma: no cover
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


def _validate_motor_ids(ids, arg_name):
    if not ids:
        raise ValueError(f"{arg_name} must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated motor ids in {arg_name}: {ids}")
    for mid in ids:
        if mid < 1 or mid > 28:
            raise ValueError(f"Invalid motor id {mid} in {arg_name}, expected in [1, 28]")


def _validate_q_indices(ids, arg_name):
    if not ids:
        raise ValueError(f"{arg_name} must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated q indices in {arg_name}: {ids}")
    for qi in ids:
        if qi < 0 or qi >= 20:
            raise ValueError(f"Invalid glove20 index {qi} in {arg_name}, expected in [0, 19]")


def _load_defaults_from_py(defaults_path: Path, hand: str):
    spec = importlib.util.spec_from_file_location("prev_m_target_q_defaults_dynamic", str(defaults_path))
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


def _parse_motor_feedback_limits_json(raw_json: str, arg_name: str):
    text = str(raw_json).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object: {{\"motor_id\": [lo, hi], ...}}")

    limits = {}
    for motor_id_raw, bound in parsed.items():
        try:
            motor_id = int(motor_id_raw)
        except Exception as exc:
            raise ValueError(f"{arg_name} has non-integer motor id key: {motor_id_raw!r}") from exc
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise ValueError(f"{arg_name} motor {motor_id} must map to [lo, hi], got: {bound!r}")
        lo = float(bound[0])
        hi = float(bound[1])
        if lo > hi:
            raise ValueError(f"{arg_name} motor {motor_id} has invalid range: lo={lo} > hi={hi}")
        limits[motor_id] = (lo, hi)
    return limits


def _parse_hand_index_offset_map_json(raw_json: str, arg_name: str, id_name: str, lo: int, hi: int):
    text = str(raw_json).strip()
    if not text:
        return {"left": {}, "right": {}}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{arg_name} must be JSON object: "
            "{\"left\": {\"0\": 0.0}, \"right\": {\"0\": 0.0}}"
        )

    out = {"left": {}, "right": {}}
    for hand_raw, offset_map in parsed.items():
        hand = str(hand_raw).strip().lower()
        if hand not in ("left", "right"):
            raise ValueError(f"{arg_name} has unsupported hand key: {hand_raw!r}, expected left/right")
        if not isinstance(offset_map, dict):
            raise ValueError(f"{arg_name}[{hand}] must be an object mapping {id_name}->offset")
        parsed_map = {}
        for idx_raw, off_raw in offset_map.items():
            try:
                idx = int(idx_raw)
            except Exception as exc:
                raise ValueError(f"{arg_name}[{hand}] has invalid {id_name} key: {idx_raw!r}") from exc
            if idx < lo or idx > hi:
                raise ValueError(f"{arg_name}[{hand}] {id_name} out of range: {idx}, expected [{lo},{hi}]")
            parsed_map[idx] = float(off_raw)
        out[hand] = parsed_map
    return out


def _parse_q_offset_map_json(raw_json: str, arg_name: str):
    return _parse_hand_index_offset_map_json(
        raw_json=raw_json,
        arg_name=arg_name,
        id_name="q_index",
        lo=0,
        hi=19,
    )


def _parse_motor_cmd_offset_map_json(raw_json: str, arg_name: str):
    return _parse_hand_index_offset_map_json(
        raw_json=raw_json,
        arg_name=arg_name,
        id_name="motor_id",
        lo=1,
        hi=28,
    )


def _mirror_motors_left_to_right(motor_ids):
    return [int(LEFT_TO_RIGHT_MOTOR_MAP.get(int(mid), int(mid))) for mid in motor_ids]


def _mirror_motors_right_to_left(motor_ids):
    return [int(RIGHT_TO_LEFT_MOTOR_MAP.get(int(mid), int(mid))) for mid in motor_ids]


def _build_motor_pos_limit_map(hand: str, limits_left=None, limits_right=None):
    hand_norm = str(hand).strip().lower()
    if hand_norm == "left":
        raw_limits = dict(MOTOR_FEEDBACK_LIMITS_LEFT if limits_left is None else limits_left)
        mode = "left_default" if limits_left is None else "left_cli_override"
    elif hand_norm == "right":
        raw_limits = dict(MOTOR_FEEDBACK_LIMITS_RIGHT if limits_right is None else limits_right)
        mode = "right_default" if limits_right is None else "right_cli_override"
    else:
        raise RuntimeError(f"Unsupported hand for motor limits: {hand!r}")

    limits = {}
    for mid, bound in raw_limits.items():
        lo = float(bound[0])
        hi = float(bound[1])
        if lo > hi:
            raise RuntimeError(f"Invalid limit for motor {mid}: lo={lo} > hi={hi}")
        limits[int(mid)] = (lo, hi)
    return limits, mode


def _infer_mlp_dims_from_state_dict(state_dict):
    linear_items = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", str(key))
        shape = getattr(value, "shape", None)
        if match is None or shape is None or len(shape) != 2:
            continue
        linear_items.append((int(match.group(1)), int(shape[0]), int(shape[1])))
    linear_items.sort(key=lambda item: item[0])
    if len(linear_items) < 1:
        raise RuntimeError("Cannot infer MLP structure from checkpoint: no net.<idx>.weight tensors found")
    in_dim = int(linear_items[0][2])
    hidden_dims = [int(item[1]) for item in linear_items[:-1]]
    out_dim = int(linear_items[-1][1])
    return in_dim, hidden_dims, out_dim


def _parse_index_col(col: str, prefix: str, dim: int, kind: str):
    if not isinstance(col, str) or not col.startswith(prefix):
        return None
    suffix = col[len(prefix):]
    try:
        idx = int(suffix)
    except Exception as exc:
        raise RuntimeError(f"Cannot parse {kind} column index from {col!r}") from exc
    if idx < 0 or idx >= int(dim):
        raise RuntimeError(f"{kind} column out of range for dim={dim}: {col!r}")
    return idx


def _load_prev_m_target_q_mlp_model(ckpt_path: Path, device: str):
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

    required_keys = ["config", "model_state", "x_scaler", "y_scaler"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")

    cfg = dict(ckpt["config"])
    q_dim = int(cfg.get("q_dim", 4))
    m_dim = int(cfg.get("m_dim", 6))
    dropout = float(cfg.get("dropout", 0.0))
    history_m_prefix = str(cfg.get("history_m_prefix", "history_m_"))
    target_q_prefix = str(cfg.get("target_q_prefix", "target_q_"))
    target_m_prefix = str(cfg.get("target_m_prefix", "target_m_"))

    input_cols = list(
        ckpt.get(
            "input_cols",
            [f"{history_m_prefix}{i}" for i in range(m_dim)]
            + [f"{target_q_prefix}{i}" for i in range(q_dim)],
        )
    )
    target_m_cols = list(ckpt.get("target_m_cols", [f"{target_m_prefix}{i}" for i in range(m_dim)]))

    state_dict = ckpt["model_state"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint model_state must be dict, got {type(state_dict)}")

    in_dim, hidden_dims, out_dim = _infer_mlp_dims_from_state_dict(state_dict)
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

    input_plan = []
    for col in input_cols:
        history_idx = _parse_index_col(col, history_m_prefix, m_dim, "history motor")
        if history_idx is not None:
            input_plan.append(("history_m", history_idx, col))
            continue
        target_q_idx = _parse_index_col(col, target_q_prefix, q_dim, "target q")
        if target_q_idx is not None:
            input_plan.append(("target_q", target_q_idx, col))
            continue
        raise RuntimeError(
            f"Unsupported input column {col!r}. Expected prefixes "
            f"{history_m_prefix!r} or {target_q_prefix!r}."
        )

    target_plan = []
    for col in target_m_cols:
        target_idx = _parse_index_col(col, target_m_prefix, m_dim, "target motor")
        if target_idx is None:
            raise RuntimeError(f"Unsupported target column {col!r}. Expected prefix {target_m_prefix!r}.")
        target_plan.append((target_idx, col))

    model = PrevMTargetQMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Incompatible checkpoint for current MLP definition: {ckpt_path}. "
            "Please export with net Linear/LayerNorm/GELU/Dropout blocks matching this runtime."
        ) from exc
    model.eval()

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "q_dim": q_dim,
        "m_dim": m_dim,
        "input_cols": input_cols,
        "target_m_cols": target_m_cols,
        "input_plan": input_plan,
        "target_plan": target_plan,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "architecture": ckpt.get("architecture", "MLP over one history motor frame plus target_q"),
    }


def _default_model_ckpt_from_parts(script_dir: Path, rel_parts):
    if not rel_parts:
        return ""
    return str(script_dir.joinpath(*rel_parts))


def _finger_arg(args, finger_name: str, suffix: str):
    return str(getattr(args, f"{finger_name}_{suffix}", "")).strip()


def _build_finger_specs(args, hand: str, script_dir: Path):
    specs = {}
    defaults = FINGER_DEFAULTS_BY_HAND[hand]
    for finger_name in DEFAULT_FINGER_ORDER:
        d = defaults[finger_name]
        enable_raw = _finger_arg(args, finger_name, "enable")
        ckpt_raw = _finger_arg(args, finger_name, "model_ckpt")
        motor_ids_raw = _finger_arg(args, finger_name, "model_motor_ids")
        q_indices_raw = _finger_arg(args, finger_name, "q_indices")
        mirror_raw = _finger_arg(args, finger_name, "mirror_right_from_left")

        if finger_name == "thumb":
            enable_raw = enable_raw or str(args.enable_model).strip()
            ckpt_raw = ckpt_raw or str(args.model_ckpt).strip()
            motor_ids_raw = motor_ids_raw or str(args.model_motor_ids).strip()
            q_indices_raw = q_indices_raw or str(args.q_indices).strip()
            mirror_raw = mirror_raw or str(args.mirror_right_from_left).strip()

        enable_raw = enable_raw or d["enable_default"]
        ckpt_raw = ckpt_raw or _default_model_ckpt_from_parts(script_dir, d["model_ckpt_rel_parts"])
        motor_ids_raw = motor_ids_raw or d["motor_ids_csv"]
        q_indices_raw = q_indices_raw or d["q_indices_csv"]
        mirror_raw = mirror_raw or d["mirror_right_from_left_default"]

        enabled = _parse_bool_arg(enable_raw, f"--enable-{finger_name}")
        mirror_right_from_left = _parse_bool_arg(mirror_raw, f"--{finger_name}-mirror-right-from-left")

        if not enabled:
            specs[finger_name] = {"enabled": False}
            continue

        if not ckpt_raw:
            raise ValueError(
                f"{finger_name}: enabled but no checkpoint path is configured. "
                f"Pass --{finger_name}-model-ckpt or disable with --enable-{finger_name}=false."
            )

        motor_ids = _parse_csv_ints(motor_ids_raw)
        q_indices = _parse_csv_ints(q_indices_raw)
        _validate_motor_ids(motor_ids, f"--{finger_name}-model-motor-ids")
        _validate_q_indices(q_indices, f"--{finger_name}-q-indices")

        specs[finger_name] = {
            "enabled": True,
            "model_ckpt": str(ckpt_raw),
            "motor_ids": motor_ids,
            "q_indices": q_indices,
            "mirror_right_from_left": mirror_right_from_left,
        }
    return specs


class PrevMTargetQMLPMappingNode(Node):
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
        motor_feedback_limits_left,
        motor_feedback_limits_right,
        q_offset_map,
        motor_cmd_offset_map,
        control_period_sec: float,
        status_period_sec: float,
        glove_max_hz: float,
        glove_sub_queue_size: int,
        glove_pending_maxlen: int,
    ):
        super().__init__(f"prev_m_target_q_mlp_mapping_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError(f"Cannot import devices_pkg.msg: {DEVICES_MSG_IMPORT_ERROR}")
        if torch is None:
            raise RuntimeError(
                "Cannot import torch. Please install torch in the same Python used by launch.\n"
                f"Import error: {TORCH_IMPORT_ERROR}"
            )

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.glove_topic = glove_topic
        self.glove_raw_skeleton_topic = glove_raw_skeleton_topic
        self.use_right_non_thumb_raw_skeleton_q = bool(right_non_thumb_q_from_raw_skeleton and hand == "right")
        if self.use_right_non_thumb_raw_skeleton_q and (not callable(compute_non_thumb_q16)):
            raise RuntimeError(
                "Cannot import replace_right_non_thumb_q_from_skeleton.py while "
                "--right-non-thumb-q-from-raw-skeleton is enabled.\n"
                f"Import error: {RIGHT_NON_THUMB_Q_IMPORT_ERROR}"
            )
        self.defaults_hand = defaults_hand
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.hand_glove_slice = LEFT_HAND_GLOVE_SLICE if hand == "left" else RIGHT_HAND_GLOVE_SLICE
        self.q_offsets_this_hand = dict(q_offset_map.get(hand, {}))
        self.motor_cmd_offsets_this_hand = dict(motor_cmd_offset_map.get(hand, {}))
        self.motor_pos_limits, self.motor_pos_limit_mode = _build_motor_pos_limit_map(
            hand=self.hand,
            limits_left=motor_feedback_limits_left,
            limits_right=motor_feedback_limits_right,
        )
        self.control_period_sec = float(control_period_sec)
        self.glove_max_hz = float(glove_max_hz)
        self.glove_min_interval_sec = (1.0 / self.glove_max_hz) if self.glove_max_hz > 0.0 else 0.0
        self.glove_sub_queue_size = max(1, int(glove_sub_queue_size))
        self.glove_pending_maxlen = max(1, int(glove_pending_maxlen))

        self.finger_models = {}
        for name in DEFAULT_FINGER_ORDER:
            spec = finger_specs[name]
            if not spec["enabled"]:
                continue
            self.finger_models[name] = self._load_finger_model(name=name, spec=spec)
        if not self.finger_models:
            raise RuntimeError("No finger model enabled. Set at least one --enable-<finger>=true")

        used_motors = {}
        for fname, bundle in self.finger_models.items():
            for mid in bundle["motor_ids"]:
                if mid in used_motors:
                    self.get_logger().warn(
                        f"Motor overlap detected: motor {mid} is used by both '{used_motors[mid]}' and '{fname}'. "
                        "Later finger in order will overwrite previous command."
                    )
                used_motors[mid] = fname

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.glove_sub = self.create_subscription(
            Float64MultiArray,
            self.glove_topic,
            self._glove_cb,
            self.glove_sub_queue_size,
        )
        self.glove_raw_sub = None
        if self.use_right_non_thumb_raw_skeleton_q:
            self.glove_raw_sub = self.create_subscription(
                Float64MultiArray,
                self.glove_raw_skeleton_topic,
                self._glove_raw_cb,
                self.glove_sub_queue_size,
            )
        self.publish_timer = self.create_timer(float(control_period_sec), self._publish_cb)
        self.status_timer = self.create_timer(float(status_period_sec), self._status_cb)
        self.raw_q_status_timer = None
        if self.use_right_non_thumb_raw_skeleton_q:
            self.raw_q_status_timer = self.create_timer(float(RAW_Q_STATUS_PERIOD_SEC), self._raw_q_status_cb)

        self.latest_glove20 = None
        self.latest_right_non_thumb_q16 = None
        self.last_glove_ts = 0.0
        self.last_glove_accept_ts = 0.0
        self.last_glove_raw_ts = 0.0
        self.last_glove_raw_accept_ts = 0.0
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.latest_feedback_matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        self.tor_idle_latched = np.zeros((MOTOR_COUNT,), dtype=np.bool_)
        self.glove_rx_total = 0
        self.glove_accept_total = 0
        self.glove_drop_rate_limited = 0
        self.glove_raw_rx_total = 0
        self.glove_raw_accept_total = 0
        self.glove_raw_drop_rate_limited = 0
        self.glove_pending = deque(maxlen=self.glove_pending_maxlen)

        self.last_cmd_matrix = self.build_idle_tor_matrix()
        self.clip_total = 0
        self.clip_last = 0
        self.infer_last_ms = 0.0
        self.infer_ema_ms = 0.0

        self.get_logger().info(
            "[prev_m_target_q_mlp_mapping] started "
            f"hand={self.hand}, cmd={self.cmd_topic}, feedback={self.feedback_topic}, glove={self.glove_topic}, "
            f"glove_max_hz={self.glove_max_hz}, glove_sub_queue={self.glove_sub_queue_size}, "
            f"glove_pending_maxlen={self.glove_pending_maxlen}, consume_latest_only=true, "
            f"right_non_thumb_raw_skeleton={self.use_right_non_thumb_raw_skeleton_q}, "
            f"raw_topic={self.glove_raw_skeleton_topic if self.use_right_non_thumb_raw_skeleton_q else 'disabled'}, "
            f"raw_q_log_period_sec={RAW_Q_STATUS_PERIOD_SEC if self.use_right_non_thumb_raw_skeleton_q else 'disabled'}"
        )
        self.get_logger().info(
            "[prev_m_target_q_mlp_mapping] "
            f"motor_pos_limit_mode={self.motor_pos_limit_mode}, "
            f"motor_pos_limit_ids={sorted(self.motor_pos_limits.keys())}, "
            f"q_offsets={self.q_offsets_this_hand}, motor_cmd_offsets={self.motor_cmd_offsets_this_hand}"
        )
        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]
            self.get_logger().info(
                "[prev_m_target_q_mlp_mapping] "
                f"finger={name}, ckpt={b['ckpt']}, architecture={b['architecture']}, "
                f"dims={len(b['input_cols'])}->{b['hidden_dims']}->{len(b['target_m_cols'])}, "
                f"q_indices={b['q_indices']}, motor_ids={b['motor_ids']}, "
                f"input_cols={b['input_cols']}, target_m_cols={b['target_m_cols']}"
            )

    def _load_finger_model(self, name: str, spec: dict):
        motor_ids = list(spec["motor_ids"])
        if spec["mirror_right_from_left"]:
            if self.hand == "right":
                motor_ids = _mirror_motors_left_to_right(motor_ids)
            else:
                motor_ids = _mirror_motors_right_to_left(motor_ids)

        model_bundle = _load_prev_m_target_q_mlp_model(Path(spec["model_ckpt"]), self.device)
        q_indices = list(spec["q_indices"])

        if len(q_indices) != int(model_bundle["q_dim"]):
            raise ValueError(
                f"{name}: input dim mismatch len(q_indices)={len(q_indices)} vs q_dim={model_bundle['q_dim']}"
            )
        if len(motor_ids) != int(model_bundle["m_dim"]):
            raise ValueError(
                f"{name}: output dim mismatch len(motor_ids)={len(motor_ids)} vs m_dim={model_bundle['m_dim']}"
            )

        return {
            "name": name,
            "ckpt": str(spec["model_ckpt"]),
            "model": model_bundle["model"],
            "x_scaler": model_bundle["x_scaler"],
            "y_scaler": model_bundle["y_scaler"],
            "q_dim": int(model_bundle["q_dim"]),
            "m_dim": int(model_bundle["m_dim"]),
            "input_cols": list(model_bundle["input_cols"]),
            "target_m_cols": list(model_bundle["target_m_cols"]),
            "input_plan": list(model_bundle["input_plan"]),
            "target_plan": list(model_bundle["target_plan"]),
            "hidden_dims": list(model_bundle["hidden_dims"]),
            "dropout": float(model_bundle["dropout"]),
            "architecture": model_bundle["architecture"],
            "q_indices": q_indices,
            "motor_ids": motor_ids,
            "last_history_m": np.zeros((int(model_bundle["m_dim"]),), dtype=np.float32),
            "last_target_q": np.zeros((int(model_bundle["q_dim"]),), dtype=np.float32),
            "last_pred_m": np.zeros((len(model_bundle["target_m_cols"]),), dtype=np.float32),
            "last_cmd_pos": np.zeros((int(model_bundle["m_dim"]),), dtype=np.float32),
            "raw_not_ready_warned": False,
        }

    def build_idle_tor_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append([0.0, 0.0, float(self.defaults_hand["tor"][i]), 0.0, 0.0])
        return matrix

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
        if self.glove_min_interval_sec > 0.0:
            if (now - self.last_glove_accept_ts) < self.glove_min_interval_sec:
                self.glove_drop_rate_limited += 1
                return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        if len(data) == 4 and len(self.finger_models) == 1:
            # Expand one selected finger's four angles into the hand-local 20D layout.
            bundle = next(iter(self.finger_models.values()))
            q_indices = bundle["q_indices"]
            if len(q_indices) != len(data):
                return
            glove20 = np.zeros((20,), dtype=np.float32)
            for value, q_index in zip(data, q_indices):
                glove20[int(q_index)] = float(value)
        elif len(data) == 20:
            # Hand-local joint-angle input used by the manual joint-angle launch.
            glove20 = np.asarray(data, dtype=np.float32)
        elif len(data) >= 48:
            # Legacy dual-hand glove packet: roots[0:8], left[8:28], right[28:48].
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

    def _build_history_m_from_feedback(self, motor_ids):
        history_m = []
        for mid in motor_ids:
            idx = int(mid) - 1
            pos = float(self.latest_feedback_matrix[idx, 0])
            if np.isnan(pos):
                pos = float(self.last_cmd_matrix[idx][0])
            history_m.append(pos)
        return np.asarray(history_m, dtype=np.float32)

    def _extract_raw_skeleton_chunk(self, data):
        if len(data) == RAW_SKELETON_ONE_HAND_DIM:
            return data
        if len(data) < RAW_SKELETON_TWO_HAND_DIM:
            return None
        chunks = [
            data[0:RAW_SKELETON_ONE_HAND_DIM],
            data[RAW_SKELETON_ONE_HAND_DIM:RAW_SKELETON_TWO_HAND_DIM],
        ]
        target_code = 1.0 if self.hand == "right" else 0.0
        for chunk in chunks:
            if abs(float(chunk[0]) - target_code) < 0.5:
                return chunk
        return chunks[0]

    def _glove_raw_cb(self, msg):
        self.glove_raw_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_raw_accept_ts) < self.glove_min_interval_sec:
            self.glove_raw_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        chunk = self._extract_raw_skeleton_chunk(data)
        if chunk is None:
            return

        try:
            q16_map = compute_non_thumb_q16(chunk)
            q16_clean = {}
            for finger_name in RIGHT_NON_THUMB_FINGERS:
                for i in range(4):
                    key = f"{finger_name}_q_{i}"
                    if key not in q16_map:
                        raise KeyError(f"Missing key from compute_non_thumb_q16: {key}")
                    q16_clean[key] = float(q16_map[key])
        except Exception as exc:
            self.get_logger().warn(
                f"[prev_m_target_q_mlp_mapping] failed to compute right non-thumb q from raw skeleton: {exc}"
            )
            return

        self.latest_right_non_thumb_q16 = q16_clean
        self.last_glove_raw_ts = now
        self.last_glove_raw_accept_ts = now
        self.glove_raw_accept_total += 1

    def _try_get_non_thumb_raw_q(self, finger_name: str, q_dim: int):
        q16_map = self.latest_right_non_thumb_q16
        if q16_map is None:
            return None
        vals = []
        for i in range(int(q_dim)):
            key = f"{finger_name}_q_{i}"
            if key not in q16_map:
                return None
            vals.append(float(q16_map[key]))
        return np.asarray(vals, dtype=np.float32)

    def _build_target_q_from_glove20(self, glove20: np.ndarray, q_indices):
        q_now = np.asarray([float(glove20[i]) for i in q_indices], dtype=np.float32)
        for i, qi in enumerate(q_indices):
            off = float(self.q_offsets_this_hand.get(int(qi), 0.0))
            if abs(off) > 1e-9:
                q_now[i] = float(q_now[i] + off)
        return q_now

    def _select_target_q_raw(self, finger_name: str, bundle: dict, glove20: np.ndarray):
        if self.use_right_non_thumb_raw_skeleton_q and self.hand == "right" and finger_name in RIGHT_NON_THUMB_FINGERS:
            return self._try_get_non_thumb_raw_q(finger_name, bundle["q_dim"])
        return np.asarray([float(glove20[i]) for i in bundle["q_indices"]], dtype=np.float32)

    def _build_model_input(self, bundle: dict, history_m: np.ndarray, target_q: np.ndarray):
        values = []
        for kind, idx, _col in bundle["input_plan"]:
            if kind == "history_m":
                values.append(float(history_m[int(idx)]))
            elif kind == "target_q":
                values.append(float(target_q[int(idx)]))
            else:
                raise RuntimeError(f"Unsupported input plan kind: {kind!r}")
        return np.asarray(values, dtype=np.float32).reshape(1, -1)

    def _hold_last_cmd_for_finger(self, matrix, motor_ids):
        for mid in motor_ids:
            idx = int(mid) - 1
            prev = self.last_cmd_matrix[idx]
            matrix[idx] = [
                float(prev[0]),
                float(prev[1]),
                float(prev[2]),
                float(prev[3]),
                float(prev[4]),
            ]

    def _build_cmd_from_glove20(self, glove20: np.ndarray):
        matrix = self.build_idle_tor_matrix()
        clip_last = 0
        infer_t0 = time.monotonic()

        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]

            history_m = self._build_history_m_from_feedback(b["motor_ids"])
            target_q_raw = self._select_target_q_raw(name, b, glove20)
            if target_q_raw is None:
                if not b["raw_not_ready_warned"]:
                    self.get_logger().warn(
                        "[prev_m_target_q_mlp_mapping] right non-thumb raw skeleton q is not ready yet "
                        f"for finger={name}; skip inference for this finger this frame and hold previous command."
                    )
                    b["raw_not_ready_warned"] = True
                self._hold_last_cmd_for_finger(matrix, b["motor_ids"])
                continue
            b["raw_not_ready_warned"] = False

            target_q = self._build_target_q_from_glove20(glove20, b["q_indices"])
            if self.use_right_non_thumb_raw_skeleton_q and self.hand == "right" and name in RIGHT_NON_THUMB_FINGERS:
                target_q = target_q_raw.copy()
                for i, qi in enumerate(b["q_indices"]):
                    off = float(self.q_offsets_this_hand.get(int(qi), 0.0))
                    if abs(off) > 1e-9:
                        target_q[i] = float(target_q[i] + off)
            x = self._build_model_input(bundle=b, history_m=history_m, target_q=target_q)
            x_norm = b["x_scaler"].transform(x)

            with torch.no_grad():
                xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                pred_m_norm = b["model"](xb).cpu().numpy()
            pred_m = b["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)

            cmd_pos_by_model_motor_idx = np.zeros((b["m_dim"],), dtype=np.float32)
            for out_col_idx, (target_idx, _target_col) in enumerate(b["target_plan"]):
                motor_id = int(b["motor_ids"][int(target_idx)])
                idx = motor_id - 1
                pred_pos = float(pred_m[out_col_idx])
                pred_pos += float(self.motor_cmd_offsets_this_hand.get(motor_id, 0.0))
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
                cmd_pos_by_model_motor_idx[int(target_idx)] = float(pred_pos)

            b["last_history_m"] = history_m.astype(np.float32)
            b["last_target_q"] = target_q.astype(np.float32)
            b["last_pred_m"] = pred_m.astype(np.float32)
            b["last_cmd_pos"] = cmd_pos_by_model_motor_idx.astype(np.float32)

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

            current_pos = float(self.latest_feedback_matrix[idx, 0])
            if not np.isfinite(current_pos):
                continue

            position_error = float(row[0]) - current_pos
            abs_error = abs(position_error)
            if bool(self.tor_idle_latched[idx]):
                if abs_error > TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG:
                    self.tor_idle_latched[idx] = False
                else:
                    row[:] = [
                        0.0,
                        0.0,
                        float(self.defaults_hand["tor"][idx]),
                        0.0,
                        0.0,
                    ]
                    continue

            if abs_error <= TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG:
                self.tor_idle_latched[idx] = True
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
            matrix = self.build_idle_tor_matrix()
        else:
            if len(self.glove_pending) > 0:
                glove20_last = self.glove_pending[-1]
                self.glove_pending.clear()
                matrix = self._build_cmd_from_glove20(glove20_last)
            elif self.latest_glove20 is not None:
                matrix = self.last_cmd_matrix
            else:
                matrix = self.build_idle_tor_matrix()
        matrix = self._apply_directional_tor_active(matrix)
        self.last_cmd_matrix = matrix
        self._publish_matrix(matrix)

    def _status_cb(self):
        now = time.monotonic()
        glove_age = (now - self.last_glove_ts) if self.last_glove_ts > 0.0 else float("inf")
        fb_age = (now - self.last_feedback_ts) if self.last_feedback_ts > 0.0 else float("inf")
        glove_raw_age = (now - self.last_glove_raw_ts) if self.last_glove_raw_ts > 0.0 else float("inf")
        infer_load = (self.infer_ema_ms / (self.control_period_sec * 1000.0)) if self.control_period_sec > 0.0 else float("nan")
        raw_non_thumb_ready = (
            self.latest_right_non_thumb_q16 is not None
            if self.use_right_non_thumb_raw_skeleton_q
            else "disabled"
        )

        finger_status = []
        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]
            cmd_pos = [round(float(v), 3) for v in b["last_cmd_pos"].tolist()]
            history_m = [round(float(v), 3) for v in b["last_history_m"].tolist()]
            target_q = [round(float(v), 3) for v in b["last_target_q"].tolist()]
            pred_m = [round(float(v), 3) for v in b["last_pred_m"].tolist()]
            cmd_tor = [
                round(float(self.last_cmd_matrix[int(mid) - 1][2]), 3)
                for mid in b["motor_ids"]
            ]
            tension_ids = [
                int(mid)
                for mid in b["motor_ids"]
                if abs(float(self.last_cmd_matrix[int(mid) - 1][3])) <= 1e-9
                and abs(float(self.last_cmd_matrix[int(mid) - 1][4])) <= 1e-9
            ]
            finger_status.append(
                f"{name}:motors={b['motor_ids']},history_m={history_m},cmd={cmd_pos},target_q={target_q},"
                f"pred_m={pred_m},cmd_tor={cmd_tor},pure_tension_ids={tension_ids}"
            )

        self.get_logger().info(
            "[prev_m_target_q_mlp_mapping] "
            f"feedback_ready={self.feedback_received}, glove_ready={self.latest_glove20 is not None}, "
            f"fb_age={fb_age:.3f}s, glove_age={glove_age:.3f}s, "
            f"glove_rx={self.glove_rx_total}, glove_accept={self.glove_accept_total}, "
            f"glove_drop_rl={self.glove_drop_rate_limited}, "
            f"glove_raw_age={glove_raw_age:.3f}s, glove_raw_rx={self.glove_raw_rx_total}, "
            f"glove_raw_accept={self.glove_raw_accept_total}, glove_raw_drop_rl={self.glove_raw_drop_rate_limited}, "
            f"right_non_thumb_q_raw_ready={raw_non_thumb_ready}, "
            f"pred_clip_last={self.clip_last}, pred_clip_total={self.clip_total}, "
            f"infer_last_ms={self.infer_last_ms:.3f}, infer_ema_ms={self.infer_ema_ms:.3f}, infer_load={infer_load:.2f}, "
            f"fingers={'; '.join(finger_status)}"
        )

    def _raw_q_status_cb(self):
        if not self.use_right_non_thumb_raw_skeleton_q:
            return
        now = time.monotonic()
        glove_raw_age = (now - self.last_glove_raw_ts) if self.last_glove_raw_ts > 0.0 else float("inf")
        q16_map = self.latest_right_non_thumb_q16
        if q16_map is None:
            q_txt = "not_ready"
        else:
            chunks = []
            for finger_name in RIGHT_NON_THUMB_FINGERS:
                vals = []
                for i in range(4):
                    key = f"{finger_name}_q_{i}"
                    val = q16_map.get(key)
                    vals.append("nan" if val is None else f"{float(val):.3f}")
                chunks.append(f"{finger_name}=[{', '.join(vals)}]")
            q_txt = "; ".join(chunks)
        self.get_logger().info(
            "[prev_m_target_q_mlp_mapping] "
            f"right_non_thumb_q16={q_txt}, glove_raw_age={glove_raw_age:.3f}s"
        )

    def wait_ready(self, timeout_sec: float):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.feedback_received and self.latest_glove20 is not None:
                if self.use_right_non_thumb_raw_skeleton_q and self.latest_right_non_thumb_q16 is None:
                    continue
                return True
        return False


def _add_finger_args(parser: argparse.ArgumentParser):
    for finger_name in DEFAULT_FINGER_ORDER:
        parser.add_argument(
            f"--enable-{finger_name}",
            dest=f"{finger_name}_enable",
            default="",
            help=f"Enable {finger_name} prev-m + target-q model: true/false. Empty uses hand-specific default.",
        )
        parser.add_argument(
            f"--{finger_name}-model-ckpt",
            dest=f"{finger_name}_model_ckpt",
            default="",
            help=f"{finger_name} prev-m + target-q MLP checkpoint path. Empty uses hand-specific default.",
        )
        parser.add_argument(
            f"--{finger_name}-model-motor-ids",
            dest=f"{finger_name}_model_motor_ids",
            default="",
            help=f"{finger_name} output motor ids in checkpoint target_m order.",
        )
        parser.add_argument(
            f"--{finger_name}-q-indices",
            dest=f"{finger_name}_q_indices",
            default="",
            help=f"{finger_name} input q indices inside hand-local glove20.",
        )
        parser.add_argument(
            f"--{finger_name}-mirror-right-from-left",
            dest=f"{finger_name}_mirror_right_from_left",
            default="",
            help=f"Mirror {finger_name} motor ids between hands before control: true/false.",
        )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Five-finger prev-m + target-q MLP mapping (ROS runtime).")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--cmd-topic", default="", help="Override command topic")
    parser.add_argument("--feedback-topic", default="", help="Override feedback topic")
    parser.add_argument("--glove-topic", default=DEFAULT_GLOVE_TOPIC, help="48D glove topic")
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default="/glove_joint_data_raw_skeleton",
        help="176D raw-skeleton topic (Float64MultiArray).",
    )
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
        help="Path to muscle_adjustment_defaults.py",
    )

    # Backward-compatible thumb aliases from the first thumb-only version.
    parser.add_argument("--enable-model", default="", help="Legacy alias for --enable-thumb")
    parser.add_argument("--model-ckpt", default="", help="Legacy alias for --thumb-model-ckpt")
    parser.add_argument("--model-motor-ids", default="", help="Legacy alias for --thumb-model-motor-ids")
    parser.add_argument("--q-indices", default="", help="Legacy alias for --thumb-q-indices")
    parser.add_argument("--mirror-right-from-left", default="", help="Legacy alias for --thumb-mirror-right-from-left")

    _add_finger_args(parser)

    parser.add_argument(
        "--motor-feedback-limits-left-json",
        default="",
        help="Left-hand motor position limits JSON: {\"motor_id\": [lo, hi], ...}",
    )
    parser.add_argument(
        "--motor-feedback-limits-right-json",
        default="",
        help="Right-hand motor position limits JSON: {\"motor_id\": [lo, hi], ...}",
    )
    parser.add_argument(
        "--q-offset-map-json",
        default="",
        help="Glove q offset map JSON by hand and q_index",
    )
    parser.add_argument(
        "--motor-cmd-offset-map-json",
        default="",
        help="Motor command offset map JSON by hand and motor_id",
    )
    parser.add_argument("--control-period-sec", type=float, default=float(CONTROL_PERIOD_SEC))
    parser.add_argument("--status-period-sec", type=float, default=2.0)
    parser.add_argument(
        "--glove-max-hz",
        type=float,
        default=100.0,
        help="Max accepted glove callback rate. <=0 disables rate limiting",
    )
    parser.add_argument("--glove-sub-queue-size", type=int, default=60)
    parser.add_argument("--glove-pending-maxlen", type=int, default=60)
    parser.add_argument(
        "--right-non-thumb-q-from-raw-skeleton",
        default="true",
        help="If enabled, right-hand index/middle/ring/little q_0..q_3 are solved from raw skeleton; thumb still uses glove q_indices.",
    )
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
    finger_specs = _build_finger_specs(args=args, hand=hand, script_dir=script_dir)
    defaults_hand = _load_defaults_from_py(Path(args.defaults_path), hand)
    motor_feedback_limits_left = _parse_motor_feedback_limits_json(
        args.motor_feedback_limits_left_json,
        "--motor-feedback-limits-left-json",
    )
    motor_feedback_limits_right = _parse_motor_feedback_limits_json(
        args.motor_feedback_limits_right_json,
        "--motor-feedback-limits-right-json",
    )
    q_offset_map = _parse_q_offset_map_json(args.q_offset_map_json, "--q-offset-map-json")
    motor_cmd_offset_map = _parse_motor_cmd_offset_map_json(
        args.motor_cmd_offset_map_json,
        "--motor-cmd-offset-map-json",
    )
    use_right_non_thumb_raw = _parse_bool_arg(
        args.right_non_thumb_q_from_raw_skeleton,
        "--right-non-thumb-q-from-raw-skeleton",
    )

    rclpy.init()
    node = None
    try:
        node = PrevMTargetQMLPMappingNode(
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            glove_topic=args.glove_topic.strip(),
            glove_raw_skeleton_topic=args.glove_raw_skeleton_topic.strip(),
            right_non_thumb_q_from_raw_skeleton=bool(use_right_non_thumb_raw),
            defaults_hand=defaults_hand,
            finger_specs=finger_specs,
            device=args.device.strip().lower(),
            motor_feedback_limits_left=motor_feedback_limits_left,
            motor_feedback_limits_right=motor_feedback_limits_right,
            q_offset_map=q_offset_map,
            motor_cmd_offset_map=motor_cmd_offset_map,
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
                    f"feedback_ready={node.feedback_received}, glove_ready={node.latest_glove20 is not None}"
                )

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[prev_m_target_q_mlp_mapping] KeyboardInterrupt received, exiting...")
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
