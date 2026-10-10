#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from datetime import datetime
import importlib.util
import json
import os
import select
import sys
import termios
import time
import tty
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
    RIGHT_INDEX_Q_IMPORT_ERROR = exc
else:
    RIGHT_INDEX_Q_IMPORT_ERROR = None


from hand_motor_map import (
    FINGER_MOTOR_IDS_BY_HAND, HAND_MOTOR_MAP, LEFT_TO_RIGHT_MOTOR_MAP, RIGHT_TO_LEFT_MOTOR_MAP,
)

from hand_motor_map import (
    model_motor_ids_csv,
)

LEFT_HAND_GLOVE_SLICE = slice(8, 28)
RIGHT_HAND_GLOVE_SLICE = slice(28, 48)
MOTOR_COUNT = 28
RAW_SKELETON_ONE_HAND_DIM = 176
RAW_SKELETON_TWO_HAND_DIM = 352
RAW_NON_THUMB_Q_LOG_PERIOD_SEC = 1.0
TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG = 3.0
TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG = 5.0
SIDE_SWING_TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG = 5.0
SIDE_SWING_TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG = 7.0


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

RIGHT_SIDE_SWING_MOTOR_IDS = {mid for finger, muscles in HAND_MOTOR_MAP['right'].items()
    if finger != "thumb" for role, mid in muscles.items() if role in ("abduction", "adduction")} 
LEFT_SIDE_SWING_MOTOR_IDS = {mid for finger, muscles in HAND_MOTOR_MAP['left'].items()
    if finger != "thumb" for role, mid in muscles.items() if role in ("abduction", "adduction")} 
SIDE_SWING_MOTOR_IDS = RIGHT_SIDE_SWING_MOTOR_IDS | LEFT_SIDE_SWING_MOTOR_IDS


def _is_side_swing_motor_id(motor_id: int, hand: str) -> bool:
    motors = LEFT_SIDE_SWING_MOTOR_IDS if hand == "left" else RIGHT_SIDE_SWING_MOTOR_IDS
    return int(motor_id) in motors

MOTOR_FEEDBACK_LIMITS_LEFT = {6: (-50.0, 500.0),
 4: (-600.0, 50.0),
 27: (-30.0, 50.0),
 7: (-50.0, 500.0),
 8: (-30.0, 50.0),
 14: (-600.0, 50.0),
 3: (-600.0, 50.0),
 25: (-50.0, 500.0),
 28: (-30.0, 50.0),
 1: (-30.0, 50.0),
 15: (-600.0, 50.0),
 16: (-600.0, 50.0),
 23: (-50.0, 500.0),
 2: (-600.0, 50.0),
 24: (-30.0, 50.0),
 9: (-30.0, 50.0),
 17: (-600.0, 50.0),
 21: (-30.0, 50.0),
 10: (-50.0, 500.0),
 22: (-30.0, 50.0),
 13: (-600.0, 50.0),
 11: (-50.0, 500.0),
 20: (-30.0, 50.0),
 18: (-30.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_RIGHT = {20: (-50.0, 500.0),
 18: (-600.0, 50.0),
 13: (-30.0, 50.0),
 21: (-50.0, 500.0),
 22: (-30.0, 50.0),
 28: (-600.0, 50.0),
 17: (-600.0, 50.0),
 11: (-50.0, 500.0),
 14: (-30.0, 50.0),
 15: (-30.0, 50.0),
 1: (-600.0, 50.0),
 2: (-500.0, 50.0),
 9: (-50.0, 500.0),
 16: (-600.0, 50.0),
 10: (-30.0, 50.0),
 23: (-30.0, 50.0),
 3: (-600.0, 50.0),
 7: (-30.0, 50.0),
 24: (-50.0, 500.0),
 8: (-30.0, 50.0),
 27: (-600.0, 50.0),
 25: (-50.0, 500.0),
 6: (-30.0, 50.0),
 4: (-30.0, 50.0)}

MOTOR_FEEDBACK_RECORD_IDS_BY_HAND = dict(FINGER_MOTOR_IDS_BY_HAND)

DEFAULT_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
RIGHT_NON_THUMB_FINGERS = ("index", "middle", "ring", "little")

# Consolidated per-finger defaults for CLI/runtime spec building.
# Hand-specific defaults let this script run correctly for left/right without
# requiring launch-time overrides.
FINGER_DEFAULTS_BY_HAND = {
    "left": {
        "thumb": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "thumb_1", "best_thumb_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'thumb'),
            "q_indices_csv": "0,1,2,3",
            "mirror_right_from_left_default": "false",
        },
        "index": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "index_delta_multihead", "best_index_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'index'),
            "q_indices_csv": "4,5,6,7",
            "mirror_right_from_left_default": "false",
        },
        "middle": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "middle_delta_multihead", "best_middle_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'middle'),
            "q_indices_csv": "8,9,10,11",
            "mirror_right_from_left_default": "false",
        },
        "ring": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "ring_delta_multihead", "best_ring_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'ring'),
            "q_indices_csv": "12,13,14,15",
            "mirror_right_from_left_default": "false",
        },
        "little": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "little_delta_multihead", "best_little_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'little'),
            "q_indices_csv": "16,17,18,19",
            "mirror_right_from_left_default": "false",
        },
    },
    "right": {
        "thumb": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "thumb_1", "best_thumb_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'thumb'),
            "q_indices_csv": "0,1,2,3",
            "mirror_right_from_left_default": "true",
        },
        "index": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp_right", "index_delta_multihead_masked", "best_index_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('right', 'index'),
            "q_indices_csv": "4,5,6,7",
            "mirror_right_from_left_default": "false",
        },
        "middle": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp_right", "middle_delta_multihead", "best_middle_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('right', 'middle'),
            "q_indices_csv": "8,9,10,11",
            "mirror_right_from_left_default": "false",
        },
        "ring": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp", "runs", "ring_delta_multihead", "best_ring_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('left', 'ring'),
            "q_indices_csv": "12,13,14,15",
            "mirror_right_from_left_default": "true",
        },
        "little": {
            "enable_default": "true",
            "model_ckpt_rel_parts": ("mp_right", "little_delta_multihead", "best_little_delta_multihead.pt"),
            "motor_ids_csv": model_motor_ids_csv('right', 'little'),
            "q_indices_csv": "16,17,18,19",
            "mirror_right_from_left_default": "false",
        },
    },
}

Q_OFFSET_MAP_DEFAULTS = {
    "left": {
        1: 0.0,
        4: 0.0,
        8: 0.0,
        11: 0.0,
        12: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
    },
    "right": {
        1: 0.0,
        4: 0.0,
        8: 0.0,
        11: 0.0,
        12: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
    },
}

MOTOR_CMD_OFFSET_MAP_DEFAULTS = {'left': {1: 0.0,
          2: 0.0,
          15: 0.0,
          27: 0.0,
          7: 0.0,
          8: 0.0,
          24: 0.0,
          9: 0.0,
          11: 0.0,
          20: 0.0,
          18: 0.0,
          14: 0.0,
          3: 0.0,
          16: 0.0,
          17: 0.0,
          13: 0.0,
          21: 0.0,
          10: 0.0,
          22: 0.0,
          23: 0.0,
          25: 0.0,
          6: 0.0,
          28: 0.0,
          4: 0.0},
 'right': {17: 0.0,
           2: 0.0,
           3: 0.0,
           27: 0.0,
           7: 0.0,
           24: 0.0,
           8: 0.0,
           9: 0.0,
           11: 0.0,
           20: 0.0,
           14: 0.0,
           18: 0.0,
           15: 0.0,
           16: 0.0,
           1: 0.0,
           13: 0.0,
           21: 0.0,
           22: 0.0,
           10: 0.0,
           23: 0.0,
           25: 0.0,
           6: 0.0,
           4: 0.0,
           28: 0.0}}


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
    class DeltaMultiHeadNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            head_hidden_dim: int = 64,
            dropout: float = 0.0,
        ):
            super().__init__()
            h = int(hidden_dim)
            self.backbone = nn.Sequential(
                nn.Linear(int(input_dim), h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(h, int(head_hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(head_hidden_dim), 1),
                    )
                    for _ in range(int(output_dim))
                ]
            )

        def forward(self, x):
            feat = self.backbone(x)
            outs = [head(feat) for head in self.heads]
            return torch.cat(outs, dim=1)


    class MaskedDeltaMultiHeadNet(nn.Module):
        """
        Per-output-head masked model.
        Each head consumes a subset of [q_t, q_t_minus_1, m_t_minus_1, dq_t]
        according to q_input_map/include_all_m_prev saved in checkpoint.
        """

        def __init__(
            self,
            head_input_dims,
            hidden_dim: int = 128,
            head_hidden_dim: int = 64,
            dropout: float = 0.0,
        ):
            super().__init__()
            h = int(hidden_dim)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(in_dim), h),
                        nn.LayerNorm(h),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(h, h),
                        nn.LayerNorm(h),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(h, int(head_hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(head_hidden_dim), 1),
                    )
                    for in_dim in head_input_dims
                ]
            )
            self.register_buffer(
                "_head_input_index_tensor",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.head_input_indices = []

        def set_head_input_indices(self, head_input_indices):
            self.head_input_indices = [
                torch.tensor(idx, dtype=torch.long)
                for idx in head_input_indices
            ]

        def forward(self, x):
            outs = []
            for head, idx in zip(self.heads, self.head_input_indices):
                idx_dev = idx.to(device=x.device)
                x_h = torch.index_select(x, dim=1, index=idx_dev)
                outs.append(head(x_h))
            return torch.cat(outs, dim=1)
else:
    class DeltaMultiHeadNet:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Cannot build model because torch/torch.nn is unavailable. "
                f"Import error: {TORCH_IMPORT_ERROR}"
            )

    class MaskedDeltaMultiHeadNet:  # pragma: no cover
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
            raise ValueError(
                f"{arg_name} has non-integer motor id key: {motor_id_raw!r}"
            ) from exc

        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise ValueError(
                f"{arg_name} motor {motor_id} must map to [lo, hi], got: {bound!r}"
            )

        lo = float(bound[0])
        hi = float(bound[1])
        if lo > hi:
            raise ValueError(f"{arg_name} motor {motor_id} has invalid range: lo={lo} > hi={hi}")
        limits[motor_id] = (lo, hi)

    return limits


def _parse_q_offset_map_json(raw_json: str, arg_name: str):
    text = str(raw_json).strip()
    if not text:
        return {"left": {}, "right": {}}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{arg_name} must be JSON object: {{\"left\": {{\"4\": 0.0}}, \"right\": {{\"8\": 10.0}}}}"
        )

    out = {"left": {}, "right": {}}
    for hand_raw, q_map in parsed.items():
        hand = str(hand_raw).strip().lower()
        if hand not in ("left", "right"):
            raise ValueError(f"{arg_name} has unsupported hand key: {hand_raw!r}, expected left/right")
        if not isinstance(q_map, dict):
            raise ValueError(f"{arg_name}[{hand}] must be object mapping q_index->offset")
        hm = {}
        for q_idx_raw, off_raw in q_map.items():
            try:
                q_idx = int(q_idx_raw)
            except Exception as exc:
                raise ValueError(f"{arg_name}[{hand}] has invalid q index key: {q_idx_raw!r}") from exc
            if q_idx < 0 or q_idx >= 20:
                raise ValueError(f"{arg_name}[{hand}] q index out of range: {q_idx}, expected [0,19]")
            hm[q_idx] = float(off_raw)
        out[hand] = hm
    return out


def _parse_motor_cmd_offset_map_json(raw_json: str, arg_name: str):
    text = str(raw_json).strip()
    if not text:
        return {"left": {}, "right": {}}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{arg_name} must be JSON object: {{\"left\": {{\"8\": 5.0}}, \"right\": {{\"22\": 5.0}}}}"
        )

    out = {"left": {}, "right": {}}
    for hand_raw, motor_map in parsed.items():
        hand = str(hand_raw).strip().lower()
        if hand not in ("left", "right"):
            raise ValueError(f"{arg_name} has unsupported hand key: {hand_raw!r}, expected left/right")
        if not isinstance(motor_map, dict):
            raise ValueError(f"{arg_name}[{hand}] must be object mapping motor_id->offset")
        hm = {}
        for motor_id_raw, off_raw in motor_map.items():
            try:
                motor_id = int(motor_id_raw)
            except Exception as exc:
                raise ValueError(f"{arg_name}[{hand}] has invalid motor_id key: {motor_id_raw!r}") from exc
            if motor_id < 1 or motor_id > 28:
                raise ValueError(f"{arg_name}[{hand}] motor_id out of range: {motor_id}, expected [1,28]")
            hm[motor_id] = float(off_raw)
        out[hand] = hm
    return out


def _load_defaults_from_py(defaults_path: Path, hand: str):
    spec = importlib.util.spec_from_file_location("five_finger_delta_defaults_dynamic", str(defaults_path))
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


def _load_delta_multihead_model(ckpt_path: Path, device: str):
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
    hidden_dim = int(cfg.get("hidden_dim", 128))
    head_hidden_dim = int(cfg.get("head_hidden_dim", 64))
    dropout = float(cfg.get("dropout", 0.0))

    q_cols = list(ckpt.get("q_cols", [f"q_{i}" for i in range(q_dim)]))
    m_cols = list(ckpt.get("m_cols", [f"m_{i}" for i in range(m_dim)]))
    model_type = str(ckpt.get("model_type", cfg.get("model_type", "dense"))).strip().lower()
    if model_type in ("dense", "default", "base"):
        model_type = "dense"
    elif model_type == "masked":
        model_type = "masked"
    else:
        raise RuntimeError(
            f"Unsupported model_type={model_type!r} in checkpoint {ckpt_path}. "
            "Supported: dense, masked."
        )

    head_input_indices = None
    if model_type == "masked":
        input_layout = list(ckpt.get("input_layout", []))
        q_input_map = ckpt.get("q_input_map", {})
        include_all_m_prev = bool(ckpt.get("include_all_m_prev", True))
        if not isinstance(q_input_map, dict):
            raise RuntimeError(f"Invalid q_input_map in checkpoint: {type(q_input_map)}")

        def _q_name_to_index(q_name: str) -> int:
            if isinstance(q_name, str) and q_name.startswith("q_"):
                return int(q_name.split("_", 1)[1])
            raise RuntimeError(f"Invalid q name in q_input_map: {q_name!r}")

        # Build global feature index ranges by input_layout.
        cursor = 0
        q_t_base = None
        q_prev_base = None
        m_prev_base = None
        dq_base = None
        for token in input_layout:
            tok = str(token).strip()
            if tok == "q_t":
                q_t_base = cursor
                cursor += q_dim
            elif tok == "q_t_minus_1":
                q_prev_base = cursor
                cursor += q_dim
            elif tok == "m_t_minus_1":
                m_prev_base = cursor
                cursor += m_dim
            elif tok == "dq_t":
                dq_base = cursor
                cursor += q_dim
            else:
                raise RuntimeError(f"Unsupported input_layout token in masked model: {tok!r}")

        head_input_indices = []
        for out_i in range(m_dim):
            m_name = m_cols[out_i] if out_i < len(m_cols) else f"m_{out_i}"
            q_name_list = q_input_map.get(m_name, q_cols)
            q_idx_list = [_q_name_to_index(qn) for qn in q_name_list]

            idx = []
            if q_t_base is not None:
                idx.extend([q_t_base + qi for qi in q_idx_list])
            if q_prev_base is not None:
                idx.extend([q_prev_base + qi for qi in q_idx_list])
            if m_prev_base is not None:
                if include_all_m_prev:
                    idx.extend([m_prev_base + mi for mi in range(m_dim)])
                else:
                    idx.append(m_prev_base + out_i)
            if dq_base is not None:
                idx.extend([dq_base + qi for qi in q_idx_list])

            if not idx:
                raise RuntimeError(
                    f"Empty masked head input for {m_name} in checkpoint {ckpt_path}"
                )
            head_input_indices.append(idx)

        model = MaskedDeltaMultiHeadNet(
            head_input_dims=[len(i) for i in head_input_indices],
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
        ).to(device)
        model.set_head_input_indices(head_input_indices)
    else:
        model = DeltaMultiHeadNet(
            input_dim=q_dim * 3 + m_dim,
            output_dim=m_dim,
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
        ).to(device)
    state_dict = ckpt["model_state"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint model_state must be dict, got {type(state_dict)}")

    # Backward compatibility: some older masked checkpoints store auxiliary
    # keys (head_feature_weights_*) that are not nn.Module parameters.
    legacy_aux_prefix = "head_feature_weights_"
    legacy_aux_keys = [k for k in state_dict.keys() if str(k).startswith(legacy_aux_prefix)]
    if legacy_aux_keys:
        state_dict = {k: v for k, v in state_dict.items() if not str(k).startswith(legacy_aux_prefix)}
        print(
            "[five_finger_delta_mapping] warning: ignoring legacy auxiliary "
            f"state keys in checkpoint {ckpt_path.name}: {legacy_aux_keys}"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Incompatible checkpoint for current model definition: {ckpt_path}. "
            "Please retrain/export with matching code, or switch to a compatible checkpoint."
        ) from exc
    model.eval()

    return {
        "model": model,
        "x_scaler": Standardizer.from_dict(ckpt["x_scaler"]),
        "y_scaler": Standardizer.from_dict(ckpt["y_scaler"]),
        "q_dim": q_dim,
        "m_dim": m_dim,
        "q_cols": q_cols,
        "m_cols": m_cols,
        "input_layout": ckpt.get("input_layout", []),
        "target_layout": ckpt.get("target_layout", "delta_m_t"),
        "model_type": model_type,
        "head_input_indices": head_input_indices,
        "hidden_dim": hidden_dim,
        "head_hidden_dim": head_hidden_dim,
        "dropout": dropout,
    }


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
        raise RuntimeError(f"Unsupported hand for MOTOR_FEEDBACK_LIMITS: {hand!r}")

    limits = {}
    for mid, bound in raw_limits.items():
        lo = float(bound[0])
        hi = float(bound[1])
        if lo > hi:
            raise RuntimeError(f"Invalid limit for motor {mid}: lo={lo} > hi={hi}")
        limits[int(mid)] = (lo, hi)

    return limits, mode


def _add_finger_cli_args(parser: argparse.ArgumentParser, finger_name: str):
    # Use None defaults first, then resolve by --hand in parse_args().
    parser.add_argument(f"--enable-{finger_name}", default=None)
    parser.add_argument(
        f"--{finger_name}-model-ckpt",
        default=None,
    )
    parser.add_argument(f"--{finger_name}-model-motor-ids", default=None)
    parser.add_argument(f"--{finger_name}-q-indices", default=None)
    parser.add_argument(
        f"--{finger_name}-mirror-right-from-left",
        default=None,
    )


class FiveFingerDeltaMappingNode(Node):
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
        record_motor_feedback_csv: bool,
        motor_feedback_csv_path: str,
        motor_cmd_offset_map,
        q_offset_map,
        motor_feedback_limits_left,
        motor_feedback_limits_right,
    ):
        super().__init__(f"five_finger_delta_mapping_{hand}")
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
        # Optional path: for right-hand non-thumb fingers (index/middle/ring/little),
        # derive q_0..q_3 from /glove_joint_data_raw_skeleton (25 joints per hand),
        # not glove48 q.
        self.use_right_non_thumb_raw_skeleton_q = bool(right_non_thumb_q_from_raw_skeleton and hand == "right")
        if self.use_right_non_thumb_raw_skeleton_q and (not callable(compute_non_thumb_q16)):
            raise RuntimeError(
                "Cannot import replace_right_non_thumb_q_from_skeleton.py while "
                "--right-non-thumb-q-from-raw-skeleton is enabled.\n"
                f"Import error: {RIGHT_INDEX_Q_IMPORT_ERROR}"
            )
        self.defaults_hand = defaults_hand
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.hand_glove_slice = LEFT_HAND_GLOVE_SLICE if hand == "left" else RIGHT_HAND_GLOVE_SLICE
        self.control_period_sec = float(control_period_sec)
        self.glove_max_hz = float(glove_max_hz)
        self.glove_min_interval_sec = (1.0 / self.glove_max_hz) if self.glove_max_hz > 0.0 else 0.0
        self.glove_sub_queue_size = max(1, int(glove_sub_queue_size))
        self.glove_pending_maxlen = max(1, int(glove_pending_maxlen))
        self.record_motor_feedback_csv = bool(record_motor_feedback_csv)
        self.motor_feedback_csv_path_template = str(motor_feedback_csv_path)
        self.motor_feedback_csv_path = ""
        self.motor_feedback_record_ids = list(MOTOR_FEEDBACK_RECORD_IDS_BY_HAND[self.hand])
        self.feedback_frame_idx = 0
        self._motor_feedback_fp = None
        self._motor_feedback_writer = None
        self._motor_feedback_has_opened = False
        self._keyboard_fd = None
        self._keyboard_termios_original = None
        self.keyboard_record_control_enabled = False

        motor_cmd_offset_map = motor_cmd_offset_map or {}
        self.motor_cmd_offset_map = {
            "left": {int(k): float(v) for k, v in dict(motor_cmd_offset_map.get("left", {})).items()},
            "right": {int(k): float(v) for k, v in dict(motor_cmd_offset_map.get("right", {})).items()},
        }
        self.motor_cmd_offsets_this_hand = dict(self.motor_cmd_offset_map.get(self.hand, {}))
        q_offset_map = q_offset_map or {}
        self.q_offset_map = {
            "left": {int(k): float(v) for k, v in dict(q_offset_map.get("left", {})).items()},
            "right": {int(k): float(v) for k, v in dict(q_offset_map.get("right", {})).items()},
        }
        self.q_offsets_this_hand = dict(self.q_offset_map.get(self.hand, {}))

        self.motor_pos_limits, self.motor_pos_limit_mode = _build_motor_pos_limit_map(
            hand=self.hand,
            limits_left=motor_feedback_limits_left,
            limits_right=motor_feedback_limits_right,
        )

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
        # Keep larger DDS queue; inference consumes newest frame only.
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
            self.raw_q_status_timer = self.create_timer(
                float(RAW_NON_THUMB_Q_LOG_PERIOD_SEC),
                self._raw_q_status_cb,
            )
        self.keyboard_timer = self.create_timer(0.05, self._keyboard_control_cb)

        self.latest_glove20 = None
        self.last_glove_ts = 0.0
        self.last_glove_accept_ts = 0.0
        self.latest_right_non_thumb_q16 = None
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
        self._raw_non_thumb_fallback_warned = set()
        self.glove_pending = deque(maxlen=self.glove_pending_maxlen)

        self.last_cmd_matrix = self.build_idle_tor_matrix()
        self.clip_total = 0
        self.clip_last = 0
        self.infer_last_ms = 0.0
        self.infer_ema_ms = 0.0
        self._init_keyboard_record_control()
        if self.record_motor_feedback_csv:
            self.start_motor_feedback_recording(reason="startup_flag")

        self.get_logger().info(
            "[five_finger_delta_mapping] started "
            f"hand={self.hand}, cmd={self.cmd_topic}, feedback={self.feedback_topic}, glove={self.glove_topic}, "
            f"glove_max_hz={self.glove_max_hz}, "
            f"glove_sub_queue={self.glove_sub_queue_size}, glove_pending_maxlen={self.glove_pending_maxlen}, "
            f"right_non_thumb_raw_skeleton={self.use_right_non_thumb_raw_skeleton_q}, "
            f"raw_topic={self.glove_raw_skeleton_topic if self.use_right_non_thumb_raw_skeleton_q else 'disabled'}, "
            f"raw_q_log_period_sec={RAW_NON_THUMB_Q_LOG_PERIOD_SEC if self.use_right_non_thumb_raw_skeleton_q else 'disabled'}, "
            f"consume_latest_only=true, "
            f"record_motor_feedback_csv_initial={self.record_motor_feedback_csv}, "
            f"motor_feedback_csv_path_template={self.motor_feedback_csv_path_template}, "
            f"keyboard_record_control={'enabled' if self.keyboard_record_control_enabled else 'disabled'}"
        )
        if self.keyboard_record_control_enabled:
            self.get_logger().info(
                "[five_finger_delta_mapping] keyboard control: press 'r' to start motor feedback CSV recording, "
                "press 'p' to pause/stop recording."
            )
        self.get_logger().info(
            "[five_finger_delta_mapping] "
            f"motor_pos_limit_mode={self.motor_pos_limit_mode}, "
            f"motor_pos_limit_ids={sorted(self.motor_pos_limits.keys())}, "
            f"motor_cmd_offsets_{self.hand}={self.motor_cmd_offsets_this_hand}, "
            f"q_offsets_{self.hand}={self.q_offsets_this_hand}"
        )
        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]
            self.get_logger().info(
                "[five_finger_delta_mapping] "
                f"finger={name}, ckpt={b['ckpt']}, q_indices={b['q_indices']}, motor_ids={b['motor_ids']}, "
                f"q_dim={b['q_dim']}, m_dim={b['m_dim']}, input_layout={b['input_layout']}, target_layout={b['target_layout']}"
            )

    def _load_finger_model(self, name: str, spec: dict):
        motor_ids = list(spec["motor_ids"])
        if spec["mirror_right_from_left"]:
            if self.hand == "right":
                motor_ids = _mirror_motors_left_to_right(motor_ids)
            else:
                motor_ids = _mirror_motors_right_to_left(motor_ids)

        model_bundle = _load_delta_multihead_model(Path(spec["model_ckpt"]), self.device)
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
            "q_cols": list(model_bundle["q_cols"]),
            "m_cols": list(model_bundle["m_cols"]),
            "input_layout": model_bundle["input_layout"],
            "target_layout": model_bundle["target_layout"],
            "q_indices": q_indices,
            "motor_ids": motor_ids,
            "prev_q": None,
            "last_pred_delta": np.zeros((int(model_bundle["m_dim"]),), dtype=np.float32),
            "last_cmd_pos": np.zeros((int(model_bundle["m_dim"]),), dtype=np.float32),
        }

    def _resolve_motor_feedback_csv_path(self, raw_path: str) -> str:
        raw = str(raw_path).strip()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_name = f"{self.hand}_motor_feedback_{ts}_pid{os.getpid()}.csv"
        if not raw or raw.lower() == "auto":
            auto_dir = Path(__file__).resolve().parent / "motor_test_logs" / "motor_feedback_frames"
            return str((auto_dir / default_name).resolve())

        rendered = (
            raw.replace("{hand}", self.hand)
            .replace("{pid}", str(os.getpid()))
            .replace("{timestamp}", ts)
        )
        p = Path(rendered).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if str(rendered).endswith("/") or p.suffix == "":
            p = p / default_name
        return str(p)

    def _open_motor_feedback_csv(self):
        if self._motor_feedback_writer is not None:
            return
        if not self.motor_feedback_csv_path:
            self.motor_feedback_csv_path = self._resolve_motor_feedback_csv_path(
                self.motor_feedback_csv_path_template
            )
        p = Path(self.motor_feedback_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._motor_feedback_has_opened else "w"
        self._motor_feedback_fp = open(str(p), mode, newline="", encoding="utf-8", buffering=1)
        self._motor_feedback_writer = csv.writer(self._motor_feedback_fp)
        if not self._motor_feedback_has_opened:
            header = ["frame_idx", "wall_time_sec", "mono_time_sec"]
            header.extend([f"motor_{int(mid)}_pos" for mid in self.motor_feedback_record_ids])
            self._motor_feedback_writer.writerow(header)
            self._motor_feedback_has_opened = True

    def start_motor_feedback_recording(self, reason: str = "keyboard"):
        if self._motor_feedback_writer is not None:
            self.get_logger().info(
                f"[motor-feedback-record] already recording: path={self.motor_feedback_csv_path}"
            )
            return
        self._open_motor_feedback_csv()
        self.get_logger().info(
            f"[motor-feedback-record] started reason={reason}, path={self.motor_feedback_csv_path}"
        )

    def stop_motor_feedback_recording(self, reason: str = "keyboard"):
        if self._motor_feedback_writer is None:
            self.get_logger().info("[motor-feedback-record] already stopped.")
            return
        path = str(self.motor_feedback_csv_path)
        self.close_motor_feedback_csv()
        self.get_logger().info(
            f"[motor-feedback-record] stopped reason={reason}, path={path}, frames={int(self.feedback_frame_idx)}"
        )

    def _write_motor_feedback_row(self, matrix: np.ndarray):
        if self._motor_feedback_writer is None:
            return
        row = [str(int(self.feedback_frame_idx)), f"{time.time():.6f}", f"{time.monotonic():.6f}"]
        for mid in self.motor_feedback_record_ids:
            pos = float(matrix[int(mid) - 1, 0])
            row.append("" if np.isnan(pos) else f"{pos:.6f}")
        self._motor_feedback_writer.writerow(row)
        self.feedback_frame_idx += 1

    def close_motor_feedback_csv(self):
        if self._motor_feedback_fp is None:
            return
        try:
            self._motor_feedback_fp.flush()
            self._motor_feedback_fp.close()
        finally:
            self._motor_feedback_fp = None
            self._motor_feedback_writer = None

    def _init_keyboard_record_control(self):
        if not sys.stdin.isatty():
            return
        try:
            self._keyboard_fd = sys.stdin.fileno()
            self._keyboard_termios_original = termios.tcgetattr(self._keyboard_fd)
            tty.setcbreak(self._keyboard_fd)
            self.keyboard_record_control_enabled = True
        except Exception as exc:
            self.keyboard_record_control_enabled = False
            self.get_logger().warn(f"[keyboard-control] disabled: {exc}")

    def close_keyboard_record_control(self):
        if self._keyboard_fd is None or self._keyboard_termios_original is None:
            return
        try:
            termios.tcsetattr(
                self._keyboard_fd,
                termios.TCSADRAIN,
                self._keyboard_termios_original,
            )
        except Exception:
            pass
        finally:
            self._keyboard_fd = None
            self._keyboard_termios_original = None
            self.keyboard_record_control_enabled = False

    def _keyboard_control_cb(self):
        if not self.keyboard_record_control_enabled:
            return
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        except Exception:
            return
        if not readable:
            return
        try:
            ch = sys.stdin.read(1)
        except Exception:
            return
        cmd = str(ch).strip().lower()
        if cmd == "r":
            self.start_motor_feedback_recording(reason="keyboard_r")
        elif cmd == "p":
            self.stop_motor_feedback_recording(reason="keyboard_p")

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
        self._write_motor_feedback_row(matrix)

    def _glove_cb(self, msg):
        self.glove_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_accept_ts) < self.glove_min_interval_sec:
            self.glove_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        if len(data) < 48:
            return
        glove20 = np.asarray(data[self.hand_glove_slice], dtype=np.float32)
        if glove20.shape[0] != 20:
            return
        self.latest_glove20 = glove20
        self.glove_pending.append(glove20)
        self.last_glove_ts = now
        self.last_glove_accept_ts = now
        self.glove_accept_total += 1

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
            self.get_logger().warn(f"[five_finger_delta_mapping] failed to compute right non-thumb q from raw skeleton: {exc}")
            return

        self.latest_right_non_thumb_q16 = q16_clean
        self.last_glove_raw_ts = now
        self.last_glove_raw_accept_ts = now
        self.glove_raw_accept_total += 1

    def _try_get_non_thumb_raw_q(self, finger_name: str, q_dim: int):
        q16_map = self.latest_right_non_thumb_q16
        if q16_map is None:
            return None
        q_vals = []
        for i in range(int(q_dim)):
            key = f"{finger_name}_q_{i}"
            if key not in q16_map:
                return None
            q_vals.append(float(q16_map[key]))
        return np.asarray(q_vals, dtype=np.float32)

    def _build_m_prev_from_feedback(self, motor_ids):
        m_prev = []
        for mid in motor_ids:
            idx = int(mid) - 1
            pos = float(self.latest_feedback_matrix[idx, 0])
            if np.isnan(pos):
                pos = float(self.last_cmd_matrix[idx][0])
            m_prev.append(pos)
        return np.asarray(m_prev, dtype=np.float32)

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

    def _apply_q_offsets(self, q_now: np.ndarray, q_indices):
        q_adj = q_now.copy()
        for i, qi in enumerate(q_indices):
            off = float(self.q_offsets_this_hand.get(int(qi), 0.0))
            if abs(off) > 1e-9:
                q_adj[i] = float(q_adj[i] + off)
        return q_adj

    def _build_cmd_from_glove20(self, glove20: np.ndarray):
        matrix = self.build_idle_tor_matrix()
        clip_last = 0
        infer_t0 = time.monotonic()

        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]

            if self.use_right_non_thumb_raw_skeleton_q and self.hand == "right" and name in RIGHT_NON_THUMB_FINGERS:
                q_now_raw = self._try_get_non_thumb_raw_q(name, b["q_dim"])
                if q_now_raw is not None:
                    self._raw_non_thumb_fallback_warned.discard(name)
                else:
                    if name not in self._raw_non_thumb_fallback_warned:
                        self.get_logger().warn(
                            "[five_finger_delta_mapping] right non-thumb raw skeleton q is not ready yet "
                            f"for finger={name}; skip inference for this finger this frame and hold previous command."
                        )
                        self._raw_non_thumb_fallback_warned.add(name)
                    self._hold_last_cmd_for_finger(matrix, b["motor_ids"])
                    continue
            else:
                q_now_raw = np.asarray([float(glove20[i]) for i in b["q_indices"]], dtype=np.float32)
            q_now = self._apply_q_offsets(q_now_raw, b["q_indices"])
            q_prev = q_now if b["prev_q"] is None else b["prev_q"]
            m_prev = self._build_m_prev_from_feedback(b["motor_ids"])
            dq = (q_now - q_prev)

            x = np.concatenate([q_now, q_prev, m_prev, dq], axis=0).reshape(1, -1)
            x_norm = b["x_scaler"].transform(x)

            with torch.no_grad():
                xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                pred_delta_norm = b["model"](xb).cpu().numpy()

            pred_delta = b["y_scaler"].inverse(pred_delta_norm)[0]
            pred_m = m_prev + pred_delta

            cmd_pos_list = []
            for out_idx, motor_id in enumerate(b["motor_ids"]):
                idx = int(motor_id) - 1
                pred_pos = float(pred_m[out_idx])
                offset = float(self.motor_cmd_offsets_this_hand.get(int(motor_id), 0.0))
                pred_pos += offset

                lim = self.motor_pos_limits.get(int(motor_id))
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
                cmd_pos_list.append(pred_pos)

            b["prev_q"] = q_now
            b["last_pred_delta"] = pred_delta.astype(np.float32)
            b["last_cmd_pos"] = np.asarray(cmd_pos_list, dtype=np.float32)

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
        glove_raw_age = (now - self.last_glove_raw_ts) if self.last_glove_raw_ts > 0.0 else float("inf")
        fb_age = (now - self.last_feedback_ts) if self.last_feedback_ts > 0.0 else float("inf")
        infer_load = (self.infer_ema_ms / (self.control_period_sec * 1000.0)) if self.control_period_sec > 0.0 else float("nan")

        finger_status = []
        for name in DEFAULT_FINGER_ORDER:
            if name not in self.finger_models:
                continue
            b = self.finger_models[name]
            cmd_pos = [round(float(v), 3) for v in b["last_cmd_pos"].tolist()]
            pred_delta = [round(float(v), 3) for v in b["last_pred_delta"].tolist()]
            finger_status.append(f"{name}:motors={b['motor_ids']},cmd={cmd_pos},delta={pred_delta}")
        raw_non_thumb_ready = (
            self.latest_right_non_thumb_q16 is not None
            if self.use_right_non_thumb_raw_skeleton_q
            else "disabled"
        )

        self.get_logger().info(
            "[five_finger_delta_mapping] "
            f"feedback_ready={self.feedback_received}, glove_ready={self.latest_glove20 is not None}, "
            f"fb_age={fb_age:.3f}s, glove_age={glove_age:.3f}s, "
            f"glove_rx={self.glove_rx_total}, glove_accept={self.glove_accept_total}, glove_drop_rl={self.glove_drop_rate_limited}, "
            f"glove_raw_age={glove_raw_age:.3f}s, glove_raw_rx={self.glove_raw_rx_total}, "
            f"glove_raw_accept={self.glove_raw_accept_total}, glove_raw_drop_rl={self.glove_raw_drop_rate_limited}, "
            f"right_non_thumb_q_raw_ready={raw_non_thumb_ready}, "
            f"pred_clip_last={self.clip_last}, pred_clip_total={self.clip_total}, "
            f"infer_last_ms={self.infer_last_ms:.3f}, infer_ema_ms={self.infer_ema_ms:.3f}, infer_load={infer_load:.2f}, "
            f"fingers={'; '.join(finger_status)}"
        )

    def _format_right_non_thumb_q16(self):
        q16_map = self.latest_right_non_thumb_q16
        if q16_map is None:
            return "not_ready"
        chunks = []
        for finger_name in RIGHT_NON_THUMB_FINGERS:
            vals = []
            for i in range(4):
                key = f"{finger_name}_q_{i}"
                val = q16_map.get(key)
                vals.append("nan" if val is None else f"{float(val):.3f}")
            chunks.append(f"{finger_name}=[{', '.join(vals)}]")
        return "; ".join(chunks)

    def _raw_q_status_cb(self):
        if not self.use_right_non_thumb_raw_skeleton_q:
            return
        now = time.monotonic()
        glove_raw_age = (now - self.last_glove_raw_ts) if self.last_glove_raw_ts > 0.0 else float("inf")
        self.get_logger().info(
            "[five_finger_delta_mapping] "
            f"right_non_thumb_q16={self._format_right_non_thumb_q16()}, "
            f"glove_raw_age={glove_raw_age:.3f}s"
        )

    def wait_ready(self, timeout_sec: float):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.feedback_received and self.latest_glove20 is not None:
                return True
        return False


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Five-finger delta-multihead mapping (single publisher runtime).")
    parser.add_argument("--hand", "--target-hand", dest="hand", default="right", choices=["left", "right"])
    parser.add_argument("--cmd-topic", default="", help="Override command topic")
    parser.add_argument("--feedback-topic", default="", help="Override feedback topic")
    parser.add_argument("--glove-topic", default=DEFAULT_GLOVE_TOPIC, help="48D glove topic (non-raw-skeleton)")
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default="/glove_joint_data_raw_skeleton",
        help="176D raw-skeleton topic (Float64MultiArray).",
    )
    raw_non_thumb_group = parser.add_mutually_exclusive_group()
    raw_non_thumb_group.add_argument(
        "--right-index-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_true",
        default=True,
        help=(
            "Deprecated alias. Same as --right-non-thumb-q-from-raw-skeleton: "
            "use 176D raw skeleton to solve q_0..q_3 for right index/middle/ring/little (default: enabled)."
        ),
    )
    raw_non_thumb_group.add_argument(
        "--right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_true",
        help="For right-hand index/middle/ring/little, solve q_0..q_3 from 176D raw skeleton instead of /glove_joint_data indices (default: enabled).",
    )
    raw_non_thumb_group.add_argument(
        "--no-right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_false",
        help="Disable raw-skeleton q replacement for right non-thumb fingers and use glove48 q indices.",
    )
    raw_non_thumb_group.add_argument(
        "--no-right-index-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_false",
        help="Deprecated alias for --no-right-non-thumb-q-from-raw-skeleton.",
    )
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
        help="Path to muscle_adjustment_defaults.py",
    )

    for finger_name in DEFAULT_FINGER_ORDER:
        _add_finger_cli_args(parser, finger_name=finger_name)

    parser.add_argument("--control-period-sec", type=float, default=float(CONTROL_PERIOD_SEC))
    parser.add_argument("--status-period-sec", type=float, default=2.0)
    parser.add_argument(
        "--glove-max-hz",
        type=float,
        default=100.0,
        help="Max accepted glove callback rate. <=0 disables rate limiting",
    )
    parser.add_argument(
        "--glove-sub-queue-size",
        type=int,
        default=60,
        help="ROS subscription queue size for glove topic (>=1).",
    )
    parser.add_argument(
        "--glove-pending-maxlen",
        type=int,
        default=60,
        help="Internal pending glove frame buffer length (>=1).",
    )
    parser.add_argument(
        "--record-motor-feedback-csv",
        action="store_true",
        help="If set, record each motor feedback frame to CSV (24 motor pos columns + frame index).",
    )
    parser.add_argument(
        "--motor-feedback-csv-path",
        default="auto",
        help="Motor feedback CSV output path. Supports placeholders: {hand}, {pid}, {timestamp}.",
    )
    parser.add_argument(
        "--motor-feedback-limits-left-json",
        default="",
        help="Override left-hand MOTOR_FEEDBACK_LIMITS via JSON, e.g. {\"26\": [-50, 500], \"28\": [-600, 50]}",
    )
    parser.add_argument(
        "--motor-feedback-limits-right-json",
        default="",
        help="Override right-hand MOTOR_FEEDBACK_LIMITS via JSON, e.g. {\"12\": [-50, 500], \"14\": [-600, 50]}",
    )
    parser.add_argument(
        "--q-offset-map-json",
        default=json.dumps(Q_OFFSET_MAP_DEFAULTS, ensure_ascii=False),
        help="Glove q offset map JSON by hand and q_index, e.g. {\"left\": {\"4\": 0.0}, \"right\": {\"8\": 10.0}}",
    )
    parser.add_argument(
        "--motor-cmd-offset-map-json",
        default=json.dumps(MOTOR_CMD_OFFSET_MAP_DEFAULTS, ensure_ascii=False),
        help="Motor command offset map JSON by hand and motor_id, e.g. {\"left\": {\"8\": 5.0}, \"right\": {\"22\": 5.0}}",
    )

    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--require-ready", action="store_true", help="Exit if startup is not ready before timeout")
    parser.add_argument("--ready-timeout-sec", type=float, default=12.0)
    args = parser.parse_args()

    hand = str(args.hand).strip().lower()
    hand_defaults = FINGER_DEFAULTS_BY_HAND[hand]
    for finger_name in DEFAULT_FINGER_ORDER:
        cfg = hand_defaults[finger_name]

        enable_attr = f"enable_{finger_name}"
        if getattr(args, enable_attr) is None:
            setattr(args, enable_attr, cfg["enable_default"])

        ckpt_attr = f"{finger_name}_model_ckpt"
        if getattr(args, ckpt_attr) is None:
            setattr(args, ckpt_attr, str(script_dir.joinpath(*cfg["model_ckpt_rel_parts"])))

        motor_attr = f"{finger_name}_model_motor_ids"
        if getattr(args, motor_attr) is None:
            setattr(args, motor_attr, cfg["motor_ids_csv"])

        q_attr = f"{finger_name}_q_indices"
        if getattr(args, q_attr) is None:
            setattr(args, q_attr, cfg["q_indices_csv"])

        mirror_attr = f"{finger_name}_mirror_right_from_left"
        if getattr(args, mirror_attr) is None:
            setattr(args, mirror_attr, cfg["mirror_right_from_left_default"])

    return args


def _build_finger_specs(args):
    raw = {}
    for finger_name in DEFAULT_FINGER_ORDER:
        raw[finger_name] = {
            "enabled": _parse_bool_arg(
                getattr(args, f"enable_{finger_name}"),
                f"--enable-{finger_name}",
            ),
            "model_ckpt": getattr(args, f"{finger_name}_model_ckpt"),
            "motor_ids": _parse_csv_ints(getattr(args, f"{finger_name}_model_motor_ids")),
            "q_indices": _parse_csv_ints(getattr(args, f"{finger_name}_q_indices")),
            "mirror_right_from_left": _parse_bool_arg(
                getattr(args, f"{finger_name}_mirror_right_from_left"),
                f"--{finger_name}-mirror-right-from-left",
            ),
        }

    for finger_name, spec in raw.items():
        if not spec["enabled"]:
            continue
        _validate_motor_ids(spec["motor_ids"], f"--{finger_name}-model-motor-ids")
        _validate_q_indices(spec["q_indices"], f"--{finger_name}-q-indices")
    return raw


def main():
    args = parse_args()
    hand = args.hand.strip().lower()
    cmd_topic, feedback_topic = get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic.strip()
    if args.feedback_topic:
        feedback_topic = args.feedback_topic.strip()

    defaults_hand = _load_defaults_from_py(Path(args.defaults_path), hand)
    finger_specs = _build_finger_specs(args)
    motor_feedback_limits_left = _parse_motor_feedback_limits_json(
        args.motor_feedback_limits_left_json,
        "--motor-feedback-limits-left-json",
    )
    motor_feedback_limits_right = _parse_motor_feedback_limits_json(
        args.motor_feedback_limits_right_json,
        "--motor-feedback-limits-right-json",
    )
    q_offset_map = _parse_q_offset_map_json(
        args.q_offset_map_json,
        "--q-offset-map-json",
    )
    motor_cmd_offset_map = _parse_motor_cmd_offset_map_json(
        args.motor_cmd_offset_map_json,
        "--motor-cmd-offset-map-json",
    )

    rclpy.init()
    node = None
    try:
        use_right_non_thumb_raw = bool(args.right_non_thumb_q_from_raw_skeleton)
        node = FiveFingerDeltaMappingNode(
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            glove_topic=args.glove_topic.strip(),
            glove_raw_skeleton_topic=args.glove_raw_skeleton_topic.strip(),
            right_non_thumb_q_from_raw_skeleton=use_right_non_thumb_raw,
            defaults_hand=defaults_hand,
            finger_specs=finger_specs,
            device=args.device.strip().lower(),
            control_period_sec=float(args.control_period_sec),
            status_period_sec=float(args.status_period_sec),
            glove_max_hz=float(args.glove_max_hz),
            glove_sub_queue_size=int(args.glove_sub_queue_size),
            glove_pending_maxlen=int(args.glove_pending_maxlen),
            record_motor_feedback_csv=bool(args.record_motor_feedback_csv),
            motor_feedback_csv_path=str(args.motor_feedback_csv_path),
            motor_cmd_offset_map=motor_cmd_offset_map,
            q_offset_map=q_offset_map,
            motor_feedback_limits_left=motor_feedback_limits_left,
            motor_feedback_limits_right=motor_feedback_limits_right,
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
        print("\n[five_finger_delta_mapping] KeyboardInterrupt received, exiting...")
    finally:
        if node is not None:
            try:
                node.publish_idle_once()
            except Exception:
                pass
            try:
                node.close_motor_feedback_csv()
            except Exception:
                pass
            try:
                node.close_keyboard_record_control()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
