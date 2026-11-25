"""Non-thumb q0 and curl model support for wrist fusion."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from hand_motor_map import HAND_MOTOR_MAP, MOTOR_COUNT, RIGHT_TO_LEFT_MOTOR_MAP, model_motor_ids

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
    from replace_right_non_thumb_q_from_skeleton import compute_non_thumb_q16
except Exception:  # pragma: no cover
    compute_non_thumb_q16 = None


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

SUPPORTED_MODEL_FEATURE_COLS = {
    "target_q_0",
    "target_curl",
}
RIGHT_SIDE_SWING_MOTOR_IDS = {
    mid
    for finger, muscles in HAND_MOTOR_MAP["right"].items() if finger != "thumb"
    for role, mid in muscles.items() if role in ("abduction", "adduction")
}
LEFT_SIDE_SWING_MOTOR_IDS = {
    mid
    for finger, muscles in HAND_MOTOR_MAP["left"].items() if finger != "thumb"
    for role, mid in muscles.items() if role in ("abduction", "adduction")
}


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


def load_finger_model(finger: str, spec: dict, *, hand: str, device: str):
    model_motor_ids = list(spec["model_motor_ids"])
    if hand == "left" and spec["mirror_left_from_right"]:
        model_motor_ids = _mirror_motor_ids_right_to_left(model_motor_ids)

    model_bundle = _load_curl_model(spec["model_ckpt"], device)
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
