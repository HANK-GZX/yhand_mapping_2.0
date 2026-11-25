"""Delta checkpoint inference and configuration shared by the wrist fusion controllers."""

import json
from pathlib import Path

import numpy as np

from hand_motor_map import LEFT_TO_RIGHT_MOTOR_MAP, model_motor_ids_csv

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


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

DEFAULT_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")

RIGHT_NON_THUMB_FINGERS = ("index", "middle", "ring", "little")

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
            "[five_finger_delta_models] warning: ignoring legacy auxiliary "
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
