#!/usr/bin/env python3
"""Shared helpers for the back-delta software-zero workflow and proxy."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


from hand_motor_map import (
    HAND_MOTOR_MAP, WRIST_MOTOR_IDS_BY_HAND,
)

MOTOR_COUNT = 28
HAND_CHOICES = ("left", "right")
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "software_zero_calibration_config.json"
DEFAULT_DEFAULTS_PATH = PACKAGE_ROOT / "scripts" / "muscle_adjustment_defaults.py"
DEFAULT_REFERENCE_DIR = "scripts/software_zero/reference"
DEFAULT_STATE_DIR = "scripts/software_zero/state"
BACK_DELTA_REFERENCE_KIND = "software_zero_back_delta_reference"
STATE_KIND = "software_zero_state"

HAND_GROUPS = {
    hand: {"wrist": list(WRIST_MOTOR_IDS_BY_HAND[hand]),
           "thumb": sorted(fingers["thumb"].values()),
           "four_finger": sorted(mid for finger, motors in fingers.items()
                                 if finger != "thumb" for mid in motors.values())}
    for hand, fingers in HAND_MOTOR_MAP.items()
}


def normalize_hand(hand: str) -> str:
    value = str(hand or "").strip().lower()
    if value not in HAND_CHOICES:
        raise ValueError(f"hand must be one of {HAND_CHOICES}, got {hand!r}")
    return value


def hand_groups(hand: str) -> dict[str, list[int]]:
    return {key: list(value) for key, value in HAND_GROUPS[normalize_hand(hand)].items()}


def resolve_pkg_path(path_like: str | Path | None, *, base: Path = PACKAGE_ROOT) -> Path | None:
    if path_like is None:
        return None
    raw = str(path_like).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def save_json_atomic(path: str | Path, obj: Any) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(p.parent),
        prefix=f".{p.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp = Path(f.name)
        f.write(text)
        f.write("\n")
    tmp.replace(p)


def current_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def validate_zero_config(config: dict[str, Any]) -> None:
    currents = config.get("currents", {})
    max_abs = float(currents.get("max_current_abs_a", 0.5))
    for key in ("release_current_a", "tension_current_a"):
        value = float(currents.get(key, 0.0))
        if abs(value) > max_abs + 1e-9:
            raise ValueError(f"{key}={value} exceeds max_current_abs_a={max_abs}")

    groups = config.get("back_delta_motor_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("back_delta_motor_groups must be an object")
    for hand in HAND_CHOICES:
        hand_groups_obj = groups.get(hand)
        if not isinstance(hand_groups_obj, dict):
            raise ValueError(f"back_delta_motor_groups.{hand} missing")
        for key in ("front_release", "rear_contract", "rear_side"):
            values = hand_groups_obj.get(key)
            if not isinstance(values, list):
                raise ValueError(f"back_delta_motor_groups.{hand}.{key} must be a list")
            bad = [mid for mid in values if int(mid) < 1 or int(mid) > MOTOR_COUNT]
            if bad:
                raise ValueError(f"back_delta_motor_groups.{hand}.{key} has invalid motor ids: {bad}")


def load_zero_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_pkg_path(path or DEFAULT_CONFIG_PATH)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"software-zero config not found: {cfg_path}")
    cfg = load_json(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"software-zero config must be object: {cfg_path}")
    validate_zero_config(cfg)
    return cfg


def default_reference_path(config: dict[str, Any], hand: str) -> Path:
    ref_dir = config.get("paths", {}).get("reference_dir", DEFAULT_REFERENCE_DIR)
    return resolve_pkg_path(ref_dir).joinpath(f"{normalize_hand(hand)}_back_delta_reference.json")


def default_state_path(config: dict[str, Any], hand: str) -> Path:
    state_dir = config.get("paths", {}).get("state_dir", DEFAULT_STATE_DIR)
    return resolve_pkg_path(state_dir).joinpath(f"{normalize_hand(hand)}_back_delta_state.json")


def as_28_list(raw: Any, *, label: str) -> list[float]:
    if not isinstance(raw, list) or len(raw) != MOTOR_COUNT:
        raise ValueError(f"{label} must be list(len={MOTOR_COUNT})")
    return [float(v) for v in raw]


def load_state(path: str | Path, hand: str) -> dict[str, Any]:
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"state must be object: {path}")
    if str(obj.get("kind", "")).strip() != STATE_KIND:
        raise ValueError(f"not a {STATE_KIND} file: {path}")
    if normalize_hand(obj.get("hand", "")) != normalize_hand(hand):
        raise ValueError(f"state hand mismatch: expected {hand}, got {obj.get('hand')}")
    as_28_list(obj.get("zero_offset"), label="state.zero_offset")
    return obj


def zero_offset_from_state(state: dict[str, Any]) -> list[float]:
    return as_28_list(state.get("zero_offset"), label="state.zero_offset")


def build_state_payload(
    hand: str,
    mode: str,
    zero_offset: list[float],
    runtime_data: dict[str, list[float]],
    reference_path: str | Path,
    config_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hand = normalize_hand(hand)
    return {
        "version": "1.0",
        "kind": STATE_KIND,
        "hand": hand,
        "mode": str(mode),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boot_id": current_boot_id(),
        "config_path": str(config_path),
        "reference_path": str(reference_path),
        "zero_offset": [float(v) for v in zero_offset],
        "runtime_data": {key: [float(v) for v in values] for key, values in runtime_data.items()},
        "groups": hand_groups(hand),
        "metadata": dict(metadata or {}),
    }


def load_muscle_defaults(defaults_path: str | Path | None, hand: str) -> dict[str, list[float]]:
    hand = normalize_hand(hand)
    path = resolve_pkg_path(defaults_path or DEFAULT_DEFAULTS_PATH)
    if path is None or not path.exists():
        raise FileNotFoundError(f"defaults file not found: {path}")
    spec = importlib.util.spec_from_file_location("software_zero_defaults_dynamic", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load defaults file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if not isinstance(defaults, dict) or hand not in defaults:
        raise RuntimeError(f"DEFAULTS[{hand!r}] not found in {path}")
    raw = defaults[hand]
    out: dict[str, list[float]] = {}
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.03)):
        values = list(raw.get(key, []))
        if len(values) < MOTOR_COUNT:
            values.extend([fallback] * (MOTOR_COUNT - len(values)))
        out[key] = [float(v) for v in values[:MOTOR_COUNT]]
    out["tor_active"] = [float(v) for v in list(raw.get("tor_active", out["tor"]))[:MOTOR_COUNT]]
    if len(out["tor_active"]) < MOTOR_COUNT:
        out["tor_active"].extend(out["tor"][len(out["tor_active"]):])
    return out
