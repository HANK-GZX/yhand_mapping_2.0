#!/usr/bin/env python3
import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None


from hand_motor_map import (
    FINGER_MOTOR_IDS_BY_HAND, MAPPING_VERSION, finger_assist_motor_layout,
)

MOTOR_COUNT = 28
CONTROL_PERIOD_SEC = 0.02
ARRIVAL_POS_TOL = 7.0
ARRIVAL_CONFIRM_FRAMES = 30
SINGLE_MOTOR_TIMEOUT_SEC = 5.0
RETURN_ZERO_TIMEOUT_SEC = 6.0
REPLAY_SNAPSHOT_TIMEOUT_SEC = 6.0
FEEDBACK_STALE_SEC = 1.0
TOPIC_CONFLICT_CONFIRM_SEC = 2.0
TOR_PAIR_MAX_DIFF_A = 0.1
ZERO_SWAY_RELEASE_TOR_A = 0.1
ZERO_SWAY_RELEASE_SEC = 0.0
ZERO_FLEXOR_RELEASE_TOR_A = 0.2
ZERO_FLEXOR_RELEASE_SEC = 0.5
SNAPSHOT_FILE_SCHEMA = "motor_position_test.snapshot.v1"
MOTOR_FEEDBACK_RECORD_IDS_BY_HAND = dict(FINGER_MOTOR_IDS_BY_HAND)

# Align antagonist-assist behavior with muscle_adjustment.py
ASSIST_MODE_OFF = "off"
ASSIST_MODE_RELEASE = "release_tor"
ASSIST_RELEASE_TOR = 0.5
# Thumb opposition <-> sway cross-release current (A).
ASSIST_RELEASE_TOR_THUMB_OPPOSITION_SWAY = 0.5
# Thumb opposition <-> sway cross-release duration (sec).
ASSIST_RELEASE_SEC_THUMB_OPPOSITION_SWAY = 0.2
# Fallback assist release duration used only when motor category is unknown.
ASSIST_RELEASE_SEC = 0.5
# Legacy fixed sway release duration.
# Kept only as last-resort fallback when sway per-degree config is unavailable.
ASSIST_RELEASE_SEC_SWAY_FIXED = 0.3
# Delta-based release scaling:
# release_sec = delta_deg * per_deg * scale(delta_deg)
ASSIST_RELEASE_DELTA_SWITCH_DEG = 20.0
ASSIST_RELEASE_PER_DEG_SCALE_SMALL_DELTA = 0.3
ASSIST_RELEASE_PER_DEG_SCALE_LARGE_DELTA = 1.0

# Finger-centric assist configuration (keep each finger's parameters together).
FINGER_ASSIST_CONFIG = {'left': {'index': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                            'superficial': {'per_deg': 0.0027, 'release_tor': 0.5}},
                    'flex_release_per_deg': {'deep': {'per_deg': 0.001, 'release_tor': 0.5},
                                             'superficial': {'per_deg': 0.001, 'release_tor': 0.5}},
                    'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00018,
                                                                     'add': 0.00018,
                                                                     'release_tor': 0.5},
                                                             'flex': {'abd': 0.0001,
                                                                      'add': 5e-05,
                                                                      'release_tor': 0.5}}},
          'middle': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.45},
                                             'superficial': {'per_deg': 0.0019,
                                                             'release_tor': 0.45}},
                     'flex_release_per_deg': {'deep': {'per_deg': 0.0008, 'release_tor': 0.5},
                                              'superficial': {'per_deg': 0.0008,
                                                              'release_tor': 0.5}},
                     'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00015,
                                                                      'abd_release_tor': 0.5,
                                                                      'add': 0.0001,
                                                                      'add_release_tor': 0.4},
                                                              'flex': {'abd': 0.0,
                                                                       'add': 0.0,
                                                                       'release_tor': 0.5}}},
          'ring': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                           'superficial': {'per_deg': 0.0015, 'release_tor': 0.5}},
                   'flex_release_per_deg': {'deep': {'per_deg': 0.001, 'release_tor': 0.5},
                                            'superficial': {'per_deg': 0.001, 'release_tor': 0.5}},
                   'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00015,
                                                                    'add': 0.00013,
                                                                    'release_tor': 0.5},
                                                            'flex': {'abd': 0.00015,
                                                                     'add': 8e-05,
                                                                     'release_tor': 0.5}}},
          'little': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                             'superficial': {'per_deg': 0.0015,
                                                             'release_tor': 0.5}},
                     'flex_release_per_deg': {'deep': {'per_deg': 0.001, 'release_tor': 0.5},
                                              'superficial': {'per_deg': 0.001,
                                                              'release_tor': 0.5}},
                     'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.0001,
                                                                      'add': 0.00015,
                                                                      'release_tor': 0.5},
                                                              'flex': {'abd': 5e-05,
                                                                       'add': 0.00015,
                                                                       'release_tor': 0.5}}},
          'thumb': {'ext_release_per_deg': 0.0015,
                    'flex_release_per_deg': {'short_ext': 0.002, 'long_ext': 0.0005},
                    'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.0,
                                                                     'add': 0.0,
                                                                     'release_tor': 0.5},
                                                             'flex': {'abd': 0.0,
                                                                      'add': 0.0,
                                                                      'release_tor': 0.5}}}},
 'right': {'index': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                             'superficial': {'per_deg': 0.0027,
                                                             'release_tor': 0.5}},
                     'flex_release_per_deg': {'deep': {'per_deg': 0.005, 'release_tor': 0.5},
                                              'superficial': {'per_deg': 0.001,
                                                              'release_tor': 0.5}},
                     'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00018,
                                                                      'add': 0.00018,
                                                                      'release_tor': 0.5},
                                                              'flex': {'abd': 0.0001,
                                                                       'add': 5e-05,
                                                                       'release_tor': 0.5}}},
           'middle': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.45},
                                              'superficial': {'per_deg': 0.0019,
                                                              'release_tor': 0.45}},
                      'flex_release_per_deg': {'deep': {'per_deg': 0.0008, 'release_tor': 0.5},
                                               'superficial': {'per_deg': 0.0008,
                                                               'release_tor': 0.5}},
                      'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00015,
                                                                       'abd_release_tor': 0.5,
                                                                       'add': 0.0001,
                                                                       'add_release_tor': 0.4},
                                                               'flex': {'abd': 0.0,
                                                                        'add': 0.0,
                                                                        'release_tor': 0.5}}},
           'ring': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                            'superficial': {'per_deg': 0.0015, 'release_tor': 0.5}},
                    'flex_release_per_deg': {'deep': {'per_deg': 0.001, 'release_tor': 0.5},
                                             'superficial': {'per_deg': 0.001, 'release_tor': 0.5}},
                    'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.00015,
                                                                     'add': 0.00013,
                                                                     'release_tor': 0.5},
                                                             'flex': {'abd': 0.00015,
                                                                      'add': 8e-05,
                                                                      'release_tor': 0.5}}},
           'little': {'ext_release_per_deg': {'deep': {'per_deg': 0.0015, 'release_tor': 0.5},
                                              'superficial': {'per_deg': 0.0015,
                                                              'release_tor': 0.5}},
                      'flex_release_per_deg': {'deep': {'per_deg': 0.001, 'release_tor': 0.5},
                                               'superficial': {'per_deg': 0.001,
                                                               'release_tor': 0.5}},
                      'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.0001,
                                                                       'add': 0.00015,
                                                                       'release_tor': 0.5},
                                                               'flex': {'abd': 5e-05,
                                                                        'add': 0.00015,
                                                                        'release_tor': 0.5}}},
           'thumb': {'ext_release_per_deg': 0.0015,
                     'flex_release_per_deg': {'short_ext': 0.002, 'long_ext': 0.0005},
                     'sway_pair_release_per_deg_on_extflex': {'ext': {'abd': 0.0,
                                                                      'add': 0.0,
                                                                      'release_tor': 0.5},
                                                              'flex': {'abd': 0.0,
                                                                       'add': 0.0,
                                                                       'release_tor': 0.5}}}}}
FINGER_ASSIST_CONFIG = {
    hand: {finger: {**params, **finger_assist_motor_layout(hand, finger)}
           for finger, params in fingers.items()}
    for hand, fingers in FINGER_ASSIST_CONFIG.items()
}


def _extract_axis_per_deg_from_cfg(axis_cfg, default_per_deg: float):
    if isinstance(axis_cfg, dict):
        return float(
            axis_cfg.get(
                "per_deg",
                axis_cfg.get(
                    "sec_per_deg",
                    axis_cfg.get("release_per_deg", default_per_deg),
                ),
            )
        )
    if axis_cfg is None:
        return float(default_per_deg)
    return float(axis_cfg)


def _normalize_abd_add_cfg(cfg_value, default_abd: float = 0.0, default_add: float | None = None):
    if default_add is None:
        default_add = default_abd
    if isinstance(cfg_value, dict):
        abd_raw = cfg_value.get("abd", cfg_value.get("out", default_abd))
        abd_v = _extract_axis_per_deg_from_cfg(abd_raw, default_abd)
        add_raw = cfg_value.get("add", cfg_value.get("in", abd_raw))
        add_v = _extract_axis_per_deg_from_cfg(add_raw, abd_v)
        return {"abd": abd_v, "add": add_v}
    if cfg_value is None:
        return {"abd": float(default_abd), "add": float(default_add)}
    scalar_v = float(cfg_value)
    return {"abd": scalar_v, "add": scalar_v}


def _normalize_sway_extflex_cfg(cfg_value):
    # New format:
    # {"ext": {"abd": x, "add": y}, "flex": {"abd": u, "add": v}}
    # Legacy format keeps one set for both ext/flex:
    # {"abd": x, "add": y} or scalar.
    if isinstance(cfg_value, dict) and any(
        key in cfg_value
        for key in ("ext", "flex", "extensor", "flexor", "on_ext", "on_flex")
    ):
        ext_raw = cfg_value.get(
            "ext",
            cfg_value.get("extensor", cfg_value.get("on_ext", None)),
        )
        ext_cfg = _normalize_abd_add_cfg(ext_raw, 0.0, 0.0)
        flex_raw = cfg_value.get(
            "flex",
            cfg_value.get("flexor", cfg_value.get("on_flex", None)),
        )
        flex_cfg = _normalize_abd_add_cfg(
            flex_raw,
            default_abd=float(ext_cfg["abd"]),
            default_add=float(ext_cfg["add"]),
        )
        return {"ext": ext_cfg, "flex": flex_cfg}

    shared_cfg = _normalize_abd_add_cfg(cfg_value, 0.0, 0.0)
    return {
        "ext": dict(shared_cfg),
        "flex": dict(shared_cfg),
    }


def _extract_release_tor_from_role_cfg(role_cfg, default_tor: float):
    if isinstance(role_cfg, dict):
        if "release_tor" in role_cfg:
            return float(role_cfg["release_tor"])
        if "tor" in role_cfg:
            return float(role_cfg["tor"])
        if "current" in role_cfg:
            return float(role_cfg["current"])
        return float(default_tor)
    if role_cfg is None:
        return float(default_tor)
    return float(role_cfg)


def _extract_axis_release_tor_from_cfg(axis_cfg, default_tor: float, allow_numeric_axis: bool = False):
    if isinstance(axis_cfg, dict):
        if "release_tor" in axis_cfg:
            return float(axis_cfg["release_tor"])
        if "tor" in axis_cfg:
            return float(axis_cfg["tor"])
        if "current" in axis_cfg:
            return float(axis_cfg["current"])
        return float(default_tor)
    if axis_cfg is None:
        return float(default_tor)
    if not allow_numeric_axis:
        return float(default_tor)
    return float(axis_cfg)


def _normalize_abd_add_release_tor_cfg(
    cfg_value,
    default_abd_tor: float = ASSIST_RELEASE_TOR,
    default_add_tor: float | None = None,
    allow_numeric_axis: bool = False,
):
    if default_add_tor is None:
        default_add_tor = default_abd_tor
    if isinstance(cfg_value, dict):
        shared_tor = _extract_release_tor_from_role_cfg(cfg_value, default_abd_tor)
        abd_raw = cfg_value.get(
            "abd_release_tor",
            cfg_value.get(
                "abd_tor",
                cfg_value.get(
                    "out_release_tor",
                    cfg_value.get("out_tor", cfg_value.get("abd", cfg_value.get("out", None))),
                ),
            ),
        )
        abd_tor = _extract_axis_release_tor_from_cfg(
            abd_raw,
            shared_tor,
            allow_numeric_axis=allow_numeric_axis,
        )
        add_raw = cfg_value.get(
            "add_release_tor",
            cfg_value.get(
                "add_tor",
                cfg_value.get(
                    "in_release_tor",
                    cfg_value.get("in_tor", cfg_value.get("add", cfg_value.get("in", None))),
                ),
            ),
        )
        add_tor = _extract_axis_release_tor_from_cfg(
            add_raw,
            abd_tor,
            allow_numeric_axis=allow_numeric_axis,
        )
        return {"abd": float(abd_tor), "add": float(add_tor)}
    if cfg_value is None:
        return {"abd": float(default_abd_tor), "add": float(default_add_tor)}
    scalar_tor = float(cfg_value)
    return {"abd": scalar_tor, "add": scalar_tor}


def _normalize_sway_extflex_release_tor_cfg(
    cfg_value,
    default_tor: float = ASSIST_RELEASE_TOR,
    allow_numeric_axis: bool = False,
):
    # New format:
    # {"ext": {"abd_release_tor": x, "add_release_tor": y}, "flex": {...}}
    # or {"ext": {"abd": {"release_tor": x}, "add": {"release_tor": y}}, "flex": {...}}
    # Legacy/shared format:
    # {"release_tor": x} or scalar x
    if isinstance(cfg_value, dict) and any(
        key in cfg_value
        for key in ("ext", "flex", "extensor", "flexor", "on_ext", "on_flex")
    ):
        top_default = float(
            cfg_value.get(
                "release_tor",
                cfg_value.get("tor", cfg_value.get("current", default_tor)),
            )
        )
        ext_raw = cfg_value.get(
            "ext",
            cfg_value.get("extensor", cfg_value.get("on_ext", None)),
        )
        ext_cfg = _normalize_abd_add_release_tor_cfg(
            ext_raw,
            default_abd_tor=top_default,
            default_add_tor=top_default,
            allow_numeric_axis=allow_numeric_axis,
        )
        flex_raw = cfg_value.get(
            "flex",
            cfg_value.get("flexor", cfg_value.get("on_flex", None)),
        )
        flex_cfg = _normalize_abd_add_release_tor_cfg(
            flex_raw,
            default_abd_tor=float(ext_cfg["abd"]),
            default_add_tor=float(ext_cfg["add"]),
            allow_numeric_axis=allow_numeric_axis,
        )
        return {"ext": ext_cfg, "flex": flex_cfg}

    shared_cfg = _normalize_abd_add_release_tor_cfg(
        cfg_value,
        default_abd_tor=default_tor,
        default_add_tor=default_tor,
        allow_numeric_axis=allow_numeric_axis,
    )
    return {"ext": dict(shared_cfg), "flex": dict(shared_cfg)}


def _normalize_extflex_release_tor_cfg(cfg_value, default_tor: float = ASSIST_RELEASE_TOR):
    # New format:
    # {"ext": {"release_tor": x, ...}, "flex": {"release_tor": y, ...}}
    # or {"ext": x, "flex": y}
    # Legacy/shared format:
    # {"release_tor": x} or scalar x
    if isinstance(cfg_value, dict) and any(
        key in cfg_value
        for key in ("ext", "flex", "extensor", "flexor", "on_ext", "on_flex")
    ):
        top_default = float(
            cfg_value.get(
                "release_tor",
                cfg_value.get("tor", cfg_value.get("current", default_tor)),
            )
        )
        ext_raw = cfg_value.get(
            "ext",
            cfg_value.get("extensor", cfg_value.get("on_ext", None)),
        )
        ext_tor = _extract_release_tor_from_role_cfg(ext_raw, top_default)
        flex_raw = cfg_value.get(
            "flex",
            cfg_value.get("flexor", cfg_value.get("on_flex", None)),
        )
        flex_tor = _extract_release_tor_from_role_cfg(flex_raw, ext_tor)
        return {"ext": float(ext_tor), "flex": float(flex_tor)}

    shared_tor = _extract_release_tor_from_role_cfg(cfg_value, default_tor)
    return {"ext": float(shared_tor), "flex": float(shared_tor)}


def _normalize_ext_flexor_split_cfg(cfg_value, default_per_deg: float = 0.0015):
    # New format for extensor release on fingers with deep/superficial flexors:
    # {"deep": x, "superficial": y}
    # Legacy scalar keeps both equal.
    if isinstance(cfg_value, dict):
        deep_v = float(
            cfg_value.get(
                "deep",
                cfg_value.get("fdp", cfg_value.get("deep_flexor", default_per_deg)),
            )
        )
        superficial_v = float(
            cfg_value.get(
                "superficial",
                cfg_value.get(
                    "shallow",
                    cfg_value.get(
                        "fds",
                        cfg_value.get("superficial_flexor", cfg_value.get("superficialFlexor", deep_v)),
                    ),
                ),
            )
        )
        return {"deep": deep_v, "superficial": superficial_v}
    scalar_v = float(default_per_deg if cfg_value is None else cfg_value)
    return {"deep": scalar_v, "superficial": scalar_v}


def _normalize_muscle_side_cfg(side_value, default_per_deg: float, default_tor: float):
    if isinstance(side_value, dict):
        per_deg = float(
            side_value.get(
                "per_deg",
                side_value.get(
                    "sec_per_deg",
                    side_value.get("release_per_deg", default_per_deg),
                ),
            )
        )
        release_tor = _extract_release_tor_from_role_cfg(side_value, default_tor)
        return {"per_deg": per_deg, "release_tor": float(release_tor)}
    if side_value is None:
        return {"per_deg": float(default_per_deg), "release_tor": float(default_tor)}
    return {"per_deg": float(side_value), "release_tor": float(default_tor)}


def _normalize_muscle_split_cfg(cfg_value, default_per_deg: float, default_tor: float):
    # Flexible format for non-thumb deep/superficial muscles:
    # 1) scalar: 0.0015
    # 2) split scalar: {"deep": 0.0014, "superficial": 0.0020}
    # 3) split full:
    #    {"deep": {"per_deg": 0.0014, "release_tor": 0.45},
    #     "superficial": {"per_deg": 0.0020, "release_tor": 0.55}}
    # 4) shared full: {"per_deg": 0.0015, "release_tor": 0.5}
    if isinstance(cfg_value, dict):
        has_split = any(
            key in cfg_value
            for key in (
                "deep",
                "superficial",
                "fdp",
                "fds",
                "deep_flexor",
                "superficial_flexor",
                "superficialFlexor",
                "shallow",
            )
        )
        shared_default = _normalize_muscle_side_cfg(cfg_value, default_per_deg, default_tor)
        if has_split:
            deep_raw = cfg_value.get(
                "deep",
                cfg_value.get("fdp", cfg_value.get("deep_flexor", None)),
            )
            superficial_raw = cfg_value.get(
                "superficial",
                cfg_value.get(
                    "shallow",
                    cfg_value.get(
                        "fds",
                        cfg_value.get("superficial_flexor", cfg_value.get("superficialFlexor", None)),
                    ),
                ),
            )
            deep_cfg = _normalize_muscle_side_cfg(
                deep_raw,
                float(shared_default["per_deg"]),
                float(shared_default["release_tor"]),
            )
            superficial_cfg = _normalize_muscle_side_cfg(
                superficial_raw,
                float(deep_cfg["per_deg"]),
                float(deep_cfg["release_tor"]),
            )
            return {"deep": deep_cfg, "superficial": superficial_cfg}
        return {"deep": dict(shared_default), "superficial": dict(shared_default)}

    shared_cfg = _normalize_muscle_side_cfg(cfg_value, default_per_deg, default_tor)
    return {"deep": dict(shared_cfg), "superficial": dict(shared_cfg)}

def _build_assist_derived_maps(finger_cfg):
    extensor_assist_map = {"left": {}, "right": {}}
    assist_duration_category = {
        "left": {"extensor": set(), "flexor": set(), "sway": set()},
        "right": {"extensor": set(), "flexor": set(), "sway": set()},
    }
    motor_to_finger = {"left": {}, "right": {}}
    finger_sway_pair = {"left": {}, "right": {}}
    thumb_extensor_ids = {"left": {}, "right": {}}
    thumb_sway_pair_for_cross = {"left": [], "right": []}
    thumb_opposition_motor = {}
    extensor_split_flexor_targets = {"left": {}, "right": {}}
    flexor_role_by_motor = {"left": {}, "right": {}}

    ext_per_deg_by_finger = {"left": {}, "right": {}}
    ext_split_per_deg_by_finger = {"left": {}, "right": {}}
    ext_split_release_tor_by_finger = {"left": {}, "right": {}}
    flex_per_deg_by_finger = {"left": {}, "right": {}}
    flex_split_per_deg_by_finger = {"left": {}, "right": {}}
    flex_split_release_tor_by_finger = {"left": {}, "right": {}}
    release_tor_on_extflex_by_finger = {"left": {}, "right": {}}
    sway_pair_release_tor_on_extflex_by_finger = {"left": {}, "right": {}}
    sway_pair_release_per_deg_on_extflex_by_finger = {"left": {}, "right": {}}
    thumb_short_per_deg = {}
    thumb_long_per_deg = {}

    for hand, hand_cfg in finger_cfg.items():
        for finger, cfg in hand_cfg.items():
            ext_raw = cfg.get("extensors", [])
            if isinstance(ext_raw, dict):
                short_id = int(ext_raw["short"])
                long_id = int(ext_raw["long"])
                ext_ids = [short_id, long_id]
                thumb_extensor_ids[hand] = {"short": short_id, "long": long_id}
            else:
                ext_ids = [int(mid) for mid in ext_raw]

            flex_ids = [int(mid) for mid in cfg.get("flexors", [])]
            sway_pair = [int(mid) for mid in cfg.get("sway_pair", [])]

            for ext_id in ext_ids:
                extensor_assist_map[hand][ext_id] = {"flexors": list(flex_ids)}
            if len(sway_pair) >= 2:
                # out -> in
                extensor_assist_map[hand][sway_pair[0]] = {"flexors": [sway_pair[1]]}
                assist_duration_category[hand]["sway"].update(sway_pair[:2])

            assist_duration_category[hand]["extensor"].update(ext_ids)
            assist_duration_category[hand]["flexor"].update(flex_ids)

            for mid in ext_ids + flex_ids:
                motor_to_finger[hand][mid] = finger
            if len(sway_pair) >= 2:
                for mid in sway_pair[:2]:
                    motor_to_finger[hand][int(mid)] = finger

            if finger == "thumb":
                if len(sway_pair) >= 2:
                    thumb_sway_pair_for_cross[hand] = [int(sway_pair[0]), int(sway_pair[1])]
                opp_id = cfg.get("opposition")
                if opp_id is not None:
                    thumb_opposition_motor[hand] = int(opp_id)
                    motor_to_finger[hand][int(opp_id)] = finger

            if finger != "thumb" and len(sway_pair) >= 2:
                finger_sway_pair[hand][finger] = list(sway_pair[:2])

            # Value semantics: seconds per degree.
            # Supports:
            # 1) new split format: {"ext": {"abd": x, "add": y}, "flex": {...}}
            # 2) legacy shared format: {"abd": x, "add": y} or scalar
            sway_pair_release_cfg = cfg.get(
                "sway_pair_release_per_deg_on_extflex",
                cfg.get(
                    "sway_pair_release_sec_on_extflex",
                    {"abd": 0.0, "add": 0.0},
                ),
            )
            sway_pair_release_tor_cfg_raw = cfg.get("sway_pair_release_tor_on_extflex")
            if sway_pair_release_tor_cfg_raw is not None:
                sway_pair_release_tor_cfg = _normalize_sway_extflex_release_tor_cfg(
                    sway_pair_release_tor_cfg_raw,
                    default_tor=ASSIST_RELEASE_TOR,
                    allow_numeric_axis=True,
                )
            else:
                sway_pair_release_tor_cfg = _normalize_sway_extflex_release_tor_cfg(
                    cfg.get("release_tor_on_extflex", sway_pair_release_cfg),
                    default_tor=ASSIST_RELEASE_TOR,
                    allow_numeric_axis=False,
                )

            release_tor_role_cfg_raw = cfg.get("release_tor_on_extflex")
            if release_tor_role_cfg_raw is not None:
                release_tor_role_cfg = _normalize_extflex_release_tor_cfg(
                    release_tor_role_cfg_raw,
                    default_tor=ASSIST_RELEASE_TOR,
                )
            else:
                ext_tor_cfg = sway_pair_release_tor_cfg.get(
                    "ext",
                    {"abd": ASSIST_RELEASE_TOR, "add": ASSIST_RELEASE_TOR},
                )
                flex_tor_cfg = sway_pair_release_tor_cfg.get("flex", ext_tor_cfg)
                release_tor_role_cfg = {
                    "ext": max(
                        float(ext_tor_cfg.get("abd", ASSIST_RELEASE_TOR)),
                        float(ext_tor_cfg.get("add", ASSIST_RELEASE_TOR)),
                    ),
                    "flex": max(
                        float(flex_tor_cfg.get("abd", ASSIST_RELEASE_TOR)),
                        float(flex_tor_cfg.get("add", ASSIST_RELEASE_TOR)),
                    ),
                }
            ext_release_cfg = cfg.get("ext_release_per_deg", 0.0015)
            ext_split_cfg = _normalize_muscle_split_cfg(
                ext_release_cfg,
                default_per_deg=0.0015,
                default_tor=float(release_tor_role_cfg.get("ext", ASSIST_RELEASE_TOR)),
            )
            ext_per_deg_by_finger[hand][finger] = max(
                float(ext_split_cfg["deep"]["per_deg"]),
                float(ext_split_cfg["superficial"]["per_deg"]),
            )
            ext_split_per_deg_by_finger[hand][finger] = {
                "deep": float(ext_split_cfg["deep"]["per_deg"]),
                "superficial": float(ext_split_cfg["superficial"]["per_deg"]),
            }
            ext_split_release_tor_by_finger[hand][finger] = {
                "deep": float(ext_split_cfg["deep"]["release_tor"]),
                "superficial": float(ext_split_cfg["superficial"]["release_tor"]),
            }
            # Role mapping for fingers that have both deep/superficial flexors.
            # Convention: flexors[0]=deep, flexors[1]=superficial.
            if len(flex_ids) >= 2:
                for ext_id in ext_ids:
                    extensor_split_flexor_targets[hand][int(ext_id)] = {
                        "deep": int(flex_ids[0]),
                        "superficial": int(flex_ids[1]),
                    }
                flexor_role_by_motor[hand][int(flex_ids[0])] = "deep"
                flexor_role_by_motor[hand][int(flex_ids[1])] = "superficial"
            elif len(flex_ids) == 1:
                flexor_role_by_motor[hand][int(flex_ids[0])] = "deep"

            sway_pair_release_per_deg_on_extflex_by_finger[hand][finger] = _normalize_sway_extflex_cfg(
                sway_pair_release_cfg
            )
            release_tor_on_extflex_by_finger[hand][finger] = dict(release_tor_role_cfg)
            sway_pair_release_tor_on_extflex_by_finger[hand][finger] = _normalize_sway_extflex_release_tor_cfg(
                sway_pair_release_tor_cfg,
                default_tor=ASSIST_RELEASE_TOR,
                allow_numeric_axis=True,
            )

            flex_release_cfg = cfg.get("flex_release_per_deg")
            if finger == "thumb":
                short_v = float(flex_release_cfg["short_ext"])
                long_v = float(flex_release_cfg["long_ext"])
                thumb_short_per_deg[hand] = short_v
                thumb_long_per_deg[hand] = long_v
            else:
                flex_split_cfg = _normalize_muscle_split_cfg(
                    flex_release_cfg,
                    default_per_deg=0.001,
                    default_tor=float(release_tor_role_cfg.get("flex", ASSIST_RELEASE_TOR)),
                )
                flex_per_deg_by_finger[hand][finger] = max(
                    float(flex_split_cfg["deep"]["per_deg"]),
                    float(flex_split_cfg["superficial"]["per_deg"]),
                )
                flex_split_per_deg_by_finger[hand][finger] = {
                    "deep": float(flex_split_cfg["deep"]["per_deg"]),
                    "superficial": float(flex_split_cfg["superficial"]["per_deg"]),
                }
                flex_split_release_tor_by_finger[hand][finger] = {
                    "deep": float(flex_split_cfg["deep"]["release_tor"]),
                    "superficial": float(flex_split_cfg["superficial"]["release_tor"]),
                }

    for hand in ("left", "right"):
        thumb_short_per_deg.setdefault(hand, 0.001)
        thumb_long_per_deg.setdefault(hand, 0.001)
        thumb_opposition_motor.setdefault(hand, None)

    return (
        extensor_assist_map,
        assist_duration_category,
        motor_to_finger,
        finger_sway_pair,
        thumb_extensor_ids,
        thumb_sway_pair_for_cross,
        thumb_opposition_motor,
        ext_per_deg_by_finger,
        ext_split_per_deg_by_finger,
        ext_split_release_tor_by_finger,
        extensor_split_flexor_targets,
        flexor_role_by_motor,
        flex_per_deg_by_finger,
        flex_split_per_deg_by_finger,
        flex_split_release_tor_by_finger,
        release_tor_on_extflex_by_finger,
        sway_pair_release_tor_on_extflex_by_finger,
        sway_pair_release_per_deg_on_extflex_by_finger,
        thumb_short_per_deg,
        thumb_long_per_deg,
    )


# Assist duration categories (by active motor id):
# extensor: |delta_pos| * ASSIST_RELEASE_SEC_EXT_PER_DEG_BY_FINGER[finger]
# flexor:   non-thumb => |delta_pos| * ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER[finger]
#           thumb => split to short/long extensor with dedicated params
# sway(abd/add): fixed ASSIST_RELEASE_SEC_SWAY_FIXED
(
    EXTENSOR_ASSIST_MAP,
    ASSIST_DURATION_CATEGORY,
    MOTOR_TO_FINGER,
    FINGER_SWAY_PAIR,
    THUMB_EXTENSOR_IDS,
    THUMB_SWAY_PAIR_FOR_OPPOSITION_CROSS,
    THUMB_OPPOSITION_MOTOR_ID,
    ASSIST_RELEASE_SEC_EXT_PER_DEG_BY_FINGER,
    ASSIST_RELEASE_SEC_EXT_SPLIT_PER_DEG_BY_FINGER,
    ASSIST_RELEASE_TOR_EXT_SPLIT_BY_FINGER,
    EXTENSOR_SPLIT_FLEXOR_TARGETS,
    FLEXOR_ROLE_BY_MOTOR,
    ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER,
    ASSIST_RELEASE_SEC_FLEX_SPLIT_PER_DEG_BY_FINGER,
    ASSIST_RELEASE_TOR_FLEX_SPLIT_BY_FINGER,
    ASSIST_RELEASE_TOR_ON_EXTFLEX_BY_FINGER,
    ASSIST_RELEASE_TOR_SWAY_PAIR_ON_EXTFLEX_BY_FINGER,
    ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER,
    ASSIST_RELEASE_SEC_FLEX_THUMB_SHORT_EXT_PER_DEG,
    ASSIST_RELEASE_SEC_FLEX_THUMB_LONG_EXT_PER_DEG,
) = _build_assist_derived_maps(FINGER_ASSIST_CONFIG)


@dataclass
class MotorFeedback:
    pos: float | None = None
    vel: float | None = None
    tor: float | None = None
    kp: float | None = None
    kd: float | None = None


@dataclass
class MotorSnapshot:
    created_ts: float
    source: str
    feedback_rows: list[MotorFeedback]


def _to_float_or_none(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _format_wall_time(ts: float | None) -> str:
    ts_v = _to_float_or_none(ts)
    if ts_v is None or ts_v <= 0.0:
        return "unknown"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts_v)))
    except Exception:
        return "unknown"


def _serialize_feedback_row(fb: MotorFeedback):
    return {
        "pos": _to_float_or_none(fb.pos),
        "vel": _to_float_or_none(fb.vel),
        "tor": _to_float_or_none(fb.tor),
        "kp": _to_float_or_none(fb.kp),
        "kd": _to_float_or_none(fb.kd),
    }


def _deserialize_feedback_row(data):
    if not isinstance(data, dict):
        return MotorFeedback()
    return MotorFeedback(
        pos=_to_float_or_none(data.get("pos")),
        vel=_to_float_or_none(data.get("vel")),
        tor=_to_float_or_none(data.get("tor")),
        kp=_to_float_or_none(data.get("kp")),
        kd=_to_float_or_none(data.get("kd")),
    )


def _serialize_snapshot(snapshot: MotorSnapshot):
    return {
        "created_ts": float(_to_float_or_none(snapshot.created_ts) or time.time()),
        "source": str(snapshot.source),
        "feedback_rows": [_serialize_feedback_row(fb) for fb in snapshot.feedback_rows],
    }


def _deserialize_snapshot(data):
    if not isinstance(data, dict):
        return None
    created_ts = _to_float_or_none(data.get("created_ts"))
    if created_ts is None:
        created_ts = time.time()
    source = str(data.get("source", "unknown"))
    rows_raw = data.get("feedback_rows", [])
    if not isinstance(rows_raw, list):
        rows_raw = []
    rows = [_deserialize_feedback_row(row) for row in rows_raw]
    return MotorSnapshot(
        created_ts=float(created_ts),
        source=source,
        feedback_rows=rows,
    )


def _is_manual_snapshot_source(source: str) -> bool:
    return str(source).strip().lower().startswith("manual_h")


def _load_snapshot_history(snapshot_path: Path):
    if not snapshot_path.exists():
        return []
    try:
        raw = snapshot_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"警告: 快照文件读取失败 {snapshot_path}: {exc}")
        return []
    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(f"警告: 快照文件JSON解析失败 {snapshot_path}: {exc}")
        return []

    if isinstance(payload, dict):
        snapshots_raw = payload.get("snapshots", [])
    elif isinstance(payload, list):
        snapshots_raw = payload
    else:
        print(f"警告: 快照文件格式不支持 {snapshot_path}，将忽略旧内容。")
        return []
    if not isinstance(snapshots_raw, list):
        print(f"警告: 快照文件中的 snapshots 字段不是列表 {snapshot_path}。")
        return []

    history = []
    for item in snapshots_raw:
        snapshot = _deserialize_snapshot(item)
        if snapshot is not None:
            history.append(snapshot)
    manual_history = [snapshot for snapshot in history if _is_manual_snapshot_source(snapshot.source)]
    ignored_count = len(history) - len(manual_history)
    if ignored_count > 0:
        print(f"提示: 已忽略 {ignored_count} 条非手动快照记录，仅保留按 h 记录的快照。")
    return manual_history


def _save_snapshot_history(snapshot_path: Path, snapshot_history: list[MotorSnapshot]):
    payload = {
        "schema": SNAPSHOT_FILE_SCHEMA,
        "motor_mapping_version": MAPPING_VERSION,
        "snapshots": [_serialize_snapshot(snapshot) for snapshot in snapshot_history],
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_tor_safety_pairs(hand: str):
    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    pairs = []
    seen = set()
    for source_id, info in hand_map.items():
        sid = int(source_id)
        for target_id in info.get("flexors", []):
            tid = int(target_id)
            if not (1 <= sid <= MOTOR_COUNT and 1 <= tid <= MOTOR_COUNT):
                continue
            key = (min(sid, tid), max(sid, tid))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((sid, tid))
    return pairs


def _validate_tor_pair_safety(hand: str, tor_values: list[float], threshold_a: float = TOR_PAIR_MAX_DIFF_A):
    violations = []
    for a_id, b_id in _build_tor_safety_pairs(hand):
        a_tor = float(tor_values[a_id - 1])
        b_tor = float(tor_values[b_id - 1])
        diff = abs(a_tor - b_tor)
        if diff > float(threshold_a):
            violations.append(
                {
                    "a_id": int(a_id),
                    "b_id": int(b_id),
                    "a_tor": float(a_tor),
                    "b_tor": float(b_tor),
                    "diff": float(diff),
                }
            )
    return violations


def _print_tor_safety_violations(hand: str, violations: list[dict], threshold_a: float = TOR_PAIR_MAX_DIFF_A):
    print(
        f"安全保护触发: hand={hand}, 发现对侧肌肉默认 tor 差值超过 {float(threshold_a):.3f}A，"
        "禁止执行任何动作并将全部电流置0。"
    )
    for item in violations:
        print(
            "  "
            f"m{int(item['a_id']):02d}(tor={float(item['a_tor']):.3f}) <-> "
            f"m{int(item['b_id']):02d}(tor={float(item['b_tor']):.3f}), "
            f"diff={float(item['diff']):.3f}A"
        )


def _topics_for_hand(hand: str):
    if hand == "left":
        return "/yhand_left_cmd", "/yhand_left_feedback"
    return "/yhand_right_cmd", "/yhand_right_feedback"


def _ensure_defaults_shape(defaults_map, hand: str):
    if hand not in defaults_map:
        raise RuntimeError(f"hand={hand} not found in defaults file")
    hand_map = defaults_map[hand]
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.01)):
        arr = list(hand_map.get(key, []))
        if len(arr) < MOTOR_COUNT:
            arr += [fallback] * (MOTOR_COUNT - len(arr))
        elif len(arr) > MOTOR_COUNT:
            arr = arr[:MOTOR_COUNT]
        hand_map[key] = [fallback if _to_float_or_none(v) is None else float(v) for v in arr]

    tor_active = list(hand_map.get("tor_active", hand_map["tor"]))
    if len(tor_active) < MOTOR_COUNT:
        for i in range(len(tor_active), MOTOR_COUNT):
            tor_active.append(hand_map["tor"][i])
    elif len(tor_active) > MOTOR_COUNT:
        tor_active = tor_active[:MOTOR_COUNT]
    hand_map["tor_active"] = [
        hand_map["tor"][i] if _to_float_or_none(v) is None else float(v)
        for i, v in enumerate(tor_active)
    ]


def _load_defaults_from_py(defaults_path: Path):
    spec = importlib.util.spec_from_file_location("muscle_adjustment_defaults_dynamic", str(defaults_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load defaults file: {defaults_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if not isinstance(defaults, dict):
        raise RuntimeError(f"DEFAULTS not found in {defaults_path}")
    defaults = copy.deepcopy(defaults)
    _ensure_defaults_shape(defaults, "left")
    _ensure_defaults_shape(defaults, "right")
    return defaults


class MotorPositionTestNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        defaults_hand,
        assist_mode: str = ASSIST_MODE_RELEASE,
        motor_feedback_csv_path: str = "auto",
    ):
        super().__init__(f"motor_position_test_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError("Cannot import devices_pkg.msg. Please source ROS2 workspace first.")

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.defaults_hand = defaults_hand
        self.assist_mode = assist_mode

        self._lock = threading.Lock()
        self.latest_feedback = [MotorFeedback() for _ in range(MOTOR_COUNT)]
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.feedback_stale = False
        self.watchdog_fallback_enabled = True
        self.active_motor_id = None
        self.command_matrix = self._build_idle_tension_matrix()
        self.motor_feedback_csv_path_template = str(motor_feedback_csv_path)
        self.motor_feedback_csv_path = ""
        self.motor_feedback_frame_idx = 0
        self.motor_feedback_record_ids = list(MOTOR_FEEDBACK_RECORD_IDS_BY_HAND[self.hand])
        self._motor_feedback_fp = None
        self._motor_feedback_writer = None
        self._motor_feedback_has_opened = False

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.subscription = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.publish_timer = self.create_timer(CONTROL_PERIOD_SEC, self._publish_cb)
        self.watchdog_timer = self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(
            f"[motor_position_test] started. hand={self.hand}, cmd_topic={self.cmd_topic}, "
            f"feedback_topic={self.feedback_topic}, assist_mode={self.assist_mode}, "
            f"motor_feedback_csv_path_template={self.motor_feedback_csv_path_template}"
        )

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
        path_obj = Path(rendered).expanduser()
        if not path_obj.is_absolute():
            path_obj = (Path.cwd() / path_obj).resolve()
        if str(rendered).endswith("/") or path_obj.suffix == "":
            path_obj = path_obj / default_name
        return str(path_obj)

    def _open_motor_feedback_csv_locked(self):
        if self._motor_feedback_writer is not None:
            return
        if not self.motor_feedback_csv_path:
            self.motor_feedback_csv_path = self._resolve_motor_feedback_csv_path(
                self.motor_feedback_csv_path_template
            )
        path_obj = Path(self.motor_feedback_csv_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._motor_feedback_has_opened else "w"
        self._motor_feedback_fp = open(str(path_obj), mode, newline="", encoding="utf-8", buffering=1)
        self._motor_feedback_writer = csv.writer(self._motor_feedback_fp)
        if not self._motor_feedback_has_opened:
            header = ["frame_idx", "wall_time_sec", "mono_time_sec"]
            header.extend([f"motor_{int(mid)}_pos" for mid in self.motor_feedback_record_ids])
            self._motor_feedback_writer.writerow(header)
            self._motor_feedback_has_opened = True

    def record_motor_feedback_once(self, reason: str = "manual_r"):
        with self._lock:
            self._open_motor_feedback_csv_locked()
            if not self.feedback_received:
                path = str(self.motor_feedback_csv_path)
                self.get_logger().warn(
                    f"[motor-feedback-record] no feedback yet, skip single-frame record. path={path}"
                )
                return
            path = str(self.motor_feedback_csv_path)
            frame_idx = int(self.motor_feedback_frame_idx)
            self._write_motor_feedback_row_locked()
            self._close_motor_feedback_csv_locked()
        self.get_logger().info(
            f"[motor-feedback-record] wrote one frame reason={reason}, path={path}, frame_idx={frame_idx}"
        )

    def _write_motor_feedback_row_locked(self):
        if self._motor_feedback_writer is None:
            return
        row = [
            str(int(self.motor_feedback_frame_idx)),
            f"{time.time():.6f}",
            f"{time.monotonic():.6f}",
        ]
        for motor_id in self.motor_feedback_record_ids:
            fb = self.latest_feedback[int(motor_id) - 1]
            row.append("" if fb.pos is None else f"{float(fb.pos):.6f}")
        self._motor_feedback_writer.writerow(row)
        self.motor_feedback_frame_idx += 1

    def _close_motor_feedback_csv_locked(self):
        if self._motor_feedback_fp is None:
            return
        try:
            self._motor_feedback_fp.flush()
            self._motor_feedback_fp.close()
        finally:
            self._motor_feedback_fp = None
            self._motor_feedback_writer = None

    def close_motor_feedback_csv(self):
        with self._lock:
            self._close_motor_feedback_csv_locked()

    def _build_idle_tension_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append([0.0, 0.0, float(self.defaults_hand["tor"][i]), 0.0, 0.0])
        return matrix

    def _build_all_zero_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    0.0,
                    float(self.defaults_hand["tor"][i]),
                    float(self.defaults_hand["kp"][i]),
                    float(self.defaults_hand["kd"][i]),
                ]
            )
        return matrix

    def _feedback_cb(self, msg):
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        now = time.monotonic()
        with self._lock:
            for i in range(MOTOR_COUNT):
                if i < len(motors):
                    m = motors[i]
                    self.latest_feedback[i] = MotorFeedback(
                        pos=_to_float_or_none(m.pos),
                        vel=_to_float_or_none(m.vel),
                        tor=_to_float_or_none(m.tor),
                        kp=_to_float_or_none(m.kp),
                        kd=_to_float_or_none(m.kd),
                    )
                else:
                    self.latest_feedback[i] = MotorFeedback()
            self.feedback_received = True
            self.last_feedback_ts = now
            self.feedback_stale = False

    def _publish_cb(self):
        with self._lock:
            matrix = [row[:] for row in self.command_matrix]

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

    def _watchdog_cb(self):
        with self._lock:
            if not self.watchdog_fallback_enabled:
                return
        with self._lock:
            if not self.feedback_received:
                return
            age = time.monotonic() - self.last_feedback_ts
        if age > FEEDBACK_STALE_SEC:
            with self._lock:
                self.feedback_stale = True
                self.command_matrix = self._build_idle_tension_matrix()
                self.active_motor_id = None

    def wait_feedback(self, timeout_sec: float):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            with self._lock:
                if self.feedback_received:
                    return True
            time.sleep(0.05)
        return False

    def is_feedback_stale(self):
        with self._lock:
            if not self.feedback_received:
                return True
            return (time.monotonic() - self.last_feedback_ts) > FEEDBACK_STALE_SEC

    def check_topic_safety(self, timeout_sec: float = 10.0):
        conflict_start_ts = None
        latest_pub_count = 0
        latest_pub_nodes_text = "<unknown>"
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            now = time.monotonic()
            pub_count = self.count_publishers(self.cmd_topic)
            cmd_sub_count = self.count_subscribers(self.cmd_topic)
            fb_pub_count = self.count_publishers(self.feedback_topic)
            latest_pub_count = pub_count

            if pub_count > 1:
                pub_infos = self.get_publishers_info_by_topic(self.cmd_topic)
                pub_nodes = []
                for info in pub_infos:
                    ns = info.node_namespace or "/"
                    name = info.node_name or "<unknown>"
                    pub_nodes.append(f"{ns}{name}")
                latest_pub_nodes_text = ", ".join(sorted(set(pub_nodes))) if pub_nodes else "<unknown>"
                if conflict_start_ts is None:
                    conflict_start_ts = now
                if (now - conflict_start_ts) >= TOPIC_CONFLICT_CONFIRM_SEC:
                    return False, (
                        f"Command topic conflict on {self.cmd_topic}. "
                        f"publishers={pub_count}, publisher_nodes=[{latest_pub_nodes_text}]"
                    )
            else:
                conflict_start_ts = None

            if cmd_sub_count >= 1 and fb_pub_count >= 1:
                return True, ""
            time.sleep(0.1)

        if latest_pub_count > 1:
            return False, (
                f"Command topic conflict on {self.cmd_topic}. "
                f"publishers={latest_pub_count}, publisher_nodes=[{latest_pub_nodes_text}]"
            )
        return False, (
            f"Driver topic not ready. cmd_subscribers={self.count_subscribers(self.cmd_topic)}, "
            f"feedback_publishers={self.count_publishers(self.feedback_topic)}."
        )

    def get_motor_feedback(self, motor_id: int):
        idx = motor_id - 1
        with self._lock:
            fb = self.latest_feedback[idx]
            return MotorFeedback(pos=fb.pos, vel=fb.vel, tor=fb.tor, kp=fb.kp, kd=fb.kd)

    def all_motor_positions(self):
        with self._lock:
            return [fb.pos for fb in self.latest_feedback]

    def all_motor_feedback(self):
        with self._lock:
            return [
                MotorFeedback(pos=fb.pos, vel=fb.vel, tor=fb.tor, kp=fb.kp, kd=fb.kd)
                for fb in self.latest_feedback
            ]

    def set_idle_tension(self):
        with self._lock:
            self.command_matrix = self._build_idle_tension_matrix()
            self.active_motor_id = None

    def set_all_zero_output(self):
        with self._lock:
            self.command_matrix = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(MOTOR_COUNT)]
            self.active_motor_id = None

    def set_idle_tension_with_overrides(self, overrides: dict[int, list]):
        with self._lock:
            matrix = self._build_idle_tension_matrix()
            for override_motor_id, row in overrides.items():
                if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                    continue
                ridx = override_motor_id - 1
                matrix[ridx] = [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                ]
            self.command_matrix = matrix
            self.active_motor_id = None

    def set_watchdog_fallback_enabled(self, enabled: bool):
        with self._lock:
            self.watchdog_fallback_enabled = bool(enabled)

    def set_single_motor_position_target(self, motor_id: int, target_pos: float):
        idx = motor_id - 1
        with self._lock:
            matrix = self._build_idle_tension_matrix()
            row = [
                float(target_pos),
                float(self.defaults_hand["vel"][idx]),
                float(self.defaults_hand["tor_active"][idx]),
                float(self.defaults_hand["kp"][idx]),
                float(self.defaults_hand["kd"][idx]),
            ]
            matrix[idx] = row
            self.command_matrix = matrix
            self.active_motor_id = motor_id

    def set_single_motor_position_target_with_overrides(self, motor_id: int, target_pos: float, overrides: dict[int, list]):
        idx = motor_id - 1
        with self._lock:
            matrix = self._build_idle_tension_matrix()
            matrix[idx] = [
                float(target_pos),
                float(self.defaults_hand["vel"][idx]),
                float(self.defaults_hand["tor_active"][idx]),
                float(self.defaults_hand["kp"][idx]),
                float(self.defaults_hand["kd"][idx]),
            ]
            for override_motor_id, row in overrides.items():
                if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                    continue
                ridx = override_motor_id - 1
                matrix[ridx] = [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                ]
            self.command_matrix = matrix
            self.active_motor_id = motor_id

    def set_all_zero_control(self):
        with self._lock:
            self.command_matrix = self._build_all_zero_matrix()
            self.active_motor_id = None

    def set_all_zero_control_with_overrides(self, overrides: dict[int, list]):
        with self._lock:
            matrix = self._build_all_zero_matrix()
            for override_motor_id, row in overrides.items():
                if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                    continue
                ridx = override_motor_id - 1
                matrix[ridx] = [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                ]
            self.command_matrix = matrix
            self.active_motor_id = None

    def set_all_motor_position_targets(self, target_positions):
        if len(target_positions) != MOTOR_COUNT:
            raise ValueError(f"target_positions length must be {MOTOR_COUNT}, got {len(target_positions)}")
        matrix = []
        for i, target in enumerate(target_positions):
            target_f = _to_float_or_none(target)
            if target_f is None:
                raise ValueError(f"target_positions[{i}] is invalid: {target}")
            matrix.append(
                [
                    float(target_f),
                    float(self.defaults_hand["vel"][i]),
                    float(self.defaults_hand["tor_active"][i]),
                    float(self.defaults_hand["kp"][i]),
                    float(self.defaults_hand["kd"][i]),
                ]
            )
        with self._lock:
            self.command_matrix = matrix
            self.active_motor_id = None


def _print_feedback(prefix: str, motor_id: int, fb: MotorFeedback):
    print(
        f"{prefix} motor={motor_id} pos={fb.pos} vel={fb.vel} tor={fb.tor} "
        f"kp={fb.kp} kd={fb.kd}"
    )


def _assist_mode_label(mode: str) -> str:
    if mode == ASSIST_MODE_RELEASE:
        def _fmt_per_hand_finger(per_hand_map, exclude_thumb: bool = False):
            parts = []
            for hand in ("left", "right"):
                finger_map = per_hand_map.get(hand, {})
                entries = [
                    f"{finger}={float(sec):.5g}"
                    for finger, sec in finger_map.items()
                    if (not exclude_thumb) or (finger != "thumb")
                ]
                parts.append(f"{hand}[{', '.join(entries)}]")
            return "; ".join(parts)

        def _fmt_ext_per_hand_finger(base_map, split_map, exclude_thumb: bool = False):
            parts = []
            for hand in ("left", "right"):
                finger_map = base_map.get(hand, {})
                finger_split_map = split_map.get(hand, {})
                entries = []
                for finger, sec in finger_map.items():
                    if exclude_thumb and finger == "thumb":
                        continue
                    split_cfg = finger_split_map.get(finger)
                    if split_cfg is not None:
                        entries.append(
                            f"{finger}(deep={float(split_cfg.get('deep', sec)):.5g}, "
                            f"superficial={float(split_cfg.get('superficial', sec)):.5g})"
                        )
                    else:
                        entries.append(f"{finger}={float(sec):.5g}")
                parts.append(f"{hand}[{', '.join(entries)}]")
            return "; ".join(parts)

        def _fmt_sway_pair_per_hand(per_hand_map, exclude_thumb: bool = False):
            parts = []
            for hand in ("left", "right"):
                finger_map = per_hand_map.get(hand, {})
                entries = []
                for finger, cfg in finger_map.items():
                    if exclude_thumb and finger == "thumb":
                        continue
                    norm_cfg = _normalize_sway_extflex_cfg(cfg)
                    ext_cfg = norm_cfg.get("ext", {"abd": 0.0, "add": 0.0})
                    flex_cfg = norm_cfg.get("flex", ext_cfg)
                    entries.append(
                        f"{finger}(ext:abd={float(ext_cfg['abd']):.5g},add={float(ext_cfg['add']):.5g}; "
                        f"flex:abd={float(flex_cfg['abd']):.5g},add={float(flex_cfg['add']):.5g})"
                    )
                parts.append(f"{hand}[{', '.join(entries)}]")
            return "; ".join(parts)

        def _fmt_release_tor_per_hand(per_hand_map, exclude_thumb: bool = False):
            parts = []
            for hand in ("left", "right"):
                finger_map = per_hand_map.get(hand, {})
                entries = []
                for finger, cfg in finger_map.items():
                    if exclude_thumb and finger == "thumb":
                        continue
                    tor_cfg = _normalize_extflex_release_tor_cfg(cfg, ASSIST_RELEASE_TOR)
                    entries.append(
                        f"{finger}(ext={float(tor_cfg.get('ext', ASSIST_RELEASE_TOR)):.5g}, "
                        f"flex={float(tor_cfg.get('flex', ASSIST_RELEASE_TOR)):.5g})"
                    )
                parts.append(f"{hand}[{', '.join(entries)}]")
            return "; ".join(parts)

        def _fmt_sway_pair_release_tor_per_hand(per_hand_map, exclude_thumb: bool = False):
            parts = []
            for hand in ("left", "right"):
                finger_map = per_hand_map.get(hand, {})
                entries = []
                for finger, cfg in finger_map.items():
                    if exclude_thumb and finger == "thumb":
                        continue
                    tor_cfg = _normalize_sway_extflex_release_tor_cfg(
                        cfg,
                        default_tor=ASSIST_RELEASE_TOR,
                        allow_numeric_axis=True,
                    )
                    ext_cfg = tor_cfg.get("ext", {"abd": ASSIST_RELEASE_TOR, "add": ASSIST_RELEASE_TOR})
                    flex_cfg = tor_cfg.get("flex", ext_cfg)
                    entries.append(
                        f"{finger}(ext:abd={float(ext_cfg['abd']):.5g},add={float(ext_cfg['add']):.5g}; "
                        f"flex:abd={float(flex_cfg['abd']):.5g},add={float(flex_cfg['add']):.5g})"
                    )
                parts.append(f"{hand}[{', '.join(entries)}]")
            return "; ".join(parts)

        ext_label = _fmt_ext_per_hand_finger(
            ASSIST_RELEASE_SEC_EXT_PER_DEG_BY_FINGER,
            ASSIST_RELEASE_SEC_EXT_SPLIT_PER_DEG_BY_FINGER,
        )
        flex_label = _fmt_per_hand_finger(ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER)
        sway_pair_label = _fmt_sway_pair_per_hand(
            ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER,
            exclude_thumb=True,
        )
        release_tor_label = _fmt_release_tor_per_hand(ASSIST_RELEASE_TOR_ON_EXTFLEX_BY_FINGER)
        sway_pair_release_tor_label = _fmt_sway_pair_release_tor_per_hand(
            ASSIST_RELEASE_TOR_SWAY_PAIR_ON_EXTFLEX_BY_FINGER,
            exclude_thumb=True,
        )
        thumb_short_label = (
            f"left={float(ASSIST_RELEASE_SEC_FLEX_THUMB_SHORT_EXT_PER_DEG.get('left', 0.001)):.5g}, "
            f"right={float(ASSIST_RELEASE_SEC_FLEX_THUMB_SHORT_EXT_PER_DEG.get('right', 0.001)):.5g}"
        )
        thumb_long_label = (
            f"left={float(ASSIST_RELEASE_SEC_FLEX_THUMB_LONG_EXT_PER_DEG.get('left', 0.001)):.5g}, "
            f"right={float(ASSIST_RELEASE_SEC_FLEX_THUMB_LONG_EXT_PER_DEG.get('right', 0.001)):.5g}"
        )
        return (
            "release_tor(对抗肌放线时长: "
            f"伸肌[{ext_label}]s/deg, "
            f"屈肌[{flex_label}, thumb_short_ext[{thumb_short_label}], "
            f"thumb_long_ext[{thumb_long_label}]]s/deg, "
            f"侧摆=delta*per_deg(异常回退{ASSIST_RELEASE_SEC_SWAY_FIXED:.3g}s), "
            f"伸/屈时内收外展追加(四指)[{sway_pair_label}]s/deg, "
            f"伸/屈时内收外展追加release_tor(四指)[{sway_pair_release_tor_label}]A, "
            "delta分段系数="
            f"(阈值={ASSIST_RELEASE_DELTA_SWITCH_DEG:.3g}deg, "
            f"small×{ASSIST_RELEASE_PER_DEG_SCALE_SMALL_DELTA:.3g}, "
            f"large×{ASSIST_RELEASE_PER_DEG_SCALE_LARGE_DELTA:.3g}), "
            "执行顺序=放线与主动位控同步(无延迟), "
            f"release_tor(按手指/主动肌ext|flex)=[{release_tor_label}], "
            f"fallback_tor={ASSIST_RELEASE_TOR:.3g})"
        )
    return "off(关闭)"


def _is_assist_motor(hand: str, motor_id: int) -> bool:
    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    if motor_id in hand_map:
        return True
    for info in hand_map.values():
        if motor_id in info.get("flexors", []):
            return True
    return False


def _get_assist_target_motors(hand: str, motor_id: int, assist_mode: str):
    if assist_mode != ASSIST_MODE_RELEASE:
        return []
    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    if motor_id in hand_map:
        return list(hand_map[motor_id].get("flexors", []))

    ext_targets = []
    for ext_id, info in hand_map.items():
        if motor_id in info.get("flexors", []):
            ext_targets.append(ext_id)
    return sorted(set(ext_targets))


def _estimate_motion_delta_deg(node: MotorPositionTestNode, motor_id: int, target_pos: float) -> float:
    current_pos = None
    fb = node.get_motor_feedback(motor_id)
    if fb.pos is not None and math.isfinite(float(fb.pos)):
        current_pos = float(fb.pos)
    if current_pos is None:
        with node._lock:  # noqa: SLF001
            try:
                current_pos = float(node.command_matrix[motor_id - 1][0])  # noqa: SLF001
            except Exception:
                current_pos = 0.0
    return abs(float(target_pos) - float(current_pos))


def _release_delta_profile(delta_deg: float):
    delta_abs = abs(float(delta_deg))
    if delta_abs >= float(ASSIST_RELEASE_DELTA_SWITCH_DEG):
        return "large", float(ASSIST_RELEASE_PER_DEG_SCALE_LARGE_DELTA)
    return "small", float(ASSIST_RELEASE_PER_DEG_SCALE_SMALL_DELTA)


def _release_sec_from_per_deg(delta_deg: float, per_deg: float) -> float:
    _, scale = _release_delta_profile(delta_deg)
    return max(0.0, abs(float(delta_deg)) * float(per_deg) * float(scale))


def _compute_sway_release_sec_by_delta(hand: str, motor_id: int, delta_deg: float) -> float:
    """Compute sway opposite-side release duration using delta*per_deg policy.

    Source of per_deg:
    - reuse finger-level sway-pair per-degree config already defined for ext/flex:
      ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER[hand][finger]
    - choose per-axis coefficient by active sway direction:
      active out(abd) -> release in(add) uses add per_deg
      active in(add)  -> release out(abd) uses abd per_deg
    - for robustness, use max(ext_role, flex_role) on each axis.
    """
    finger = MOTOR_TO_FINGER.get(hand, {}).get(int(motor_id))
    if finger is None:
        return float(ASSIST_RELEASE_SEC_SWAY_FIXED)

    sway_pair = list(FINGER_SWAY_PAIR.get(hand, {}).get(finger, []))
    if len(sway_pair) < 2:
        return float(ASSIST_RELEASE_SEC_SWAY_FIXED)
    out_id, in_id = int(sway_pair[0]), int(sway_pair[1])

    cfg_raw = ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER.get(hand, {}).get(
        finger,
        ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER.get(hand, {}).get(
            "index",
            {"ext": {"abd": 0.0, "add": 0.0}, "flex": {"abd": 0.0, "add": 0.0}},
        ),
    )
    cfg = _normalize_sway_extflex_cfg(cfg_raw)
    ext_cfg = _normalize_abd_add_cfg(cfg.get("ext"), 0.0, 0.0)
    flex_cfg = _normalize_abd_add_cfg(cfg.get("flex"), 0.0, 0.0)
    abd_per_deg = max(float(ext_cfg["abd"]), float(flex_cfg["abd"]))
    add_per_deg = max(float(ext_cfg["add"]), float(flex_cfg["add"]))

    if int(motor_id) == out_id:
        per_deg = add_per_deg
    elif int(motor_id) == in_id:
        per_deg = abd_per_deg
    else:
        return float(ASSIST_RELEASE_SEC_SWAY_FIXED)

    if per_deg <= 0.0:
        return float(ASSIST_RELEASE_SEC_SWAY_FIXED)
    return _release_sec_from_per_deg(delta_deg, per_deg)


def _compute_assist_release_sec(hand: str, motor_id: int, target_pos: float, delta_deg: float | None = None) -> float:
    if delta_deg is None:
        delta_deg = abs(float(target_pos))
    category = ASSIST_DURATION_CATEGORY.get(hand, {})
    ext_set = category.get("extensor", set())
    flex_set = category.get("flexor", set())
    sway_set = category.get("sway", set())

    if motor_id in sway_set:
        return _compute_sway_release_sec_by_delta(hand, motor_id, delta_deg)
    if motor_id in ext_set:
        finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
        per_deg = ASSIST_RELEASE_SEC_EXT_PER_DEG_BY_FINGER.get(hand, {}).get(
            finger,
            ASSIST_RELEASE_SEC_EXT_PER_DEG_BY_FINGER.get(hand, {}).get("index", 0.0015),
        )
        return _release_sec_from_per_deg(delta_deg, per_deg)
    if motor_id in flex_set:
        finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
        if finger == "thumb":
            thumb_default = max(
                float(ASSIST_RELEASE_SEC_FLEX_THUMB_SHORT_EXT_PER_DEG.get(hand, 0.001)),
                float(ASSIST_RELEASE_SEC_FLEX_THUMB_LONG_EXT_PER_DEG.get(hand, 0.001)),
            )
            return _release_sec_from_per_deg(delta_deg, thumb_default)
        role = FLEXOR_ROLE_BY_MOTOR.get(hand, {}).get(int(motor_id))
        split_cfg = ASSIST_RELEASE_SEC_FLEX_SPLIT_PER_DEG_BY_FINGER.get(hand, {}).get(finger, {})
        if role in ("deep", "superficial"):
            per_deg = float(
                split_cfg.get(
                    role,
                    ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER.get(hand, {}).get("index", 0.001),
                )
            )
        else:
            per_deg = ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER.get(hand, {}).get(
                finger,
                ASSIST_RELEASE_SEC_FLEX_PER_DEG_BY_FINGER.get(hand, {}).get("index", 0.001),
            )
        return _release_sec_from_per_deg(delta_deg, per_deg)
    return float(ASSIST_RELEASE_SEC)


def _compute_primary_release_plan_sec(
    hand: str,
    motor_id: int,
    delta_deg: float,
    primary_target_ids: list[int],
):
    release_plan_sec = {}
    base_release_sec = _compute_assist_release_sec(hand, motor_id, 0.0, delta_deg=delta_deg)
    category = _assist_motor_category(hand, motor_id)
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)

    # Extensor active on fingers with deep/superficial flexors:
    # use split release duration for each antagonist flexor target.
    if category == "extensor":
        split_cfg = ASSIST_RELEASE_SEC_EXT_SPLIT_PER_DEG_BY_FINGER.get(hand, {}).get(finger)
        role_targets = EXTENSOR_SPLIT_FLEXOR_TARGETS.get(hand, {}).get(int(motor_id), {})
        if split_cfg and role_targets:
            deep_id = role_targets.get("deep")
            superficial_id = role_targets.get("superficial")
            deep_per_deg = float(split_cfg.get("deep", 0.0015))
            superficial_per_deg = float(split_cfg.get("superficial", deep_per_deg))
            for tid in primary_target_ids:
                if tid == deep_id:
                    sec = _release_sec_from_per_deg(delta_deg, deep_per_deg)
                elif tid == superficial_id:
                    sec = _release_sec_from_per_deg(delta_deg, superficial_per_deg)
                else:
                    sec = float(base_release_sec)
                release_plan_sec[tid] = max(float(release_plan_sec.get(tid, 0.0)), float(sec))
            return release_plan_sec

    # Thumb flexor drives both thumb extensors; split duration by short/long extensor.
    if category == "flexor" and finger == "thumb":
        thumb_ext = THUMB_EXTENSOR_IDS.get(hand, {})
        short_id = thumb_ext.get("short")
        long_id = thumb_ext.get("long")
        short_per_deg = float(ASSIST_RELEASE_SEC_FLEX_THUMB_SHORT_EXT_PER_DEG.get(hand, 0.001))
        long_per_deg = float(ASSIST_RELEASE_SEC_FLEX_THUMB_LONG_EXT_PER_DEG.get(hand, 0.001))
        for tid in primary_target_ids:
            if tid == short_id:
                sec = _release_sec_from_per_deg(delta_deg, short_per_deg)
            elif tid == long_id:
                sec = _release_sec_from_per_deg(delta_deg, long_per_deg)
            else:
                sec = float(base_release_sec)
            release_plan_sec[tid] = max(float(release_plan_sec.get(tid, 0.0)), float(sec))
        return release_plan_sec

    for tid in primary_target_ids:
        release_plan_sec[tid] = max(float(release_plan_sec.get(tid, 0.0)), float(base_release_sec))
    return release_plan_sec


def _compute_primary_release_plan_tor(
    hand: str,
    motor_id: int,
    primary_target_ids: list[int],
):
    release_tor_plan = {}
    category = _assist_motor_category(hand, motor_id)
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
    default_tor = _get_release_tor_on_extflex(hand, motor_id)

    # Extensor active on fingers with deep/superficial flexors:
    # split target tor by deep/superficial antagonist.
    if category == "extensor":
        split_tor_cfg = ASSIST_RELEASE_TOR_EXT_SPLIT_BY_FINGER.get(hand, {}).get(finger)
        role_targets = EXTENSOR_SPLIT_FLEXOR_TARGETS.get(hand, {}).get(int(motor_id), {})
        if split_tor_cfg and role_targets:
            deep_id = role_targets.get("deep")
            superficial_id = role_targets.get("superficial")
            deep_tor = float(split_tor_cfg.get("deep", default_tor))
            superficial_tor = float(split_tor_cfg.get("superficial", deep_tor))
            for tid in primary_target_ids:
                if tid == deep_id:
                    release_tor_plan[int(tid)] = deep_tor
                elif tid == superficial_id:
                    release_tor_plan[int(tid)] = superficial_tor
                else:
                    release_tor_plan[int(tid)] = float(default_tor)
            return release_tor_plan

    # Flexor active (non-thumb): choose tor by active flexor role deep/superficial.
    if category == "flexor" and finger != "thumb":
        role = FLEXOR_ROLE_BY_MOTOR.get(hand, {}).get(int(motor_id))
        split_tor_cfg = ASSIST_RELEASE_TOR_FLEX_SPLIT_BY_FINGER.get(hand, {}).get(finger, {})
        role_tor = float(split_tor_cfg.get(role, default_tor))
        for tid in primary_target_ids:
            release_tor_plan[int(tid)] = role_tor
        return release_tor_plan

    # Fallback: use one tor for all primary targets.
    for tid in primary_target_ids:
        release_tor_plan[int(tid)] = float(default_tor)
    return release_tor_plan


def _assist_motor_category(hand: str, motor_id: int) -> str:
    category = ASSIST_DURATION_CATEGORY.get(hand, {})
    if motor_id in category.get("extensor", set()):
        return "extensor"
    if motor_id in category.get("flexor", set()):
        return "flexor"
    if motor_id in category.get("sway", set()):
        return "sway"
    return "other"


def _get_release_tor_on_extflex(hand: str, motor_id: int) -> float:
    category = _assist_motor_category(hand, motor_id)
    if category not in ("extensor", "flexor"):
        return float(ASSIST_RELEASE_TOR)
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
    if category == "flexor" and finger != "thumb":
        role = FLEXOR_ROLE_BY_MOTOR.get(hand, {}).get(int(motor_id))
        split_tor_cfg = ASSIST_RELEASE_TOR_FLEX_SPLIT_BY_FINGER.get(hand, {}).get(finger, {})
        if role in ("deep", "superficial"):
            return float(split_tor_cfg.get(role, ASSIST_RELEASE_TOR))
    hand_map = ASSIST_RELEASE_TOR_ON_EXTFLEX_BY_FINGER.get(hand, {})
    cfg = hand_map.get(
        finger,
        hand_map.get("index", {"ext": ASSIST_RELEASE_TOR, "flex": ASSIST_RELEASE_TOR}),
    )
    role_key = "ext" if category == "extensor" else "flex"
    return float(cfg.get(role_key, ASSIST_RELEASE_TOR))


def _get_sway_pair_targets_for_extflex(hand: str, motor_id: int):
    if _assist_motor_category(hand, motor_id) not in ("extensor", "flexor"):
        return []
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
    if finger is None or finger == "thumb":
        return []
    return list(FINGER_SWAY_PAIR.get(hand, {}).get(finger, []))


def _get_thumb_opposition_sway_cross_targets(hand: str, motor_id: int):
    mid = int(motor_id)
    opp_id = THUMB_OPPOSITION_MOTOR_ID.get(hand)
    sway_pair = list(THUMB_SWAY_PAIR_FOR_OPPOSITION_CROSS.get(hand, []))
    if opp_id is None or len(sway_pair) < 2:
        return []
    out_id, in_id = int(sway_pair[0]), int(sway_pair[1])
    if mid == int(opp_id):
        return [out_id, in_id]
    if mid == out_id or mid == in_id:
        return [int(opp_id)]
    return []


def _get_sway_pair_release_plan_on_extflex(
    hand: str,
    motor_id: int,
    sway_pair_target_ids: list[int],
    delta_deg: float,
):
    category = _assist_motor_category(hand, motor_id)
    if category not in ("extensor", "flexor"):
        return {}
    if len(sway_pair_target_ids) < 2:
        return {}
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
    hand_map = ASSIST_RELEASE_SEC_SWAY_PAIR_ON_EXTFLEX_PER_DEG_BY_FINGER.get(hand, {})
    cfg_raw = hand_map.get(
        finger,
        hand_map.get(
            "index",
            {"ext": {"abd": 0.0, "add": 0.0}, "flex": {"abd": 0.0, "add": 0.0}},
        ),
    )
    cfg = _normalize_sway_extflex_cfg(cfg_raw)
    role_key = "ext" if category == "extensor" else "flex"
    role_cfg = _normalize_abd_add_cfg(cfg.get(role_key), 0.0, 0.0)
    abd_per_deg = float(role_cfg["abd"])
    add_per_deg = float(role_cfg["add"])

    out_id, in_id = int(sway_pair_target_ids[0]), int(sway_pair_target_ids[1])
    return {
        out_id: _release_sec_from_per_deg(delta_deg, abd_per_deg),  # abduction (out)
        in_id: _release_sec_from_per_deg(delta_deg, add_per_deg),   # adduction (in)
    }


def _get_sway_pair_release_tor_plan_on_extflex(
    hand: str,
    motor_id: int,
    sway_pair_target_ids: list[int],
    default_tor: float,
):
    category = _assist_motor_category(hand, motor_id)
    if category not in ("extensor", "flexor"):
        return {}
    if len(sway_pair_target_ids) < 2:
        return {}
    finger = MOTOR_TO_FINGER.get(hand, {}).get(motor_id)
    hand_map = ASSIST_RELEASE_TOR_SWAY_PAIR_ON_EXTFLEX_BY_FINGER.get(hand, {})
    cfg_raw = hand_map.get(
        finger,
        hand_map.get(
            "index",
            {
                "ext": {"abd": default_tor, "add": default_tor},
                "flex": {"abd": default_tor, "add": default_tor},
            },
        ),
    )
    cfg = _normalize_sway_extflex_release_tor_cfg(
        cfg_raw,
        default_tor=default_tor,
        allow_numeric_axis=True,
    )
    role_key = "ext" if category == "extensor" else "flex"
    role_cfg = _normalize_abd_add_release_tor_cfg(
        cfg.get(role_key),
        default_abd_tor=default_tor,
        default_add_tor=default_tor,
        allow_numeric_axis=True,
    )

    out_id, in_id = int(sway_pair_target_ids[0]), int(sway_pair_target_ids[1])
    return {
        out_id: float(role_cfg["abd"]),  # abduction (out)
        in_id: float(role_cfg["add"]),   # adduction (in)
    }


def _max_release_sec(release_plan_sec: dict[int, float]) -> float:
    if not release_plan_sec:
        return 0.0
    return max(
        float(sec)
        for sec in release_plan_sec.values()
    )


def _normalized_release_plan_text(release_plan_sec: dict[int, float]):
    return {
        int(mid): float(sec)
        for mid, sec in sorted(release_plan_sec.items())
        if float(sec) > 0.0
    }


def _sway_pair_release_sec_for_log(
    sway_pair_release_plan_sec: dict[int, float],
    sway_pair_target_ids: list[int],
):
    if not sway_pair_target_ids:
        return 0.0
    return _max_release_sec(
        {
            int(mid): float(sway_pair_release_plan_sec.get(mid, 0.0))
            for mid in sway_pair_target_ids
        }
    )


def _apply_antagonist_assist_phase(node: MotorPositionTestNode, motor_id: int, target_pos: float):
    if node.assist_mode != ASSIST_MODE_RELEASE:
        return {"applied": False, "targets": []}

    primary_target_ids = _get_assist_target_motors(node.hand, motor_id, node.assist_mode)
    raw_sway_pair_target_ids = _get_sway_pair_targets_for_extflex(node.hand, motor_id)
    thumb_cross_target_ids = _get_thumb_opposition_sway_cross_targets(node.hand, motor_id)
    if (not primary_target_ids) and (not raw_sway_pair_target_ids) and (not thumb_cross_target_ids):
        return {"applied": False, "targets": []}

    delta_deg = _estimate_motion_delta_deg(node, motor_id, target_pos)
    delta_profile, delta_scale = _release_delta_profile(delta_deg)
    sway_pair_release_plan_sec = _get_sway_pair_release_plan_on_extflex(
        node.hand,
        motor_id,
        sway_pair_target_ids=raw_sway_pair_target_ids,
        delta_deg=delta_deg,
    )
    # Keep only positive-duration sway targets.
    sway_pair_release_plan_sec = {
        int(tid): float(sec)
        for tid, sec in sway_pair_release_plan_sec.items()
        if float(sec) > 0.0
    }
    sway_pair_target_ids = sorted(sway_pair_release_plan_sec.keys())
    primary_plan_sec = _compute_primary_release_plan_sec(
        node.hand,
        motor_id,
        delta_deg=delta_deg,
        primary_target_ids=primary_target_ids,
    )
    primary_release_sec = (
        _max_release_sec(primary_plan_sec)
        if primary_plan_sec
        else _compute_assist_release_sec(node.hand, motor_id, target_pos, delta_deg=delta_deg)
    )
    sway_pair_release_sec = _sway_pair_release_sec_for_log(
        sway_pair_release_plan_sec,
        sway_pair_target_ids,
    )
    thumb_cross_release_base_sec = float(ASSIST_RELEASE_SEC_THUMB_OPPOSITION_SWAY)
    thumb_cross_plan_sec = {
        int(tid): float(thumb_cross_release_base_sec)
        for tid in thumb_cross_target_ids
        if float(thumb_cross_release_base_sec) > 0.0
    }
    primary_start_delay_sec = 0.0

    release_start_sec = {}
    release_end_sec = {}

    def _merge_release_window(target_id: int, start_sec: float, duration_sec: float):
        duration_v = float(duration_sec)
        if duration_v <= 0.0:
            return
        start_v = max(0.0, float(start_sec))
        end_v = start_v + duration_v
        if target_id in release_end_sec:
            release_start_sec[target_id] = min(float(release_start_sec[target_id]), start_v)
            release_end_sec[target_id] = max(float(release_end_sec[target_id]), end_v)
            return
        release_start_sec[target_id] = start_v
        release_end_sec[target_id] = end_v

    for tid, sec in primary_plan_sec.items():
        _merge_release_window(int(tid), primary_start_delay_sec, float(sec))
    for tid, sec in sway_pair_release_plan_sec.items():
        _merge_release_window(int(tid), 0.0, float(sec))
    for tid, sec in thumb_cross_plan_sec.items():
        _merge_release_window(int(tid), 0.0, float(sec))

    if not release_end_sec:
        return {"applied": False, "targets": []}

    default_release_tor = _get_release_tor_on_extflex(node.hand, motor_id)
    sway_pair_release_plan_tor = _get_sway_pair_release_tor_plan_on_extflex(
        node.hand,
        motor_id,
        sway_pair_target_ids=raw_sway_pair_target_ids,
        default_tor=default_release_tor,
    )
    sway_pair_release_plan_tor = {
        int(tid): float(sway_pair_release_plan_tor.get(int(tid), default_release_tor))
        for tid in sway_pair_release_plan_sec.keys()
    }
    primary_plan_tor = _compute_primary_release_plan_tor(
        node.hand,
        motor_id,
        primary_target_ids=primary_target_ids,
    )
    thumb_cross_plan_tor = {
        int(tid): float(ASSIST_RELEASE_TOR_THUMB_OPPOSITION_SWAY)
        for tid in thumb_cross_plan_sec.keys()
    }
    release_tor_plan = {}

    def _merge_release_tor(target_id: int, tor_value: float):
        tid = int(target_id)
        tor_v = float(tor_value)
        if tid in release_tor_plan:
            release_tor_plan[tid] = max(float(release_tor_plan[tid]), tor_v)
            return
        release_tor_plan[tid] = tor_v

    for tid in primary_plan_sec.keys():
        _merge_release_tor(int(tid), primary_plan_tor.get(int(tid), default_release_tor))
    for tid in sway_pair_release_plan_sec.keys():
        _merge_release_tor(int(tid), sway_pair_release_plan_tor.get(int(tid), default_release_tor))
    for tid in thumb_cross_plan_sec.keys():
        _merge_release_tor(int(tid), thumb_cross_plan_tor.get(int(tid), default_release_tor))
    for tid in release_end_sec.keys():
        _merge_release_tor(int(tid), release_tor_plan.get(int(tid), default_release_tor))

    release_tor = max(float(v) for v in release_tor_plan.values())
    release_tor_plan_log = {
        int(mid): float(tor)
        for mid, tor in sorted(release_tor_plan.items())
    }
    max_release_sec = max(float(v) for v in release_end_sec.values())
    # Synchronous mode: antagonist release and active motor position control run together.
    node.set_single_motor_position_target(motor_id, target_pos)
    start = time.monotonic()
    last_active_ids = None
    while (time.monotonic() - start) < max_release_sec:
        if node.is_feedback_stale():
            elapsed_now = time.monotonic() - start
            active_ids_now = [
                tid
                for tid, end_sec in release_end_sec.items()
                if float(release_start_sec.get(tid, 0.0)) <= float(elapsed_now) < float(end_sec)
            ]
            return {
                "applied": True,
                "targets": sorted(release_end_sec.keys()),
                "primary_targets": sorted(set(primary_target_ids)),
                "sway_pair_targets": sorted(set(sway_pair_target_ids)),
                "thumb_cross_targets": sorted(set(thumb_cross_plan_sec.keys())),
                "feedback_stale": True,
                "release_sec": float(max_release_sec),
                "primary_release_sec": float(primary_release_sec),
                "primary_start_delay_sec": float(primary_start_delay_sec),
                "sway_pair_release_sec": float(sway_pair_release_sec),
                "sway_pair_release_plan_sec": _normalized_release_plan_text(sway_pair_release_plan_sec),
                "thumb_cross_release_plan_sec": _normalized_release_plan_text(thumb_cross_plan_sec),
                "active_targets_now": sorted(active_ids_now),
                "delta_deg": float(delta_deg),
                "delta_release_profile": str(delta_profile),
                "delta_release_scale": float(delta_scale),
                "release_tor": float(release_tor),
                "release_tor_plan": dict(release_tor_plan_log),
                "thumb_cross_release_tor": float(ASSIST_RELEASE_TOR_THUMB_OPPOSITION_SWAY),
            }

        elapsed = time.monotonic() - start
        active_ids = sorted(
            [
                tid
                for tid, end_sec in release_end_sec.items()
                if float(release_start_sec.get(tid, 0.0)) <= float(elapsed) < float(end_sec)
            ]
        )
        if active_ids != last_active_ids:
            if active_ids:
                overrides = {
                    tid: [0.0, 0.0, float(release_tor_plan.get(tid, default_release_tor)), 0.0, 0.0]
                    for tid in active_ids
                }
                node.set_single_motor_position_target_with_overrides(motor_id, target_pos, overrides)
            else:
                node.set_single_motor_position_target(motor_id, target_pos)
            last_active_ids = active_ids
        time.sleep(CONTROL_PERIOD_SEC)

    node.set_single_motor_position_target(motor_id, target_pos)
    return {
        "applied": True,
        "targets": sorted(release_end_sec.keys()),
        "primary_targets": sorted(set(primary_target_ids)),
        "sway_pair_targets": sorted(set(sway_pair_target_ids)),
        "thumb_cross_targets": sorted(set(thumb_cross_plan_sec.keys())),
        "feedback_stale": False,
        "release_sec": float(max_release_sec),
        "primary_release_sec": float(primary_release_sec),
        "primary_start_delay_sec": float(primary_start_delay_sec),
        "sway_pair_release_sec": float(sway_pair_release_sec),
        "sway_pair_release_plan_sec": _normalized_release_plan_text(sway_pair_release_plan_sec),
        "thumb_cross_release_plan_sec": _normalized_release_plan_text(thumb_cross_plan_sec),
        "delta_deg": float(delta_deg),
        "delta_release_profile": str(delta_profile),
        "delta_release_scale": float(delta_scale),
        "release_tor": float(release_tor),
        "release_tor_plan": dict(release_tor_plan_log),
        "thumb_cross_release_tor": float(ASSIST_RELEASE_TOR_THUMB_OPPOSITION_SWAY),
    }


def _prompt_select_motor():
    while True:
        raw = input(
            "选择电机编号(1-28)，输入 h 记录快照，replay 按索引回放快照，"
            "r 记录一条状态CSV，z 全部回0，x 退出: "
        ).strip().lower()
        if raw in ("x", "quit", "exit"):
            return "quit", None
        if raw in ("h", "hold", "snapshot"):
            return "snapshot", None
        if raw in ("replay", "restore"):
            return "replay", None
        if raw in ("r", "record"):
            return "record_once", None
        if raw in ("z", "zero"):
            return "zero", None
        try:
            mid = int(raw)
        except ValueError:
            print("输入无效，请输入 1-28 / h / replay / r / z / x")
            continue
        if 1 <= mid <= MOTOR_COUNT:
            return "motor", mid
        print("电机编号超范围，请输入 1-28")


def _prompt_target_position():
    while True:
        raw = input(
            "输入目标位置pos（m 切换电机 / h 记录快照 / replay 按索引回放 / "
            "r 记录一条状态CSV / z 全部回0 / x 退出）: "
        ).strip().lower()
        if raw in ("x", "quit", "exit"):
            return "quit", None
        if raw in ("m", "motor"):
            return "change_motor", None
        if raw in ("h", "hold", "snapshot"):
            return "snapshot", None
        if raw in ("replay", "restore"):
            return "replay", None
        if raw in ("r", "record"):
            return "record_once", None
        if raw in ("z", "zero"):
            return "zero", None
        try:
            pos = float(raw)
        except ValueError:
            print("输入无效，请输入数字位置或 m / h / replay / r / z / x")
            continue
        return "target", pos


def _parse_bool_text(text: str, default: bool = True) -> bool:
    if text is None:
        return default
    value = str(text).strip().lower()
    if value in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return True
    if value in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return False
    return default


def _capture_motor_snapshot(node: MotorPositionTestNode, source: str) -> MotorSnapshot:
    return MotorSnapshot(
        created_ts=time.time(),
        source=str(source),
        feedback_rows=node.all_motor_feedback(),
    )


def _append_snapshot_and_report(
    snapshot_history: list[MotorSnapshot],
    snapshot_path: Path,
    node: MotorPositionTestNode,
    source: str,
):
    snapshot = _capture_motor_snapshot(node, source=source)
    snapshot_history.append(snapshot)
    saved_ok = True
    save_error = ""
    try:
        _save_snapshot_history(snapshot_path, snapshot_history)
    except Exception as exc:
        saved_ok = False
        save_error = str(exc)

    snapshot_idx = len(snapshot_history)
    _, snapshot_invalid_ids = _snapshot_target_positions(snapshot)
    created_text = _format_wall_time(snapshot.created_ts)
    saved_text = f"已落盘: {snapshot_path}" if saved_ok else f"落盘失败: {save_error}"
    if snapshot_invalid_ids:
        print(
            f"已记录快照#{snapshot_idx} (time={created_text}, source={snapshot.source})，"
            f"但以下电机pos无效，暂不可回放: {snapshot_invalid_ids}。{saved_text}"
        )
    else:
        print(
            f"已记录快照#{snapshot_idx} (time={created_text}, source={snapshot.source})，"
            f"可输入 replay 并按索引执行28电机同步回放。{saved_text}"
        )
    return snapshot


def _print_snapshot_catalog(snapshot_history: list[MotorSnapshot], max_lines: int = 20):
    if not snapshot_history:
        print("暂无可回放快照：请先按 h 记录，或先执行一次角度指令。")
        return
    total = len(snapshot_history)
    start_idx = max(1, total - int(max_lines) + 1)
    print(f"可回放快照索引总数: {total}，以下显示 #{start_idx}..#{total}:")
    for idx in range(start_idx, total + 1):
        snapshot = snapshot_history[idx - 1]
        _, invalid_ids = _snapshot_target_positions(snapshot)
        status = "可回放" if not invalid_ids else f"不可回放(invalid={invalid_ids})"
        print(
            f"  #{idx}: time={_format_wall_time(snapshot.created_ts)}, "
            f"source={snapshot.source}, {status}"
        )


def _prompt_replay_index(snapshot_history: list[MotorSnapshot]):
    if not snapshot_history:
        print("暂无可回放快照：请先按 h 记录，或先执行一次角度指令。")
        return None
    _print_snapshot_catalog(snapshot_history)
    total = len(snapshot_history)
    while True:
        raw = input("输入回放索引号（回车=最新，c=取消）: ").strip().lower()
        if raw in ("c", "cancel", "q", "quit", "x"):
            return None
        if raw == "":
            return total
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入整数索引号，或回车/c。")
            continue
        if 1 <= idx <= total:
            return idx
        print(f"索引超范围，请输入 1..{total}。")


def _snapshot_target_positions(snapshot: MotorSnapshot):
    rows = list(snapshot.feedback_rows)
    if len(rows) < MOTOR_COUNT:
        rows += [MotorFeedback() for _ in range(MOTOR_COUNT - len(rows))]
    elif len(rows) > MOTOR_COUNT:
        rows = rows[:MOTOR_COUNT]

    target_positions = []
    invalid_ids = []
    for i, fb in enumerate(rows, start=1):
        pos = _to_float_or_none(fb.pos)
        target_positions.append(pos)
        if pos is None:
            invalid_ids.append(i)
    return target_positions, invalid_ids


def _evaluate_replay_reached(node: MotorPositionTestNode, target_positions):
    pos_list = node.all_motor_positions()
    missing_ids = []
    failed_ids = []
    max_err = 0.0

    for i, target in enumerate(target_positions, start=1):
        if target is None:
            missing_ids.append(i)
            failed_ids.append(i)
            continue
        cur = pos_list[i - 1] if (i - 1) < len(pos_list) else None
        if cur is None:
            missing_ids.append(i)
            failed_ids.append(i)
            continue
        err = abs(float(cur) - float(target))
        if err > max_err:
            max_err = err
        if err > ARRIVAL_POS_TOL:
            failed_ids.append(i)

    reached = (not missing_ids) and (not failed_ids)
    return reached, failed_ids, missing_ids, max_err


def _replay_failure_details(node: MotorPositionTestNode, target_positions):
    pos_list = node.all_motor_positions()
    failed_ids = []
    pos_fmt = []
    for i, target in enumerate(target_positions, start=1):
        cur = pos_list[i - 1] if (i - 1) < len(pos_list) else None
        if target is None:
            failed_ids.append(i)
        elif cur is None or abs(float(cur) - float(target)) > ARRIVAL_POS_TOL:
            failed_ids.append(i)
        cur_txt = "None" if cur is None else f"{float(cur):.3f}"
        tgt_txt = "None" if target is None else f"{float(target):.3f}"
        pos_fmt.append(f"m{i}:{cur_txt}/{tgt_txt}")
    return failed_ids, ", ".join(pos_fmt)


def _execute_replay_snapshot(
    node: MotorPositionTestNode,
    snapshot: MotorSnapshot,
    snapshot_index: int | None = None,
):
    snapshot_label = f"#{int(snapshot_index)}" if snapshot_index is not None else "<unknown>"
    snapshot_time = _format_wall_time(snapshot.created_ts)
    target_positions, invalid_ids = _snapshot_target_positions(snapshot)
    if invalid_ids:
        print(f"快照{snapshot_label}不可回放：以下电机pos无效 {invalid_ids}")
        return {
            "result": "snapshot_invalid",
            "elapsed_sec": 0.0,
            "failed_ids": invalid_ids,
            "max_error": None,
        }

    print(f"开始同步位控28电机到快照{snapshot_label}(time={snapshot_time}, source={snapshot.source})...")
    node.set_all_motor_position_targets(target_positions)
    start = time.monotonic()
    ok_frames = 0
    result = "timeout"
    last_failed_ids = []
    last_missing_ids = []
    last_max_err = 0.0
    node.set_watchdog_fallback_enabled(False)
    try:
        while True:
            if node.is_feedback_stale():
                result = "feedback_stale"
                break
            reached, failed_ids, missing_ids, max_err = _evaluate_replay_reached(node, target_positions)
            last_failed_ids = failed_ids
            last_missing_ids = missing_ids
            last_max_err = max_err

            if reached:
                ok_frames += 1
            else:
                ok_frames = 0
            if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                result = "reached"
                break
            if (time.monotonic() - start) >= REPLAY_SNAPSHOT_TIMEOUT_SEC:
                break
            time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)

    elapsed = float(time.monotonic() - start)
    if result == "reached":
        print(f"快照回放完成：28电机达到目标（{elapsed:.2f}s, max_error={last_max_err:.3f}）。")
    elif result == "feedback_stale":
        node.set_idle_tension()
        print("快照回放中断：反馈超时，已切回默认张紧。")
        failed_ids, pos_fmt = _replay_failure_details(node, target_positions)
        print(f"回放失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前/目标位置: {pos_fmt}")
    else:
        node.set_idle_tension()
        print(f"快照回放超时（{elapsed:.2f}s），已切回默认张紧。")
        failed_ids, pos_fmt = _replay_failure_details(node, target_positions)
        print(f"回放失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前/目标位置: {pos_fmt}")

    return {
        "result": result,
        "elapsed_sec": elapsed,
        "failed_ids": list(last_failed_ids),
        "missing_ids": list(last_missing_ids),
        "max_error": float(last_max_err),
    }


def _execute_single_motor_to_target(node: MotorPositionTestNode, motor_id: int, target_pos: float):
    node.set_watchdog_fallback_enabled(False)
    assist_info = _apply_antagonist_assist_phase(node, motor_id, target_pos)
    if assist_info.get("feedback_stale"):
        post_fb = node.get_motor_feedback(motor_id)
        node.set_watchdog_fallback_enabled(True)
        node.set_idle_tension()
        return {
            "result": "feedback_stale",
            "target_pos": float(target_pos),
            "position_error": None,
            "elapsed_sec": 0.0,
            "assist_mode": node.assist_mode,
            "assist_applied": bool(assist_info.get("applied", False)),
            "assist_targets": assist_info.get("targets", []),
            "assist_primary_targets": assist_info.get("primary_targets", []),
            "assist_sway_pair_targets": assist_info.get("sway_pair_targets", []),
            "assist_release_sec": float(assist_info.get("release_sec", 0.0)),
            "assist_primary_release_sec": float(assist_info.get("primary_release_sec", 0.0)),
            "assist_sway_pair_release_sec": float(assist_info.get("sway_pair_release_sec", 0.0)),
            "assist_sway_pair_release_plan_sec": assist_info.get("sway_pair_release_plan_sec", {}),
            "assist_primary_start_delay_sec": float(assist_info.get("primary_start_delay_sec", 0.0)),
            "assist_delta_deg": float(assist_info.get("delta_deg", 0.0)),
            "assist_delta_release_profile": str(assist_info.get("delta_release_profile", "small")),
            "assist_delta_release_scale": float(assist_info.get("delta_release_scale", 1.0)),
            "assist_release_tor": float(assist_info.get("release_tor", ASSIST_RELEASE_TOR)),
            "post_fb": post_fb,
        }

    # Keep active target latched for arrival checking after assist phase.
    node.set_single_motor_position_target(motor_id, target_pos)
    start = time.monotonic()
    ok_frames = 0
    result = "timeout"
    last_err = None

    try:
        while True:
            if node.is_feedback_stale():
                result = "feedback_stale"
                break
            fb = node.get_motor_feedback(motor_id)
            if fb.pos is not None:
                last_err = abs(float(fb.pos) - float(target_pos))
                if last_err <= ARRIVAL_POS_TOL:
                    ok_frames += 1
                else:
                    ok_frames = 0
                if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                    result = "reached"
                    break
            if (time.monotonic() - start) >= SINGLE_MOTOR_TIMEOUT_SEC:
                result = "timeout"
                break
            time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)

    post_fb = node.get_motor_feedback(motor_id)
    if result != "reached":
        node.set_idle_tension()
    return {
        "result": result,
        "target_pos": float(target_pos),
        "position_error": last_err,
        "elapsed_sec": float(time.monotonic() - start),
        "assist_mode": node.assist_mode,
        "assist_applied": bool(assist_info.get("applied", False)),
        "assist_targets": assist_info.get("targets", []),
        "assist_primary_targets": assist_info.get("primary_targets", []),
        "assist_sway_pair_targets": assist_info.get("sway_pair_targets", []),
        "assist_release_sec": float(assist_info.get("release_sec", 0.0)),
        "assist_primary_release_sec": float(assist_info.get("primary_release_sec", 0.0)),
        "assist_sway_pair_release_sec": float(assist_info.get("sway_pair_release_sec", 0.0)),
        "assist_sway_pair_release_plan_sec": assist_info.get("sway_pair_release_plan_sec", {}),
        "assist_primary_start_delay_sec": float(assist_info.get("primary_start_delay_sec", 0.0)),
        "assist_delta_deg": float(assist_info.get("delta_deg", 0.0)),
        "assist_delta_release_profile": str(assist_info.get("delta_release_profile", "small")),
        "assist_delta_release_scale": float(assist_info.get("delta_release_scale", 1.0)),
        "assist_release_tor": float(assist_info.get("release_tor", ASSIST_RELEASE_TOR)),
        "post_fb": post_fb,
    }


def _all_reached_zero(node: MotorPositionTestNode):
    pos_list = node.all_motor_positions()
    available = [p for p in pos_list if p is not None]
    if not available:
        return False
    for p in available:
        if abs(float(p)) > ARRIVAL_POS_TOL:
            return False
    return True


def _zero_failure_details(node: MotorPositionTestNode):
    pos_list = node.all_motor_positions()
    failed_ids = []
    for i, p in enumerate(pos_list, start=1):
        if p is None:
            failed_ids.append(i)
            continue
        if abs(float(p)) > ARRIVAL_POS_TOL:
            failed_ids.append(i)
    pos_fmt = ", ".join(
        [f"m{i+1}:{('None' if p is None else f'{float(p):.3f}')}" for i, p in enumerate(pos_list)]
    )
    return failed_ids, pos_fmt


def _zero_sway_motor_ids(hand: str):
    return sorted(
        int(mid)
        for mid in ASSIST_DURATION_CATEGORY.get(hand, {}).get("sway", set())
        if 1 <= int(mid) <= MOTOR_COUNT
    )


def _zero_flexor_motor_ids(hand: str):
    category = ASSIST_DURATION_CATEGORY.get(hand, {})
    return sorted(
        int(mid)
        for mid in set(category.get("flexor", set()))
        if 1 <= int(mid) <= MOTOR_COUNT
    )


def _execute_all_zero(node: MotorPositionTestNode):
    print("开始全部回0...")
    zero_start = None
    ok_frames = 0
    result = "timeout"
    sway_motor_ids = _zero_sway_motor_ids(node.hand)
    flexor_motor_ids = _zero_flexor_motor_ids(node.hand)
    node.set_watchdog_fallback_enabled(False)
    try:
        pre_release_total_sec = max(float(ZERO_SWAY_RELEASE_SEC), float(ZERO_FLEXOR_RELEASE_SEC))
        if pre_release_total_sec > 0.0 and (sway_motor_ids or flexor_motor_ids):
            if sway_motor_ids:
                print(
                    "回0前侧摆放线: "
                    f"motors={sway_motor_ids}, tor={ZERO_SWAY_RELEASE_TOR_A:.3f}A, "
                    f"duration={ZERO_SWAY_RELEASE_SEC:.3f}s"
                )
            if flexor_motor_ids:
                print(
                    "回0前屈肌放线: "
                    f"motors={flexor_motor_ids}, tor={ZERO_FLEXOR_RELEASE_TOR_A:.3f}A, "
                    f"duration={ZERO_FLEXOR_RELEASE_SEC:.3f}s"
                )

            release_start = time.monotonic()
            while (time.monotonic() - release_start) < pre_release_total_sec:
                if node.is_feedback_stale():
                    result = "feedback_stale"
                    break
                elapsed = time.monotonic() - release_start
                overrides = {}
                if sway_motor_ids and elapsed < float(ZERO_SWAY_RELEASE_SEC):
                    for mid in sway_motor_ids:
                        overrides[int(mid)] = [0.0, 0.0, float(ZERO_SWAY_RELEASE_TOR_A), 0.0, 0.0]
                if flexor_motor_ids and elapsed < float(ZERO_FLEXOR_RELEASE_SEC):
                    for mid in flexor_motor_ids:
                        cur_row = overrides.get(int(mid))
                        cur = float(cur_row[2]) if cur_row is not None else 0.0
                        overrides[int(mid)] = [
                            0.0,
                            0.0,
                            max(cur, float(ZERO_FLEXOR_RELEASE_TOR_A)),
                            0.0,
                            0.0,
                        ]
                # Keep all motors in zero position-control during pre-release,
                # then override flexor/sway motors to pure release rows.
                node.set_all_zero_control_with_overrides(overrides)
                time.sleep(CONTROL_PERIOD_SEC)

        if result != "feedback_stale":
            node.set_all_zero_control()
            zero_start = time.monotonic()
            while True:
                if node.is_feedback_stale():
                    result = "feedback_stale"
                    break
                if _all_reached_zero(node):
                    ok_frames += 1
                else:
                    ok_frames = 0
                if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                    result = "reached"
                    break
                if (time.monotonic() - zero_start) >= RETURN_ZERO_TIMEOUT_SEC:
                    break
                time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)
    node.set_idle_tension()
    elapsed = (
        time.monotonic() - zero_start
        if zero_start is not None
        else 0.0
    )
    if result == "reached":
        print(f"全部回0完成：已达到阈值（{elapsed:.2f}s）并切回默认张紧。")
    elif result == "feedback_stale":
        print("全部回0中断：反馈超时，已切回默认张紧。")
        failed_ids, pos_fmt = _zero_failure_details(node)
        print(f"回0失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前28个电机位置: {pos_fmt}")
    else:
        print(f"全部回0超时（{elapsed:.2f}s），已切回默认张紧。")
        failed_ids, pos_fmt = _zero_failure_details(node)
        print(f"回0失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前28个电机位置: {pos_fmt}")
    return result


def _print_motor_defaults(hand_defaults, motor_id: int):
    idx = motor_id - 1
    print(
        f"默认参数 motor={motor_id}: "
        f"vel={float(hand_defaults['vel'][idx])}, "
        f"tor_active={float(hand_defaults['tor_active'][idx])}, "
        f"kp={float(hand_defaults['kp'][idx])}, kd={float(hand_defaults['kd'][idx])}, "
        f"idle_tor={float(hand_defaults['tor'][idx])}"
    )


def _run_interactive_loop(
    node: MotorPositionTestNode,
    defaults_hand,
    snapshot_history: list[MotorSnapshot],
    snapshot_path: Path,
):
    while True:
        act, motor_id = _prompt_select_motor()
        if act == "quit":
            _execute_all_zero(node)
            return
        if act == "snapshot":
            _append_snapshot_and_report(
                snapshot_history,
                snapshot_path,
                node,
                source="manual_h(select_menu)",
            )
            continue
        if act == "record_once":
            node.record_motor_feedback_once(reason="manual_r(select_menu)")
            continue
        if act == "replay":
            replay_idx = _prompt_replay_index(snapshot_history)
            if replay_idx is not None:
                _execute_replay_snapshot(
                    node,
                    snapshot_history[replay_idx - 1],
                    snapshot_index=replay_idx,
                )
            continue
        if act == "zero":
            _execute_all_zero(node)
            continue

        _print_motor_defaults(defaults_hand, motor_id)

        while True:
            cur_fb = node.get_motor_feedback(motor_id)
            _print_feedback("当前反馈", motor_id, cur_fb)

            cmd_act, target_pos = _prompt_target_position()
            if cmd_act == "quit":
                _execute_all_zero(node)
                return
            if cmd_act == "zero":
                _execute_all_zero(node)
                break
            if cmd_act == "snapshot":
                _append_snapshot_and_report(
                    snapshot_history,
                    snapshot_path,
                    node,
                    source=f"manual_h(motor={motor_id})",
                )
                continue
            if cmd_act == "record_once":
                node.record_motor_feedback_once(reason=f"manual_r(motor={motor_id})")
                continue
            if cmd_act == "replay":
                replay_idx = _prompt_replay_index(snapshot_history)
                if replay_idx is not None:
                    _execute_replay_snapshot(
                        node,
                        snapshot_history[replay_idx - 1],
                        snapshot_index=replay_idx,
                    )
                continue
            if cmd_act == "change_motor":
                node.set_idle_tension()
                break

            result = _execute_single_motor_to_target(node, motor_id, target_pos)
            post_fb = result["post_fb"]
            print(
                f"执行结果: {result['result']}, target_pos={result['target_pos']}, "
                f"position_error={result['position_error']}, elapsed={result['elapsed_sec']:.2f}s"
            )
            if result.get("assist_applied"):
                print(
                    f"对抗肌辅助已执行: mode={_assist_mode_label(result.get('assist_mode', ASSIST_MODE_OFF))}, "
                    f"targets={result.get('assist_targets', [])}, "
                    f"release_sec={result.get('assist_release_sec', 0.0):.3f}, "
                    f"primary_targets={result.get('assist_primary_targets', [])}, "
                    f"sway_pair_targets={result.get('assist_sway_pair_targets', [])}, "
                    f"primary_release_sec={result.get('assist_primary_release_sec', 0.0):.3f}, "
                    f"primary_start_delay_sec={result.get('assist_primary_start_delay_sec', 0.0):.3f}, "
                    f"sway_pair_release_sec={result.get('assist_sway_pair_release_sec', 0.0):.3f}, "
                    f"sway_pair_release_plan_sec={result.get('assist_sway_pair_release_plan_sec', {})}, "
                    f"delta_deg={result.get('assist_delta_deg', 0.0):.3f}, "
                    f"delta_profile={result.get('assist_delta_release_profile', 'small')}, "
                    f"delta_scale={result.get('assist_delta_release_scale', 1.0):.3f}, "
                    f"release_tor={result.get('assist_release_tor', ASSIST_RELEASE_TOR):.3f}"
                )
            _print_feedback("停止后反馈", motor_id, post_fb)
            if result["result"] == "reached":
                print("该电机保持当前目标位置，其他电机维持默认张紧。")
            else:
                print("未到位或异常，已切回默认张紧。")


def _hold_zero_output_until_exit(node: MotorPositionTestNode):
    print("已进入安全锁定模式：全部电流已置0，不接受任何动作。按 Ctrl+C 退出。")
    while True:
        node.set_all_zero_output()
        time.sleep(0.2)


def _parse_args():
    parser = argparse.ArgumentParser(description="Single-motor position test for Y-hand.")
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument("--defaults-path", default="")
    parser.add_argument("--snapshot-path", default="")
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument(
        "--motor-feedback-csv-path",
        default="auto",
        help=(
            "Motor feedback CSV path or directory. Supports {hand}/{pid}/{timestamp}. "
            "Use auto for script-local default directory."
        ),
    )
    parser.add_argument(
        "--assist-enable",
        default="true",
        help="Enable antagonist assist mapping (true/false). Default: true",
    )
    return parser.parse_args()


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def main():
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    defaults_path = Path(args.defaults_path) if args.defaults_path else (script_dir / "muscle_adjustment_defaults.py")
    snapshot_path = (
        Path(args.snapshot_path).expanduser()
        if args.snapshot_path
        else (script_dir / f"motor_position_snapshots_{args.hand}.json")
    )
    snapshot_history = _load_snapshot_history(snapshot_path)
    defaults = _load_defaults_from_py(defaults_path)
    defaults_hand = defaults[args.hand]
    tor_safety_violations = _validate_tor_pair_safety(args.hand, defaults_hand["tor"])
    runtime_defaults_hand = defaults_hand
    if tor_safety_violations:
        runtime_defaults_hand = copy.deepcopy(defaults_hand)
        runtime_defaults_hand["tor"] = [0.0] * MOTOR_COUNT
        runtime_defaults_hand["tor_active"] = [0.0] * MOTOR_COUNT

    cmd_topic, feedback_topic = _topics_for_hand(args.hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic
    if args.feedback_topic:
        feedback_topic = args.feedback_topic
    assist_enabled = _parse_bool_text(args.assist_enable, default=True)
    assist_mode = ASSIST_MODE_RELEASE if assist_enabled else ASSIST_MODE_OFF

    rclpy.init()
    node = MotorPositionTestNode(
        hand=args.hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        defaults_hand=runtime_defaults_hand,
        assist_mode=assist_mode,
        motor_feedback_csv_path=str(args.motor_feedback_csv_path),
    )
    if tor_safety_violations:
        node.set_all_zero_output()
    else:
        node.set_idle_tension()
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if tor_safety_violations:
            _print_tor_safety_violations(args.hand, tor_safety_violations)
            _hold_zero_output_until_exit(node)
            return

        ok, msg = node.check_topic_safety(timeout_sec=12.0)
        if not ok:
            print(f"错误: {msg}")
            return

        if not node.wait_feedback(timeout_sec=6.0):
            cmd_pub = node.count_publishers(cmd_topic)
            cmd_sub = node.count_subscribers(cmd_topic)
            fb_pub = node.count_publishers(feedback_topic)
            fb_sub = node.count_subscribers(feedback_topic)
            print(
                "错误: 启动后未收到反馈数据，请检查设备与话题连接。"
                f"\n当前话题状态: {cmd_topic} pubs={cmd_pub}, subs={cmd_sub}; "
                f"{feedback_topic} pubs={fb_pub}, subs={fb_sub}"
            )
            return

        print(
            f"进入电机位置测试模式 hand={args.hand}, cmd_topic={cmd_topic}, feedback_topic={feedback_topic}\n"
            "规则: 只输入主动电机目标位置pos；主动电机其余4维使用 defaults(vel/tor_active/kp/kd)\n"
            "非主动电机: 纯默认张紧 [0,0,tor,0,0]\n"
            f"快照文件: {snapshot_path} (已加载 {len(snapshot_history)} 条)\n"
            "快照: 仅在输入 h 时记录并落盘；输入 replay 可按索引回放\n"
            "状态CSV: 每次输入 r 记录一条当前28电机反馈状态\n"
            f"对抗肌辅助: {_assist_mode_label(assist_mode)}\n"
            f"到位判定: |error|<={ARRIVAL_POS_TOL} 且连续 {ARRIVAL_CONFIRM_FRAMES} 帧, "
            f"单电机超时 {SINGLE_MOTOR_TIMEOUT_SEC}s, 全部回0超时 {RETURN_ZERO_TIMEOUT_SEC}s, "
            f"快照回放超时 {REPLAY_SNAPSHOT_TIMEOUT_SEC}s"
        )
        _run_interactive_loop(
            node,
            defaults_hand,
            snapshot_history=snapshot_history,
            snapshot_path=snapshot_path,
        )
    except KeyboardInterrupt:
        print("收到 Ctrl+C，执行全部回0后退出。")
        _execute_all_zero(node)
    finally:
        node.close_motor_feedback_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from DEVICES_MSG_IMPORT_ERROR
    main()
