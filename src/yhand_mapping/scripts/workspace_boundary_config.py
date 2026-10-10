#!/usr/bin/env python3
from dataclasses import dataclass

from hand_motor_map import (
    HAND_MOTOR_MAP,
)

MOTOR_COUNT = 28
DEFAULT_HAND = "left"
DEFAULT_SAMPLE_HZ = 200.0
DEFAULT_MAX_OUTWARD_EXPAND = 14.0
SIMULTANEOUS_FLEXOR_CURRENT = -0.03
SIMULTANEOUS_EXTENSOR_CURRENT = -0.05
SIMULTANEOUS_SIDE_CURRENT = -0.05

# Per-motor passive tension current (A), explicit 28 entries.
# Index 0 -> motor 1, ..., index 27 -> motor 28.
DEFAULT_TENSION_CURRENT_BY_HAND = {'left': [-0.05,
          -0.03,
          -0.03,
          -0.04,
          -0.03,
          -0.04,
          -0.04,
          -0.04,
          -0.05,
          -0.05,
          -0.05,
          -0.03,
          -0.03,
          -0.04,
          -0.03,
          -0.03,
          -0.03,
          -0.05,
          -0.03,
          -0.05,
          -0.05,
          -0.05,
          -0.05,
          -0.05,
          -0.05,
          -0.03,
          -0.04,
          -0.05],
 'right': [-0.03,
           -0.03,
           -0.03,
           -0.05,
           -0.03,
           -0.05,
           -0.05,
           -0.05,
           -0.05,
           -0.05,
           -0.05,
           -0.03,
           -0.04,
           -0.05,
           -0.05,
           -0.03,
           -0.03,
           -0.04,
           -0.03,
           -0.04,
           -0.04,
           -0.04,
           -0.05,
           -0.05,
           -0.05,
           -0.03,
           -0.03,
           -0.04]}

HAND_TOPICS = {
    "left": {
        "cmd_topic": "/yhand_left_cmd",
        "feedback_topic": "/yhand_left_feedback",
    },
    "right": {
        "cmd_topic": "/yhand_right_cmd",
        "feedback_topic": "/yhand_right_feedback",
    },
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]
FINGER_LABELS = {
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "little": "小指",
}

# Motor roles are defined in hand_motor_map.py.


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    finger: str
    dims: int
    x_motor: int
    y_in_motor: int
    y_out_motor: int
    z_motor: int | None


def _thumb_specs(hand_map):
    thumb = hand_map["thumb"]
    return [
        ModelSpec(
            model_id="thumb_ext_short",
            label="拇指-短伸",
            finger="thumb",
            dims=3,
            x_motor=int(thumb["ext_short"]),
            y_in_motor=int(thumb["adduction"]),
            y_out_motor=int(thumb["abduction"]),
            z_motor=int(thumb["opposition"]),
        ),
        ModelSpec(
            model_id="thumb_ext_long",
            label="拇指-长伸",
            finger="thumb",
            dims=3,
            x_motor=int(thumb["ext_long"]),
            y_in_motor=int(thumb["adduction"]),
            y_out_motor=int(thumb["abduction"]),
            z_motor=int(thumb["opposition"]),
        ),
        ModelSpec(
            model_id="thumb_flex_long",
            label="拇指-长屈",
            finger="thumb",
            dims=3,
            x_motor=int(thumb["flex_long"]),
            y_in_motor=int(thumb["adduction"]),
            y_out_motor=int(thumb["abduction"]),
            z_motor=int(thumb["opposition"]),
        ),
    ]


def _index_specs(hand_map):
    index = hand_map["index"]
    return [
        ModelSpec(
            model_id="index_deep",
            label="食指-深屈",
            finger="index",
            dims=2,
            x_motor=int(index["flex_deep"]),
            y_in_motor=int(index["adduction"]),
            y_out_motor=int(index["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="index_shallow",
            label="食指-浅屈",
            finger="index",
            dims=2,
            x_motor=int(index["flex_shallow"]),
            y_in_motor=int(index["adduction"]),
            y_out_motor=int(index["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="index_ext",
            label="食指-伸肌",
            finger="index",
            dims=2,
            x_motor=int(index["extensor"]),
            y_in_motor=int(index["adduction"]),
            y_out_motor=int(index["abduction"]),
            z_motor=None,
        ),
    ]


def _middle_specs(hand_map):
    middle = hand_map["middle"]
    return [
        ModelSpec(
            model_id="middle_deep",
            label="中指-深屈",
            finger="middle",
            dims=2,
            x_motor=int(middle["flex_deep"]),
            y_in_motor=int(middle["adduction"]),
            y_out_motor=int(middle["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="middle_shallow",
            label="中指-浅屈",
            finger="middle",
            dims=2,
            x_motor=int(middle["flex_shallow"]),
            y_in_motor=int(middle["adduction"]),
            y_out_motor=int(middle["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="middle_ext",
            label="中指-伸肌",
            finger="middle",
            dims=2,
            x_motor=int(middle["extensor"]),
            y_in_motor=int(middle["adduction"]),
            y_out_motor=int(middle["abduction"]),
            z_motor=None,
        ),
    ]


def _ring_specs(hand_map):
    ring = hand_map["ring"]
    return [
        ModelSpec(
            model_id="ring_flex",
            label="无名指-屈肌",
            finger="ring",
            dims=2,
            x_motor=int(ring["flexor"]),
            y_in_motor=int(ring["adduction"]),
            y_out_motor=int(ring["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="ring_ext",
            label="无名指-伸肌",
            finger="ring",
            dims=2,
            x_motor=int(ring["extensor"]),
            y_in_motor=int(ring["adduction"]),
            y_out_motor=int(ring["abduction"]),
            z_motor=None,
        ),
    ]


def _little_specs(hand_map):
    little = hand_map["little"]
    return [
        ModelSpec(
            model_id="little_flex",
            label="小指-屈肌",
            finger="little",
            dims=2,
            x_motor=int(little["flexor"]),
            y_in_motor=int(little["adduction"]),
            y_out_motor=int(little["abduction"]),
            z_motor=None,
        ),
        ModelSpec(
            model_id="little_ext",
            label="小指-伸肌",
            finger="little",
            dims=2,
            x_motor=int(little["extensor"]),
            y_in_motor=int(little["adduction"]),
            y_out_motor=int(little["abduction"]),
            z_motor=None,
        ),
    ]


def build_model_specs(hand: str):
    hand_l = hand.strip().lower()
    if hand_l not in HAND_MOTOR_MAP:
        raise ValueError(f"Unsupported hand={hand}, expected one of {list(HAND_MOTOR_MAP)}")

    hand_map = HAND_MOTOR_MAP[hand_l]
    specs = []
    specs.extend(_index_specs(hand_map))
    specs.extend(_middle_specs(hand_map))
    specs.extend(_ring_specs(hand_map))
    specs.extend(_little_specs(hand_map))
    specs.extend(_thumb_specs(hand_map))
    return specs


def get_hand_topics(hand: str):
    hand_l = hand.strip().lower()
    if hand_l not in HAND_TOPICS:
        raise ValueError(f"Unsupported hand={hand}, expected one of {list(HAND_TOPICS)}")
    return HAND_TOPICS[hand_l]["cmd_topic"], HAND_TOPICS[hand_l]["feedback_topic"]


def get_default_tension_currents(hand: str):
    hand_l = hand.strip().lower()
    if hand_l not in DEFAULT_TENSION_CURRENT_BY_HAND:
        raise ValueError(
            f"Unsupported hand={hand}, expected one of {list(DEFAULT_TENSION_CURRENT_BY_HAND)}"
        )
    values = DEFAULT_TENSION_CURRENT_BY_HAND[hand_l]
    if len(values) != MOTOR_COUNT:
        raise ValueError(
            f"DEFAULT_TENSION_CURRENT_BY_HAND['{hand_l}'] must have {MOTOR_COUNT} values, got {len(values)}"
        )
    return [float(v) for v in values]


def get_simultaneous_tension_currents(hand: str):
    """Whole-finger baseline; thumb and wrist retain their default currents."""
    currents = get_default_tension_currents(hand)
    for finger in ("index", "middle", "ring", "little"):
        for role, mid in HAND_MOTOR_MAP[hand.strip().lower()][finger].items():
            if role == "extensor":
                current = SIMULTANEOUS_EXTENSOR_CURRENT
            elif role in ("abduction", "adduction"):
                current = SIMULTANEOUS_SIDE_CURRENT
            else:
                current = SIMULTANEOUS_FLEXOR_CURRENT
            currents[mid - 1] = current
    return currents
