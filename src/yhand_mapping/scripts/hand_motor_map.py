"""Motor assignments from 新腕带下的电机&肌肉对照表.xlsx (1-based SDK channels).

Model axis order describes muscles, so existing checkpoints keep their feature
order while their commands and feedback use the current hardware channels.
"""

MAPPING_VERSION = "new_wristband_20261008"
MAPPING_SOURCE = "新腕带下的电机&肌肉对照表.xlsx"
MOTOR_COUNT = 28

HAND_MOTOR_MAP = {
    "left": {
        "thumb": dict(ext_short=6, ext_long=7, flex_long=4, abduction=27, adduction=8, opposition=14),
        "index": dict(extensor=25, flex_shallow=3, flex_deep=15, abduction=28, adduction=1),
        "middle": dict(extensor=23, flex_shallow=16, flex_deep=2, abduction=24, adduction=9),
        "ring": dict(extensor=10, flexor=17, abduction=21, adduction=22),
        "little": dict(extensor=11, flexor=13, abduction=20, adduction=18),
    },
    "right": {
        "thumb": dict(ext_short=20, ext_long=21, flex_long=18, abduction=13, adduction=22, opposition=28),
        "index": dict(extensor=11, flex_shallow=17, flex_deep=1, abduction=14, adduction=15),
        "middle": dict(extensor=9, flex_shallow=2, flex_deep=16, abduction=10, adduction=23),
        "ring": dict(extensor=24, flexor=3, abduction=7, adduction=8),
        "little": dict(extensor=25, flexor=27, abduction=6, adduction=4),
    },
}
WRIST_MOTOR_MAP = {
    hand: dict(left_front=19, left_back=12, right_front=5, right_back=26)
    for hand in HAND_MOTOR_MAP
}
WRIST_MOTOR_IDS_BY_HAND = {
    hand: tuple(sorted(motors.values())) for hand, motors in WRIST_MOTOR_MAP.items()
}
FINGER_MOTOR_IDS_BY_HAND = {
    hand: tuple(sorted(mid for finger in fingers.values() for mid in finger.values()))
    for hand, fingers in HAND_MOTOR_MAP.items()
}
SIDE_SWING_MOTOR_IDS_BY_HAND = {
    hand: {finger[role] for finger in fingers.values() for role in ("abduction", "adduction")}
    for hand, fingers in HAND_MOTOR_MAP.items()
}
FLEXOR_MOTOR_IDS_BY_HAND = {
    hand: {mid for finger in fingers.values() for role, mid in finger.items()
           if role in ("flex_shallow", "flex_deep", "flexor", "flex_long")}
    for hand, fingers in HAND_MOTOR_MAP.items()
}
LEFT_TO_RIGHT_MOTOR_MAP = {
    mid: HAND_MOTOR_MAP["right"][finger][role]
    for finger, motors in HAND_MOTOR_MAP["left"].items() for role, mid in motors.items()
}
LEFT_TO_RIGHT_MOTOR_MAP.update({mid: WRIST_MOTOR_MAP["right"][role]
                                for role, mid in WRIST_MOTOR_MAP["left"].items()})
RIGHT_TO_LEFT_MOTOR_MAP = {right: left for left, right in LEFT_TO_RIGHT_MOTOR_MAP.items()}

# Historical channel permutations, used only to migrate per-muscle parameters
# and to interpret recordings made before this wiring change.
LEGACY_TO_CURRENT_MOTOR_IDS = {
    "left": {
        19: 19, 11: 12, 4: 5, 25: 26,
        26: 6, 6: 7, 28: 4, 5: 27, 7: 8, 14: 14,
        24: 25, 15: 3, 3: 15, 27: 28, 1: 1,
        23: 23, 16: 16, 2: 2, 8: 24, 9: 9,
        21: 10, 17: 17, 20: 21, 22: 22,
        10: 11, 18: 13, 12: 20, 13: 18,
    },
    "right": {
        5: 19, 25: 12, 18: 5, 11: 26,
        12: 20, 20: 21, 14: 18, 19: 13, 21: 22, 28: 28,
        10: 11, 1: 17, 17: 1, 13: 14, 15: 15,
        9: 9, 2: 2, 16: 16, 22: 10, 23: 23,
        7: 24, 3: 3, 6: 7, 8: 8,
        24: 25, 4: 27, 26: 6, 27: 4,
    },
}

MODEL_AXIS_ROLES_BY_HAND = {
    "left": {
        "thumb": ("abduction", "ext_long", "adduction", "opposition", "ext_short", "flex_long"),
        "index": ("adduction", "flex_deep", "flex_shallow", "extensor", "abduction"),
        "middle": ("flex_deep", "abduction", "adduction", "flex_shallow", "extensor"),
        "ring": ("flexor", "abduction", "extensor", "adduction"),
        "little": ("extensor", "abduction", "adduction", "flexor"),
    },
    "right": {
        "thumb": ("ext_short", "flex_long", "abduction", "ext_long", "adduction", "opposition"),
        "index": ("flex_shallow", "extensor", "abduction", "adduction", "flex_deep"),
        "middle": ("flex_shallow", "extensor", "flex_deep", "abduction", "adduction"),
        "ring": ("flexor", "abduction", "extensor", "adduction"),
        "little": ("flexor", "extensor", "abduction", "adduction"),
    },
}


def model_motor_ids(hand, finger):
    """Current channel IDs in the bundled checkpoint's original muscle axis order."""
    return [HAND_MOTOR_MAP[hand][finger][role] for role in MODEL_AXIS_ROLES_BY_HAND[hand][finger]]


def model_motor_ids_csv(hand, finger):
    return ",".join(map(str, model_motor_ids(hand, finger)))


def remap_motor_ids(hand, ids):
    return [LEGACY_TO_CURRENT_MOTOR_IDS[hand][int(mid)] for mid in ids]


def remap_motor_dict(hand, values):
    return {LEGACY_TO_CURRENT_MOTOR_IDS[hand][int(mid)]: value for mid, value in values.items()}


def remap_motor_profile(hand, values):
    if len(values) != MOTOR_COUNT:
        raise ValueError(f"Expected {MOTOR_COUNT} motor values, got {len(values)}")
    result = [None] * MOTOR_COUNT
    for old, new in LEGACY_TO_CURRENT_MOTOR_IDS[hand].items():
        result[new - 1] = values[old - 1]
    return result


def named_motors(hand):
    """Names retained for the Manus calibration action definitions."""
    result = dict(zip(("wrist_front", "wrist_back", "wrist_right_front", "wrist_right_back"),
                      (WRIST_MOTOR_MAP[hand][role] for role in
                       ("left_front", "left_back", "right_front", "right_back"))))
    aliases = {"ext_short": "short_ext", "ext_long": "long_ext", "flex_long": "long_flex",
               "extensor": "ext", "flex_shallow": "flex_superficial", "flex_deep": "flex_deep",
               "flexor": "flex", "abduction": "abd", "adduction": "add", "opposition": "opposition"}
    for finger, motors in HAND_MOTOR_MAP[hand].items():
        prefix = "pinky" if finger == "little" else finger
        result.update({f"{prefix}_{aliases[role]}": mid for role, mid in motors.items()})
    return result


def finger_assist_motor_layout(hand, finger):
    motors = HAND_MOTOR_MAP[hand][finger]
    layout = {"sway_pair": [motors["abduction"], motors["adduction"]]}
    if finger == "thumb":
        layout.update(extensors={"short": motors["ext_short"], "long": motors["ext_long"]},
                      flexors=[motors["flex_long"]], opposition=motors["opposition"])
    else:
        flexors = [motors[role] for role in ("flex_shallow", "flex_deep", "flexor") if role in motors]
        layout.update(extensors=[motors["extensor"]], flexors=flexors)
    return layout


def extensor_assist_map(hand):
    result = {}
    for finger in HAND_MOTOR_MAP[hand]:
        layout = finger_assist_motor_layout(hand, finger)
        extensors = layout["extensors"]
        if isinstance(extensors, dict):
            extensors = list(extensors.values())
        for mid in extensors:
            result[mid] = {"flexors": list(layout["flexors"])}
        result[layout["sway_pair"][0]] = {"flexors": [layout["sway_pair"][1]]}
    return result
