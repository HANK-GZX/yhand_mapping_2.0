"""Verify hardware assignments and preserve per-muscle tuning without ROS nodes."""
import importlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_motor_map as mapping

LEGACY = json.loads((ROOT / "tests/fixtures/legacy_motor_parameters.json").read_text())
LABELS = {
    "左前手腕": "wrist_front", "左后手腕": "wrist_back",
    "右前手腕": "wrist_right_front", "右后手腕": "wrist_right_back",
    "拇短伸肌": "thumb_short_ext", "拇长伸肌": "thumb_long_ext",
    "拇长屈肌": "thumb_long_flex", "拇展肌": "thumb_abd",
    "拇收肌": "thumb_add", "拇对掌肌": "thumb_opposition",
    "食指伸肌": "index_ext", "食指指浅屈肌": "index_flex_superficial",
    "食指指深屈肌": "index_flex_deep", "食指外展": "index_abd", "食指内收": "index_add",
    "中指伸肌": "middle_ext", "中指指浅屈肌": "middle_flex_superficial",
    "中指指深屈肌": "middle_flex_deep", "中指外展": "middle_abd", "中指内收": "middle_add",
    "无名指伸肌": "ring_ext", "无名指屈肌": "ring_flex",
    "无名指外展": "ring_abd", "无名指内收": "ring_add",
    "小指伸肌": "pinky_ext", "小指屈肌": "pinky_flex",
    "小指外展": "pinky_abd", "小指内收": "pinky_add",
}


@lru_cache
def spreadsheet_motors(hand):
    wb = openpyxl.load_workbook(ROOT / mapping.MAPPING_SOURCE, data_only=True)
    offset = 0 if hand == "left" else 3
    result = {LABELS[row[offset + 2]]: row[offset] for row in wb.active.values
              if isinstance(row[offset], int)}
    wb.close()
    return result


@lru_cache
def permutation(hand):
    old = LEGACY["assignments"]["scripts/manus_calibration.py"][hand.upper() + "_HAND_MOTORS"]
    new = spreadsheet_motors(hand)
    return {int(mid): new[muscle] for muscle, mid in old.items()}


def profile(hand, values):
    result = [None] * 28
    for old, new in permutation(hand).items():
        result[new - 1] = values[old - 1]
    return result


def keyed(hand, values):
    return {permutation(hand)[int(mid)]: value for mid, value in values.items()}


def normalized(value):
    return json.loads(json.dumps(value))


@lru_cache
def module(relative):
    if relative.startswith("scripts/"):
        return importlib.import_module(Path(relative).stem)
    spec = importlib.util.spec_from_file_location("mapping_test_" + Path(relative).stem, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("hand", ["left", "right"])
def test_all_28_channels_match_spreadsheet(hand):
    assert mapping.named_motors(hand) == spreadsheet_motors(hand)
    assert sorted(mapping.named_motors(hand).values()) == list(range(1, 29))
    assert mapping.LEGACY_TO_CURRENT_MOTOR_IDS[hand] == permutation(hand)
    assert set(mapping.FINGER_MOTOR_IDS_BY_HAND[hand]).isdisjoint(mapping.WRIST_MOTOR_IDS_BY_HAND[hand])
    assert len(mapping.FINGER_MOTOR_IDS_BY_HAND[hand]) == 24
    for muscle, left in spreadsheet_motors("left").items():
        right = spreadsheet_motors("right")[muscle]
        assert mapping.LEFT_TO_RIGHT_MOTOR_MAP[left] == right
        assert mapping.RIGHT_TO_LEFT_MOTOR_MAP[right] == left


def migrated(hand, values):
    return [permutation(hand)[int(mid)] for mid in values]


PARAMETERS = [(path, name, value) for path, items in LEGACY["assignments"].items()
              for name, value in items.items()
              if name not in {"HAND_MOTOR_MAP", "LEFT_HAND_MOTORS", "RIGHT_HAND_MOTORS"}]


@pytest.mark.parametrize("path,name,original", PARAMETERS, ids=[p + ":" + n for p, n, _ in PARAMETERS])
def test_parameters_stay_with_original_muscle(path, name, original):
    current = getattr(module(path), name)
    if name.startswith("MOTOR_FEEDBACK_LIMITS_"):
        expected = keyed("left" if "LEFT" in name else "right", original)
    elif name == "DEFAULTS":
        expected = {hand: {key: profile(hand, values) for key, values in profiles.items()}
                    for hand, profiles in original.items()}
    elif name == "DEFAULT_TENSION_CURRENT_BY_HAND":
        # Boundary currents were explicitly retuned after the wiring migration.
        expected = {hand: [-0.03] * len(values) for hand, values in original.items()}
        for hand, currents in expected.items():
            for muscle, motor_id in spreadsheet_motors(hand).items():
                if muscle.startswith("thumb_"):
                    currents[motor_id - 1] = -0.02
                elif muscle.endswith(("_ext", "_abd", "_add")):
                    currents[motor_id - 1] = -0.05
    elif name in {"WRIST_GEOMETRY_BY_HAND", "MOTOR_CMD_OFFSET_MAP_DEFAULTS",
                  "RING_EXT_CMD_OFFSET_BY_HAND", "RING_SIDE_SWING_CMD_OFFSET_BY_HAND"}:
        expected = {hand: keyed(hand, values) for hand, values in original.items()}
    elif name.startswith("DATA_COLLECTION_RELEASE_"):
        expected = {hand: keyed(hand, {active: keyed(hand, targets) for active, targets in values.items()})
                    for hand, values in original.items()}
    elif name == "EXTENSOR_ASSIST_MAP":
        expected = {hand: keyed(hand, {ext: {"flexors": migrated(hand, params["flexors"])}
                                     for ext, params in values.items()})
                    for hand, values in original.items()}
    elif name == "FINGER_ASSIST_CONFIG":
        expected = json.loads(json.dumps(original))
        for hand, fingers in expected.items():
            for params in fingers.values():
                for key in ("extensors", "flexors", "sway_pair"):
                    values = params[key]
                    params[key] = {role: permutation(hand)[mid] for role, mid in values.items()} \
                        if isinstance(values, dict) else migrated(hand, values)
                if "opposition" in params:
                    params["opposition"] = permutation(hand)[params["opposition"]]
    elif name in {"FINGER_DEFAULTS_BY_HAND", "FINGER_RUNTIME_PRESETS"}:
        expected = json.loads(json.dumps(original))
        finger_defaults = [(finger, params) for fingers in expected.values() for finger, params in fingers.items()] \
            if name == "FINGER_DEFAULTS_BY_HAND" else list(expected.items())
        old_fingers = LEGACY["assignments"]["scripts/workspace_boundary_config.py"]["HAND_MOTOR_MAP"]
        for finger, params in finger_defaults:
            field = next(key for key in ("motor_ids", "motor_ids_csv", "right_native_motor_ids", "model_motor_ids")
                         if key in params)
            values = params[field]
            ids = list(map(int, values.split(","))) if isinstance(values, str) else values
            origin = next(hand for hand in ("left", "right")
                          if set(ids) == set(old_fingers[hand][finger].values()))
            new = migrated(origin, ids)
            params[field] = ",".join(map(str, new)) if isinstance(values, str) else new
    else:
        raise AssertionError(name)
    assert normalized(current) == normalized(expected)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_side_swing_uses_selected_hand(hand):
    expected = {mid for muscle, mid in spreadsheet_motors(hand).items()
                if not muscle.startswith("thumb_") and muscle.endswith(("_abd", "_add"))}
    for name in ("five_finger_delta_mapping_main", "curl_m0_m2_mlp_mapping_main"):
        mod = importlib.import_module(name)
        assert {mid for mid in range(1, 29) if mod._is_side_swing_motor_id(mid, hand)} == expected


@pytest.mark.parametrize("filename", LEGACY["config"])
def test_json_offsets_and_zero_groups(filename):
    original = LEGACY["config"][filename]
    current = json.loads((ROOT / "config" / filename).read_text())
    expected = json.loads(json.dumps(original))
    entries = expected.get("hands", {}) or ({expected["hand"]: expected} if "hand" in expected else {})
    for hand, entry in entries.items():
        for field in ("motor_position_offsets_deg", "motor_offsets_deg", "wrist_beta_positive_offsets_deg"):
            if field in entry:
                entry[field] = profile(hand, entry[field])
    if "back_delta_motor_groups" in expected:
        expected["back_delta_motor_groups"] = {
            hand: {group: migrated(hand, ids) for group, ids in groups.items()}
            for hand, groups in expected["back_delta_motor_groups"].items()}
    assert current == expected


def test_saved_snapshot_feedback_and_labels_migrate_together():
    current = json.loads((ROOT / "scripts/motor_position_snapshots_right.json").read_text())
    assert current["motor_mapping_version"] == mapping.MAPPING_VERSION
    for old, new in zip(LEGACY["snapshots"]["snapshots"], current["snapshots"], strict=True):
        assert new["feedback_rows"] == profile("right", old["feedback_rows"])
        assert new["created_ts"] == old["created_ts"]
    assert current["snapshots"][0]["source"] == "manual_h(motor=18)"


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("angles", [(0, 0), (5, 30), (-8, -20)])
def test_wrist_compensation_preserves_per_muscle_results(hand, angles, monkeypatch):
    wrist = module("scripts/wrist_angle_motor_control.py")
    current = wrist._compute_motor_targets_deg(*angles, hand=hand)
    old = LEGACY["assignments"]["scripts/wrist_angle_motor_control.py"]["WRIST_GEOMETRY_BY_HAND"]
    with monkeypatch.context() as patch:
        patch.setattr(wrist, "WRIST_GEOMETRY_BY_HAND", {h: {int(k): v for k, v in vals.items()} for h, vals in old.items()})
        previous = wrist._compute_motor_targets_deg(*angles, hand=hand)
    for before, after in zip(previous, current, strict=True):
        assert after == pytest.approx(profile(hand, before))


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("source", ["current", "legacy"])
def test_preprocessing_excludes_wrist_and_preserves_thumb_axes(hand, source, tmp_path):
    raw = importlib.import_module("mp.preprocess_raw")
    thumb = importlib.import_module("mp.preprocess_thumb")
    current_positions = list(range(100, 128))
    positions = [current_positions[permutation(hand)[mid] - 1] for mid in range(1, 29)] \
        if source == "legacy" else current_positions
    cmd = openpyxl.Workbook()
    feedback = openpyxl.Workbook()
    feedback.active.append(["pos"])
    for value in positions:
        cmd.active.append([value])
        feedback.active.append([value])
    cmd.save(tmp_path / "cmd.xlsx")
    feedback.save(tmp_path / "feedback.xlsx")
    glove = openpyxl.Workbook()
    glove.active.append(list(range(48)))
    glove.save(tmp_path / "glove.xlsx")
    assert raw.first_row_slice(tmp_path / "glove.xlsx", raw.GLOVE_Q_SLICES_BY_HAND[hand], 20) == \
        list(range(8, 28) if hand == "left" else range(28, 48))
    cmd_ignore, feedback_ignore, order = raw.motor_row_layout(hand, source)
    cmd_values = raw.first_column_without_rows(tmp_path / "cmd.xlsx", cmd_ignore, 24)
    values = raw.first_column_without_rows(tmp_path / "feedback.xlsx", feedback_ignore, 24)
    compact = [values[i] for i in order]
    expected = [current_positions[mid - 1] for mid in mapping.FINGER_MOTOR_IDS_BY_HAND[hand]]
    assert compact == expected == [cmd_values[i] for i in order]
    source_columns = thumb.thumb_source_columns(hand)
    assert [compact[int(col[2:])] for col in source_columns] == [current_positions[mid - 1] for mid in mapping.model_motor_ids(hand, "thumb")]


@pytest.mark.parametrize("hand", ["left", "right"])
def test_csv_replay_remaps_old_header_channels(hand, tmp_path):
    mod = module("scripts/data_collection_from_motor_feedback_csv.py")
    path = tmp_path / "legacy.csv"
    path.write_text("frame_idx,motor_1_pos,motor_3_pos,motor_14_pos\n0,10,30,140\n")
    args = SimpleNamespace(hand=hand, source_motor_map="legacy", cmd_csv_path=str(path),
                           collection_name="test", start_row=0, stride=1, max_points=0)
    collection = mod._load_collections(args)[0]
    assert collection.frames[0].target_by_motor == keyed(hand, {1: 10.0, 3: 30.0, 14: 140.0})
    assert collection.controlled_motor_ids == sorted(permutation(hand)[mid] for mid in (1, 3, 14))


def test_geometry_workbooks_match_current_labels():
    inverse_labels = {alias: label for label, alias in LABELS.items()}
    for path in ROOT.glob("Yhand*.xlsx"):
        wb = openpyxl.load_workbook(path, data_only=False)
        for sheet in wb:
            if "Coordinates" not in sheet.title and "Calculator" not in sheet.title and "Formula_Text" not in sheet.title:
                continue
            hand = "left" if sheet.title.startswith("Left_") else "right"
            expected = {inverse_labels[alias]: mid for alias, mid in spreadsheet_motors(hand).items()}
            rows = [(row[0], row[1]) for row in sheet.values if isinstance(row[0], int)]
            assert len(rows) == 28
            assert {label: mid for mid, label in rows} == expected
        wb.close()
