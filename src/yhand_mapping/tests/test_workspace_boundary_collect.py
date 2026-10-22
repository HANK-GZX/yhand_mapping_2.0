"""Check whole-finger collection and default data paths without real ROS nodes."""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import workspace_boundary_collect as collect
import data_collection_main as data_collection
from hand_motor_map import HAND_MOTOR_MAP
from workspace_boundary_config import (
    FINGER_ORDER, build_model_specs, get_default_tension_currents,
    get_simultaneous_tension_currents,
)
BOUNDARY_NODE = collect.PassiveBoundaryNode


def fake_node(currents):
    sent = []
    node = SimpleNamespace(
        _lock=threading.Lock(), tension_currents=np.asarray(currents),
        publisher=SimpleNamespace(publish=sent.append),
        wait_feedback=lambda timeout_sec: True, destroy_node=lambda: None,
    )
    for name in ("_publish_tension_cb", "_publish_currents_locked", "set_tension_currents"):
        setattr(node, name, MethodType(getattr(BOUNDARY_NODE, name), node))
    return node, sent


@pytest.mark.parametrize("hand", ["left", "right"])
def test_four_fingers_use_role_currents_and_thumb_wrist_keep_defaults(hand):
    currents = get_simultaneous_tension_currents(hand)
    defaults = get_default_tension_currents(hand)
    four_finger_motors = set()
    for finger in ("index", "middle", "ring", "little"):
        for role, mid in HAND_MOTOR_MAP[hand][finger].items():
            four_finger_motors.add(mid)
            assert currents[mid - 1] == pytest.approx(
                -0.05 if role in ("extensor", "abduction", "adduction") else -0.03)
    for mid in set(range(1, 29)) - four_finger_motors:
        assert currents[mid - 1] == defaults[mid - 1]
    node, sent = fake_node(currents)
    node._publish_tension_cb()
    assert len(sent[-1].motors) == 28
    assert [motor.tor for motor in sent[-1].motors] == pytest.approx(currents)
    assert all(motor.kp == motor.kd == motor.pos == motor.vel == 0 for motor in sent[-1].motors)
    assert not sent[-1].pretightenforce.flag


@pytest.mark.parametrize("hand", ["left", "right"])
def test_each_finger_projects_only_its_own_samples_to_all_combinations(hand):
    samples = {finger: [{"pos": np.arange(28, dtype=float) + idx * 100}]
               for idx, finger in enumerate(FINGER_ORDER)}
    specs = build_model_specs(hand)
    projected = collect._project_samples(samples, specs)
    for spec in specs:
        offset = FINGER_ORDER.index(spec.finger) * 100
        for direction, y_motor in (("in", spec.y_in_motor), ("out", spec.y_out_motor)):
            expected = [spec.x_motor - 1 + offset, y_motor - 1 + offset]
            if spec.dims == 3:
                expected.append(spec.z_motor - 1 + offset)
            assert projected[spec.model_id][direction] == [expected]
    fitted = collect._fit_all_models(collect._to_numpy_projected(projected), specs, 14,
                                    get_simultaneous_tension_currents(hand))
    assert len(fitted) == 32
    assert all(item["status"] == "insufficient_points" for item in fitted.values())


def configure_main(monkeypatch, tmp_path, hand, answers):
    inputs = iter(answers)
    recorded = []
    created = []

    def create_node(**kwargs):
        node, sent = fake_node(kwargs["tension_currents"])
        created.append((node, sent))
        return node

    def record(node, live_sample_cb, **kwargs):
        assert node.tension_currents == pytest.approx(get_simultaneous_tension_currents(hand))
        node._publish_tension_cb()
        count = 50 if not recorded else 25
        recorded.append(count)
        samples = [{"wall": float(i), "pos": -np.arange(28, dtype=float) - i,
                    "vel": np.zeros(28), "tor": node.tension_currents.copy()} for i in range(count)]
        for sample in samples:
            live_sample_cb(sample)
        return samples, False

    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(collect, "PassiveBoundaryNode", create_node)
    monkeypatch.setattr(collect, "_record_one_segment", record)
    monkeypatch.setattr(collect, "_spin_node", lambda node: None)
    monkeypatch.setattr(collect.rclpy, "init", lambda: None)
    monkeypatch.setattr(collect.rclpy, "ok", lambda: False)
    monkeypatch.setattr(collect, "_parse_args", lambda: SimpleNamespace(
        hand=hand, cmd_topic="", feedback_topic="", sample_hz=200,
        max_outward_expand=14, output_dir="", disable_plot=True))
    original_output = collect._build_default_output_dir
    monkeypatch.setattr(collect, "_build_default_output_dir", lambda script_dir: original_output(tmp_path))
    return recorded, created


@pytest.mark.parametrize("hand", ["left", "right"])
def test_default_entrypoint_collects_all_fingers_and_is_readable_by_data_collection(monkeypatch, tmp_path, hand):
    answers = []
    for option in ("1", "2", "3", "4", "5"):
        answers.extend([option, "r", "b"])
    answers.extend(["f", "x"])
    recorded, created = configure_main(monkeypatch, tmp_path, hand, answers)
    collect.main()
    assert recorded == [50, 25, 25, 25, 25]
    hand_dir = tmp_path / "motor_test_logs" / "workspace_boundaries" / hand
    index = json.loads((hand_dir / "run_index.json").read_text())
    assert index["collection_mode"] == "per_finger"
    assert sum(index["finger_counts"].values()) == 150
    assert len(index["combo_counts"]) == 20
    assert all(value == 25 for value in index["combo_counts"].values())
    models = json.loads((hand_dir / "boundary_models.json").read_text())
    assert models["collection_mode"] == "per_finger_simultaneous"
    assert models["tension_current_scope"] == "all_collection"
    for finger in FINGER_ORDER:
        assert len(list((hand_dir / "raw" / finger).glob("session_*.csv"))) == 1
        assert not any(path.is_dir() for path in (hand_dir / "raw" / finger).iterdir())
    for key, model in models["combos"].items():
        assert model["status"] == "ok"
        assert model["point_count"] == (50 if key.startswith("thumb_") else 25)
        if not key.startswith("thumb_"):
            assert model["collection"]["mode"] == "per_finger"
            assert model["collection"]["tension_current_by_motor"] == get_simultaneous_tension_currents(hand)
    _, loaded = data_collection._load_boundary_json(hand_dir / "boundary_models.json", hand)
    accepted = data_collection._build_combo_specs(loaded)
    assert {finger: len(combos) for finger, combos in accepted.items()} == dict(
        thumb=11, index=6, middle=6, ring=4, little=4)
    # Workspace retains all 12 models; only short/out/on is excluded.
    assert len([key for key in loaded if key.startswith('thumb_')]) == 12
    accepted_thumb_ids = {combo.combo_id for combo in accepted['thumb']}
    assert accepted_thumb_ids == {key for key in loaded if key.startswith('thumb_')} - {
        'thumb_ext_short_out_opp_on'}
    assert {'thumb_ext_long_in_opp_on', 'thumb_ext_long_out_opp_on',
            'thumb_ext_short_in_opp_on',
            'thumb_ext_short_in_opp_off', 'thumb_ext_short_out_opp_off'} <= accepted_thumb_ids
    total, lower, upper, _ = data_collection._build_finger_quota_plan(
        'thumb', accepted['thumb'], points_per_combo=1000)
    assert total == 11000 and set(lower) == set(upper) == accepted_thumb_ids
    assert set(lower.values()) == set(upper.values()) == {1000}
    node, sent = created[0]
    assert all([motor.tor for motor in msg.motors] == pytest.approx(get_simultaneous_tension_currents(hand))
               for msg in sent)


def test_overwrite_clears_current_finger_only(monkeypatch, tmp_path):
    configure_main(monkeypatch, tmp_path, "left", ["2", "r", "b", "3", "r", "b", "2", "o", "f", "x"])
    collect.main()
    hand_dir = tmp_path / "motor_test_logs/workspace_boundaries/left"
    assert not list((hand_dir / "raw/index").glob("session_*.csv"))
    assert len(list((hand_dir / "raw/middle").glob("session_*.csv"))) == 1
    index = json.loads((hand_dir / "run_index.json").read_text())
    assert index["finger_counts"]["index"] == 0
    assert index["finger_counts"]["middle"] == 25
    assert all(value == 0 for key, value in index["combo_counts"].items() if key.startswith("index_"))


@pytest.mark.parametrize("hand", ["left", "right"])
def test_separate_runs_keep_previous_fingers_and_collector_can_load_them(monkeypatch, tmp_path, hand):
    hand_dir = tmp_path / "motor_test_logs/workspace_boundaries" / hand
    saved = {}
    for option, finger in [("2", "index"), ("3", "middle"), ("4", "ring")]:
        configure_main(monkeypatch, tmp_path, hand, [option, "r", "f", "x"])
        collect.main()
        model = json.loads((hand_dir / "boundary_models.json").read_text())
        for previous_finger, combos in saved.items():
            assert {key:value for key,value in model["combos"].items()
                    if value["finger"] == previous_finger} == combos
        saved[finger] = {key:value for key,value in model["combos"].items() if value["finger"] == finger}
        assert all(value["status"] == "ok" for value in saved[finger].values())
        assert model["updated_fingers"] == [finger]
        index = json.loads((hand_dir / "run_index.json").read_text())
        assert all(index["finger_counts"][f] == 50 for f in saved)
    _, loaded = data_collection._load_boundary_json(hand_dir / "boundary_models.json", hand)
    accepted = data_collection._build_combo_specs(loaded)
    assert {finger:len(accepted[finger]) for finger in saved} == dict(index=6, middle=6, ring=4)
    backups = sorted((hand_dir / "backups").glob("boundary_models_*.json"))
    assert len(backups) == 2
    assert all(json.loads(path.read_text())["hand"] == hand for path in backups)


def test_fit_without_current_samples_keeps_saved_models_and_index(monkeypatch, tmp_path):
    configure_main(monkeypatch, tmp_path, "left", ["2", "r", "f", "x"])
    collect.main()
    hand_dir = tmp_path / "motor_test_logs/workspace_boundaries/left"
    before = {name:(hand_dir/name).read_bytes() for name in ["boundary_models.json", "run_index.json"]}
    configure_main(monkeypatch, tmp_path, "left", ["f", "x"])
    collect.main()
    assert {name:(hand_dir/name).read_bytes() for name in before} == before
    assert not (hand_dir / "backups").exists()


def test_clear_in_later_run_only_invalidates_selected_finger(monkeypatch, tmp_path):
    configure_main(monkeypatch, tmp_path, "left", ["2", "r", "b", "3", "r", "f", "x"])
    collect.main()
    hand_dir = tmp_path / "motor_test_logs/workspace_boundaries/left"
    before = json.loads((hand_dir / "boundary_models.json").read_text())["combos"]
    configure_main(monkeypatch, tmp_path, "left", ["2", "o", "f", "x"])
    collect.main()
    after = json.loads((hand_dir / "boundary_models.json").read_text())["combos"]
    assert all(model["status"] == "insufficient_points" for model in after.values() if model["finger"] == "index")
    assert {k:v for k,v in after.items() if v["finger"] == "middle"} == {
        k:v for k,v in before.items() if v["finger"] == "middle"}
    index = json.loads((hand_dir / "run_index.json").read_text())
    assert index["finger_counts"]["index"] == 0 and index["finger_counts"]["middle"] == 25
    assert not list((hand_dir / "raw/index").glob("session_*.csv"))
    assert len(list((hand_dir / "raw/middle").glob("session_*.csv"))) == 1


@pytest.mark.parametrize("old_content", ["broken json", '{"hand":"right","combos":{}}',
                                      '{"hand":"left","combos":null}'])
def test_invalid_saved_model_is_not_overwritten(tmp_path, old_content):
    path = tmp_path / "boundary_models.json"
    path.write_text(old_content)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        collect._save_boundary_models(path, {"hand":"left", "combos":{}}, {"middle"})
    assert path.read_text() == old_content
    assert not (tmp_path / "backups").exists()


def test_atomic_model_write_failure_keeps_old_file(monkeypatch, tmp_path):
    path = tmp_path / "boundary_models.json"
    original = {"hand":"left", "combos":{"index_deep_in":{"finger":"index", "status":"ok"}}}
    path.write_text(json.dumps(original))

    def fail_replace(*args):
        raise OSError("simulated failed replacement")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        collect._save_boundary_models(path, {"hand":"left", "combos":{}}, {"middle"})
    assert json.loads(path.read_text()) == original
    assert json.loads(next((tmp_path / "backups").glob("*.json")).read_text()) == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("hand", ["left", "right"])
def test_default_launch_has_only_hand_argument_and_uses_canonical_script(monkeypatch, tmp_path, hand):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    launch_path = ROOT / "launch/workspace_boundary_launch.py"
    spec = importlib.util.spec_from_file_location("workspace_launch_test", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/gnome-terminal")
    description = module.generate_launch_description()
    assert [action.name for action in description.entities if isinstance(action, DeclareLaunchArgument)] == ["target_hand"]
    context = LaunchContext()
    context.launch_configurations.update(target_hand=hand)
    actions = module._launch_setup(context)
    driver = next(action for action in actions if isinstance(action, ExecuteProcess))
    collector = next(action for action in actions if isinstance(action, TimerAction)).actions[0]
    driver_cmd = [part.perform(context) for arg in driver.cmd for part in arg]
    collect_cmd = [part.perform(context) for arg in collector.cmd for part in arg]
    assert driver_cmd[4] == f"y_hand_{hand}_workspace_node"
    assert str(ROOT / "scripts/workspace_boundary_collect.py") in collect_cmd
    assert collect_cmd[collect_cmd.index("--hand") + 1] == hand
