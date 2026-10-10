import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration


CONTROL_SCRIPT_START_DELAY_SEC = 1.5
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")

# Runtime wrapper:
# keep data recording logic untouched in scripts/, but force CSV collection
# to use -0.08A reel current during post-reach settle and 1.0s pre-record delay.
CSV_COLLECTION_REEL_WRAPPER_TEMPLATE = r"""
import importlib.util
import pathlib
import random
import sys
import time

RANDOM_CMD_PUBLISH = __RANDOM_CMD_PUBLISH__
MIN_CMD_GAP_FRAMES = __MIN_CMD_GAP_FRAMES__
CMD_RANDOM_SEED = __CMD_RANDOM_SEED__

script_path = pathlib.Path(sys.argv[1]).resolve()
args = list(sys.argv[2:])
sys.path.insert(0, str(script_path.parent))

spec = importlib.util.spec_from_file_location("dccsv_runtime", str(script_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _frame_gap(a, b):
    return abs(int(getattr(a, "frame_idx", 0)) - int(getattr(b, "frame_idx", 0)))


def _random_reorder_with_min_gap(frames, min_gap, seed):
    seq = list(frames)
    total = len(seq)
    if total <= 1 or int(min_gap) <= 0:
        return seq

    rng = random.Random(int(seed))
    attempts = max(60, min(400, total // 2))
    best_order = []

    for _ in range(int(attempts)):
        remaining = list(seq)
        rng.shuffle(remaining)
        order = [remaining.pop()]

        while remaining:
            last = order[-1]
            candidate_idx = [i for i, f in enumerate(remaining) if _frame_gap(last, f) >= int(min_gap)]
            if not candidate_idx:
                break
            pick_idx = candidate_idx[rng.randrange(len(candidate_idx))]
            order.append(remaining.pop(pick_idx))

        if remaining:
            rem = list(remaining)
            rng.shuffle(rem)
            progress = True
            while rem and progress:
                progress = False
                i = 0
                while i < len(rem):
                    frame = rem[i]
                    positions = list(range(len(order) + 1))
                    rng.shuffle(positions)
                    placed = False
                    for pos in positions:
                        prev_ok = (pos == 0) or (_frame_gap(order[pos - 1], frame) >= int(min_gap))
                        next_ok = (pos == len(order)) or (_frame_gap(frame, order[pos]) >= int(min_gap))
                        if prev_ok and next_ok:
                            order.insert(pos, frame)
                            rem.pop(i)
                            placed = True
                            progress = True
                            break
                    if not placed:
                        i += 1
            remaining = rem

        if not remaining:
            return order
        if len(order) > len(best_order):
            best_order = list(order)

    raise RuntimeError(
        "Cannot build random cmd publish order with required frame gap: "
        f"min_gap={int(min_gap)}, built={len(best_order)}/{total}. "
        "Please lower min_cmd_frame_gap or adjust CSV filtering."
    )


original_load_csv_frames = module._load_csv_frames


def _patched_load_csv_frames(*call_args, **call_kwargs):
    frames, empty_rows, incomplete_rows, controlled_motor_ids = original_load_csv_frames(*call_args, **call_kwargs)
    if not bool(RANDOM_CMD_PUBLISH):
        return frames, empty_rows, incomplete_rows, controlled_motor_ids

    seed = int(CMD_RANDOM_SEED)
    if seed <= 0:
        seed = int(time.time_ns() & 0x7FFFFFFF)
    min_gap = int(MIN_CMD_GAP_FRAMES)
    reordered = _random_reorder_with_min_gap(frames, min_gap=min_gap, seed=seed)
    print(
        "[csv-data-collection] cmd_publish_mode=random, "
        f"min_cmd_frame_gap={min_gap}, seed={seed}, total={len(reordered)}"
    )
    return reordered, empty_rows, incomplete_rows, controlled_motor_ids


module._load_csv_frames = _patched_load_csv_frames

original_execute_point = module.dcm._execute_point

def _patched_execute_point(*call_args, **call_kwargs):
    original_builder = module.dcm._build_idle_tor_matrix
    def _reel_idle_builder(defaults_hand):
        return [[0.0, 0.0, -0.08, 0.0, 0.0] for _ in range(module.dcm.MOTOR_COUNT)]
    module.dcm._build_idle_tor_matrix = _reel_idle_builder
    try:
        return original_execute_point(*call_args, **call_kwargs)
    finally:
        module.dcm._build_idle_tor_matrix = original_builder

module.dcm._execute_point = _patched_execute_point

def _force_arg(argv, flag, value):
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    out.extend([flag, value])
    return out

args = _force_arg(args, "--post-reach-pretighten-sec", "1.0")
sys.argv = [str(script_path)] + args
module.main()
"""


def _build_csv_collection_runtime_wrapper(
    random_cmd_publish: bool,
    min_cmd_frame_gap: int,
    cmd_random_seed: int,
) -> str:
    return (
        CSV_COLLECTION_REEL_WRAPPER_TEMPLATE
        .replace("__RANDOM_CMD_PUBLISH__", "True" if random_cmd_publish else "False")
        .replace("__MIN_CMD_GAP_FRAMES__", str(int(min_cmd_frame_gap)))
        .replace("__CMD_RANDOM_SEED__", str(int(cmd_random_seed)))
    )


def _find_workspace_root() -> Path:
    launch_file = Path(__file__).resolve()
    for parent in launch_file.parents:
        if (parent / "src" / "yhand_mapping").is_dir():
            return parent
    raise RuntimeError(f"Cannot locate ROS2 workspace root from launch file: {launch_file}")


def _load_yhand_dev_config(config_var_name: str) -> str:
    hand = {"config_dev1": "left", "config_dev2": "right"}[config_var_name]
    config_file = Path(__file__).resolve().parent.parent / "config" / f"{hand}_hand_device_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    lines = [f"PC_IP: {config['pc_ip']}", f"SN: {config['sn']}", "Boards:"]
    for board in config["boards"]:
        lines.extend([f"  - Id: {board['id']}", f"    IP: {board['ip']}"])
        if "port" in board:
            lines.append(f"    Port: {board['port']}")
    return "\n".join(lines) + "\n"


def _hand_runtime_info(hand: str):
    if hand == "left":
        return {
            "config_var": "config_dev1",
            "driver_node_name": "y_hand_left_csv_collection_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_csv_collection_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported target_hand={hand}, expected 'left' or 'right'.")


def _bool_from_launch_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean launch arg value: {value!r}")


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (workspace_root / p).resolve()


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    start_driver_node = _bool_from_launch_arg(LaunchConfiguration("start_driver_node").perform(context))
    start_glove_node = _bool_from_launch_arg(LaunchConfiguration("start_glove_node").perform(context))
    disable_glove_required = _bool_from_launch_arg(
        LaunchConfiguration("disable_glove_required").perform(context)
    )
    disable_glove_raw_required = _bool_from_launch_arg(
        LaunchConfiguration("disable_glove_raw_required").perform(context)
    )
    random_cmd_publish = _bool_from_launch_arg(
        LaunchConfiguration("random_cmd_publish").perform(context)
    )
    zero_calibrate_at_start = _bool_from_launch_arg(
        LaunchConfiguration("zero_calibrate_at_start").perform(context)
    )
    zero_between_csv = _bool_from_launch_arg(
        LaunchConfiguration("zero_between_csv").perform(context)
    )
    use_antagonist_release = _bool_from_launch_arg(
        LaunchConfiguration("use_antagonist_release").perform(context)
    )
    min_cmd_frame_gap = int(LaunchConfiguration("min_cmd_frame_gap").perform(context))
    if min_cmd_frame_gap < 0:
        raise ValueError("min_cmd_frame_gap must be >= 0.")
    cmd_random_seed = int(LaunchConfiguration("cmd_random_seed").perform(context))

    cmd_csv_path_raw = LaunchConfiguration("cmd_csv_path").perform(context).strip()
    finger_csv_path_raw = {
        finger: LaunchConfiguration(f"{finger}_cmd_csv_path").perform(context).strip()
        for finger in FINGER_ORDER
    }
    provided_finger_csvs = [
        finger for finger, raw_path in finger_csv_path_raw.items() if raw_path
    ]
    batch_mode = bool(provided_finger_csvs)
    if batch_mode:
        if cmd_csv_path_raw:
            raise ValueError(
                "cmd_csv_path cannot be combined with per-finger CSV path arguments."
            )
    elif not cmd_csv_path_raw:
        raise ValueError(
            "Provide cmd_csv_path for legacy single-CSV mode, or provide 1-5 "
            "<finger>_cmd_csv_path arguments."
        )

    info = _hand_runtime_info(hand)
    workspace_root = _find_workspace_root()
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"

    collect_script = scripts_dir / "data_collection_from_motor_feedback_csv.py"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    if not collect_script.exists():
        raise FileNotFoundError(f"data_collection_from_motor_feedback_csv.py not found: {collect_script}")
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")

    resolved_finger_csvs = {}
    if batch_mode:
        for finger in provided_finger_csvs:
            csv_path = _resolve_path(finger_csv_path_raw[finger], workspace_root)
            if not csv_path.exists():
                raise FileNotFoundError(f"{finger}_cmd_csv_path not found: {csv_path}")
            resolved_finger_csvs[finger] = csv_path
        cmd_csv_path = None
    else:
        cmd_csv_path = _resolve_path(cmd_csv_path_raw, workspace_root)
        if not cmd_csv_path.exists():
            raise FileNotFoundError(f"cmd_csv_path not found: {cmd_csv_path}")

    defaults_path_raw = LaunchConfiguration("defaults_path").perform(context).strip()
    output_dir_raw = LaunchConfiguration("output_dir").perform(context).strip()

    defaults_path = (
        _resolve_path(defaults_path_raw, workspace_root)
        if defaults_path_raw
        else (scripts_dir / "muscle_adjustment_defaults.py").resolve()
    )
    if not defaults_path.exists():
        raise FileNotFoundError(f"defaults_path not found: {defaults_path}")

    cmd_topic_override = LaunchConfiguration("cmd_topic").perform(context).strip()
    feedback_topic_override = LaunchConfiguration("feedback_topic").perform(context).strip()
    cmd_topic = cmd_topic_override if cmd_topic_override else info["command_topic"]
    feedback_topic = feedback_topic_override if feedback_topic_override else info["feedback_topic"]
    collection_wrapper = _build_csv_collection_runtime_wrapper(
        random_cmd_publish=random_cmd_publish,
        min_cmd_frame_gap=min_cmd_frame_gap,
        cmd_random_seed=cmd_random_seed,
    )

    collect_cmd = [
        "python3",
        "-u",
        "-c",
        collection_wrapper,
        str(collect_script),
        "--source-motor-map",
        LaunchConfiguration("source_motor_map").perform(context),
        "--hand",
        hand,
        "--cmd-topic",
        cmd_topic,
        "--feedback-topic",
        feedback_topic,
        "--glove-topic",
        LaunchConfiguration("glove_topic").perform(context),
        "--glove-raw-skeleton-topic",
        LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
        "--defaults-path",
        str(defaults_path),
        "--start-row",
        LaunchConfiguration("start_row").perform(context),
        "--stride",
        LaunchConfiguration("stride").perform(context),
        "--max-points",
        LaunchConfiguration("max_points").perform(context),
        "--collection-name",
        LaunchConfiguration("collection_name").perform(context),
        "--point-arrival-tol",
        LaunchConfiguration("point_arrival_tol").perform(context),
        "--arrival-confirm-sec",
        LaunchConfiguration("arrival_confirm_sec").perform(context),
        "--post-reach-pretighten-sec",
        LaunchConfiguration("post_reach_pretighten_sec").perform(context),
        "--result-window-sec",
        LaunchConfiguration("result_window_sec").perform(context),
        "--zero-calibrate-every-points",
        LaunchConfiguration("zero_calibrate_every_points").perform(context),
        "--zero-calibrate-at-start",
        "true" if zero_calibrate_at_start else "false",
        "--zero-between-csv",
        "true" if zero_between_csv else "false",
        "--use-antagonist-release",
        "true" if use_antagonist_release else "false",
    ]
    if batch_mode:
        for finger in provided_finger_csvs:
            collect_cmd.extend(
                [f"--{finger}-csv-path", str(resolved_finger_csvs[finger])]
            )
    else:
        collect_cmd.extend(["--cmd-csv-path", str(cmd_csv_path)])
    if output_dir_raw:
        collect_cmd.extend(["--output-dir", str(_resolve_path(output_dir_raw, workspace_root))])
    if disable_glove_required:
        collect_cmd.append("--disable-glove-required")
    if disable_glove_raw_required:
        collect_cmd.append("--disable-glove-raw-required")

    terminal_mode_raw = LaunchConfiguration("collect_in_terminal").perform(context).strip().lower()
    if terminal_mode_raw not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError("Invalid collect_in_terminal. Expected one of: auto/true/false")

    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gnome_terminal = shutil.which("gnome-terminal")

    if terminal_mode_raw == "auto":
        use_gnome_terminal = (gnome_terminal is not None) and display_available
    else:
        use_gnome_terminal = _bool_from_launch_arg(terminal_mode_raw)
        if use_gnome_terminal and gnome_terminal is None:
            raise RuntimeError(
                "collect_in_terminal=true but gnome-terminal not found. "
                "Please install gnome-terminal or set collect_in_terminal:=false."
            )
        if use_gnome_terminal and (not display_available):
            raise RuntimeError(
                "collect_in_terminal=true but DISPLAY/WAYLAND_DISPLAY is not available. "
                "Please set collect_in_terminal:=false to run without GUI terminal."
            )

    collect_process_cmd = collect_cmd
    collect_runner = "direct"
    if use_gnome_terminal:
        collect_process_cmd = [
            gnome_terminal,
            "--wait",
            "--title",
            f"csv_data_collection_{hand}",
            "--",
            *collect_cmd,
        ]
        collect_runner = "gnome-terminal"
    elif terminal_mode_raw == "auto":
        if gnome_terminal is None:
            collect_runner = "direct(auto:no-gnome-terminal)"
        elif not display_available:
            collect_runner = "direct(auto:no-display)"

    shutdown_on_collect_exit = _bool_from_launch_arg(
        LaunchConfiguration("shutdown_on_collect_exit").perform(context)
    )
    collect_kwargs = {
        "cmd": collect_process_cmd,
        "name": f"csv_data_collection_{hand}",
        "output": "screen",
    }
    if shutdown_on_collect_exit:
        collect_kwargs["on_exit"] = Shutdown(reason="csv collection launch is shutting down")
    collect_process = ExecuteProcess(**collect_kwargs)

    actions = [
        LogInfo(
            msg=(
                "[data_collection_from_motor_feedback_csv_launch] "
                f"target_hand={hand}, start_driver_node={start_driver_node}, "
                f"start_glove_node={start_glove_node}, disable_glove_required={disable_glove_required}, "
                f"disable_glove_raw_required={disable_glove_raw_required}, "
                f"mode={'per_finger_csv' if batch_mode else 'legacy_single_csv'}, "
                f"cmd_csv_path={cmd_csv_path}, "
                f"finger_csv_paths={resolved_finger_csvs if batch_mode else '{}'}, "
                f"start_row={LaunchConfiguration('start_row').perform(context)}, "
                f"stride={LaunchConfiguration('stride').perform(context)}, "
                f"max_points={LaunchConfiguration('max_points').perform(context)}, "
                f"random_cmd_publish={random_cmd_publish}, min_cmd_frame_gap={min_cmd_frame_gap}, "
                f"cmd_random_seed={cmd_random_seed}, "
                f"collection_name={LaunchConfiguration('collection_name').perform(context)}, "
                f"arrival_confirm_sec={LaunchConfiguration('arrival_confirm_sec').perform(context)}, "
                f"post_reach_pretighten_sec={LaunchConfiguration('post_reach_pretighten_sec').perform(context)}, "
                "post_reach_reel_forced=-0.08A@1.0s, "
                f"result_window_sec={LaunchConfiguration('result_window_sec').perform(context)}, "
                f"zero_between_csv={zero_between_csv}, "
                f"use_antagonist_release={use_antagonist_release}, "
                f"collect_runner={collect_runner}, shutdown_on_collect_exit={shutdown_on_collect_exit}, "
                f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}"
            )
        )
    ]

    if start_driver_node:
        dev_config = _load_yhand_dev_config(info["config_var"])
        driver_process = ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "devices_pkg",
                "Y_Hand_Node",
                info["driver_node_name"],
                info["feedback_topic"],
                info["command_topic"],
                dev_config,
            ],
            name=info["driver_node_name"],
            output="screen",
            on_exit=Shutdown(reason="csv collection launch is shutting down"),
        )
        actions.append(driver_process)

    if start_glove_node:
        glove_process = ExecuteProcess(
            cmd=["python3", "-u", str(glove_script)],
            name="glove_publisher",
            output="screen",
        )
        actions.append(glove_process)

    start_delay = float(LaunchConfiguration("control_script_start_delay_sec").perform(context))
    if start_driver_node:
        actions.append(TimerAction(period=start_delay, actions=[collect_process]))
    else:
        actions.append(collect_process)
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "source_motor_map", default_value="current", choices=["current", "legacy"],
                description="Motor numbering used in the source recording.",
            ),
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Target hand for CSV-driven full-hand data collection: left/right",
            ),
            DeclareLaunchArgument(
                "cmd_csv_path",
                default_value="",
                description=(
                    "Legacy single-CSV mode path (relative to workspace root or absolute). "
                    "Leave empty when any per-finger CSV paths are provided."
                ),
            ),
            *[
                DeclareLaunchArgument(
                    f"{finger}_cmd_csv_path",
                    default_value="",
                    description=(
                        f"Optional CSV path for {finger}. Provide any 1-5 per-finger paths."
                    ),
                )
                for finger in FINGER_ORDER
            ],
            DeclareLaunchArgument(
                "start_driver_node",
                default_value="true",
                description="Whether to start Y_Hand_Node in this launch: true/false",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Whether to start minimal_example_dual_stream_commented.py glove publisher: true/false",
            ),
            DeclareLaunchArgument(
                "disable_glove_required",
                default_value="false",
                description="Continue when glove data is not ready: true/false",
            ),
            DeclareLaunchArgument(
                "disable_glove_raw_required",
                default_value="false",
                description="Require glove48 but allow missing glove_raw_skeleton at startup: true/false",
            ),
            DeclareLaunchArgument(
                "random_cmd_publish",
                default_value="true",
                description="Randomly reorder CSV cmd publish sequence while still traversing all selected rows: true/false",
            ),
            DeclareLaunchArgument(
                "min_cmd_frame_gap",
                default_value="30",
                description="Minimum absolute frame_idx gap required between any two consecutive published cmd points.",
            ),
            DeclareLaunchArgument(
                "cmd_random_seed",
                default_value="0",
                description="Random seed for cmd publish order. <=0 means auto seed from current time.",
            ),
            DeclareLaunchArgument(
                "start_row",
                default_value="0",
                description="Start data row index in CSV (0-based).",
            ),
            DeclareLaunchArgument(
                "stride",
                default_value="1",
                description="Use every N-th row from CSV.",
            ),
            DeclareLaunchArgument(
                "max_points",
                default_value="0",
                description="Max points to execute. 0 means all selected rows.",
            ),
            DeclareLaunchArgument(
                "collection_name",
                default_value="csv_full_hand",
                description="Sub-folder name under <hand>_hand for this run.",
            ),
            DeclareLaunchArgument(
                "point_arrival_tol",
                default_value="7.0",
                description="Arrival tolerance in degrees for all active motors.",
            ),
            DeclareLaunchArgument(
                "arrival_confirm_sec",
                default_value="1.0",
                description=(
                    "Continuous duration (sec) that all controlled motors' errors must "
                    "stay <= point_arrival_tol to mark reached."
                ),
            ),
            DeclareLaunchArgument(
                "post_reach_pretighten_sec",
                default_value="2.0",
                description="Pretighten wait time (sec) after reached and before result recording starts.",
            ),
            DeclareLaunchArgument(
                "result_window_sec",
                default_value="1.0",
                description="Result averaging window length (sec) after post_reach_pretighten_sec.",
            ),
            DeclareLaunchArgument(
                "zero_calibrate_every_points",
                default_value="50",
                description="Periodic zero calibration interval. <=0 disables periodic calibration.",
            ),
            DeclareLaunchArgument(
                "zero_calibrate_at_start",
                default_value="false",
                description="Run one zero calibration before first CSV point: true/false",
            ),
            DeclareLaunchArgument(
                "zero_between_csv",
                default_value="true",
                description=(
                    "When multiple per-finger CSVs are provided, run zero calibration "
                    "between adjacent CSV files: true/false"
                ),
            ),
            DeclareLaunchArgument(
                "use_antagonist_release",
                default_value="false",
                description=(
                    "Enable legacy antagonist release/rewind timeline during point move: true/false. "
                    "Default false for pure 24-motor combination position-control."
                ),
            ),
            DeclareLaunchArgument(
                "cmd_topic",
                default_value="",
                description="Override command topic. Empty uses hand default.",
            ),
            DeclareLaunchArgument(
                "feedback_topic",
                default_value="",
                description="Override feedback topic. Empty uses hand default.",
            ),
            DeclareLaunchArgument(
                "glove_topic",
                default_value="/glove_joint_data",
                description="ROS topic for 48D glove data.",
            ),
            DeclareLaunchArgument(
                "glove_raw_skeleton_topic",
                default_value="/glove_joint_data_raw_skeleton",
                description="ROS topic for raw glove skeleton data.",
            ),
            DeclareLaunchArgument(
                "defaults_path",
                default_value="",
                description="Path to muscle_adjustment_defaults.py. Empty uses script default.",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="",
                description="Output directory. Empty uses script default.",
            ),
            DeclareLaunchArgument(
                "collect_in_terminal",
                default_value="auto",
                description="Run collection script in gnome-terminal (true/false/auto).",
            ),
            DeclareLaunchArgument(
                "shutdown_on_collect_exit",
                default_value="true",
                description="Shutdown whole launch when collection process exits: true/false",
            ),
            DeclareLaunchArgument(
                "control_script_start_delay_sec",
                default_value=str(CONTROL_SCRIPT_START_DELAY_SEC),
                description="Delay before collection script starts when driver node is enabled.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
