import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from hand_motor_map import (
    model_motor_ids_csv,
)

CONTROL_SCRIPT_START_DELAY_SEC = 1.5
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")

# Runtime timing / queue defaults (single source of truth).
CONTROL_PERIOD_SEC_DEFAULT = "0.01"
STATUS_PERIOD_SEC_DEFAULT = "2.0"
GLOVE_ACCEPT_HZ_DEFAULT = ""
GLOVE_MAX_HZ_DEFAULT = "100.0"
GLOVE_SUB_QUEUE_SIZE_DEFAULT = "1"
GLOVE_PENDING_MAXLEN_DEFAULT = "10"
RIGHT_INDEX_Q_FROM_RAW_SKELETON_DEFAULT = "true"
GLOVE_RAW_SKELETON_TOPIC_DEFAULT = "/glove_joint_data_raw_skeleton"
RECORD_MOTOR_FEEDBACK_CSV_DEFAULT = "false"
MOTOR_FEEDBACK_CSV_PATH_DEFAULT = "auto"
MAPPING_IN_TERMINAL_DEFAULT = "auto"

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
MOTOR_CMD_OFFSET_MAP_DEFAULTS_JSON = json.dumps(MOTOR_CMD_OFFSET_MAP_DEFAULTS, ensure_ascii=False)

# Keep aligned with scripts/five_finger_delta_mapping_main.py defaults where possible.
FINGER_DEFAULTS_BY_HAND = {
    "left": {
        "thumb": {
            "enable": "true",
            "ckpt_rel_parts": ("mp", "runs", "thumb_1", "best_thumb_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('left', 'thumb'),
            "q_indices": "0,1,2,3",
            "mirror_right_from_left": "false",
        },
        # "index": {
        #     "enable": "true",
        #     "ckpt_rel_parts": ("mp", "runs", "index_delta_multihead", "best_index_delta_multihead.pt"),
        #     "motor_ids": model_motor_ids_csv("left", "index"),
        #     "q_indices": "4,5,6,7",
        #     "mirror_right_from_left": "false",
        # },
        "index": {
            "enable": "true",
            "ckpt_rel_parts": ("mp_right", "index_delta_multihead_masked", "best_index_delta_multihead_1.pt"),
            "motor_ids": model_motor_ids_csv('right', 'index'),
            "q_indices": "4,5,6,7",
            "mirror_right_from_left": "true",
        },
        "middle": {
            "enable": "true",
            "ckpt_rel_parts": ("mp", "runs", "middle_delta_multihead", "best_middle_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('left', 'middle'),
            "q_indices": "8,9,10,11",
            "mirror_right_from_left": "false",
        },
        "ring": {
            "enable": "true",
            "ckpt_rel_parts": ("mp", "runs", "ring_delta_multihead", "best_ring_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('left', 'ring'),
            "q_indices": "12,13,14,15",
            "mirror_right_from_left": "false",
        },
        "little": {
            "enable": "true",
            "ckpt_rel_parts": ("mp", "runs", "little_delta_multihead", "best_little_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('left', 'little'),
            "q_indices": "16,17,18,19",
            "mirror_right_from_left": "false",
        },
    },
    "right": {
        # "thumb": {
        #     "enable": "true",
        #     "ckpt_rel_parts": ("mp_right", "thumb_delta_multihead_masked","thumb_delta_multihead_masked","thumb_delta_multihead_masked_aug_1", "best_thumb_delta_multihead.pt"),
        #     "motor_ids": model_motor_ids_csv("right", "thumb"),
        #     "q_indices": "0,1,2,3",
        #     "mirror_right_from_left": "false",
        # },
        "thumb": {
            "enable": "true",
            "ckpt_rel_parts": ("mp", "runs","thumb_1", "best_thumb_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('left', 'thumb'),
            "q_indices": "0,1,2,3",
            "mirror_right_from_left": "true",
        },
        "index": {
            "enable": "true",
            "ckpt_rel_parts": ("mp_right", "index_delta_multihead_masked", "best_index_delta_multihead_1.pt"),
            "motor_ids": model_motor_ids_csv('right', 'index'),
            "q_indices": "4,5,6,7",
            "mirror_right_from_left": "false",
        },
        "middle": {
            "enable": "true",
            "ckpt_rel_parts": ("mp_right", "middle_delta_multihead_masked", "best_middle_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('right', 'middle'),
            "q_indices": "8,9,10,11",
            "mirror_right_from_left": "false",
        },
        "ring": {
            "enable": "true",
            "ckpt_rel_parts": ("mp_right", "ring_delta_multihead_masked", "best_ring_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('right', 'ring'),
            "q_indices": "12,13,14,15",
            "mirror_right_from_left": "false",
        },
        "little": {
            "enable": "true",
            "ckpt_rel_parts": ("mp_right", "little_delta_multihead_masked", "best_little_delta_multihead.pt"),
            "motor_ids": model_motor_ids_csv('right', 'little'),
            "q_indices": "16,17,18,19",
            "mirror_right_from_left": "false",
        },
    },
}

# Offsets currently consumed by five_finger_delta_mapping_main.py.
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
    # "right": {
    #     1: 0.0,
    #     2: -10,
    #     4: 5.0,
    #     6:-5.0,
    #     8: 4.0,
    #     11: -8.0,
    #     12: -5.0,
    #     13: -5.0,
    #     15: -10.0,
    #     16: -20.0,
    #     17: 5.0,
    #     19: -10,
    # },
    "right": {
        1: 0.0,
        2: 0.0,
        4: 0.0,
        6: 0.0,
        8: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        19: 0.0,
    },
}
Q_OFFSET_MAP_DEFAULTS_JSON = json.dumps(Q_OFFSET_MAP_DEFAULTS, ensure_ascii=False)


MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT = {6: (-50.0, 500.0),
 4: (-600.0, 50.0),
 27: (-150.0, 100.0),
 7: (-50.0, 500.0),
 8: (-100.0, 150.0),
 14: (-600.0, 50.0),
 3: (-600.0, 50.0),
 25: (-50.0, 500.0),
 28: (-50.0, 50.0),
 1: (-50.0, 50.0),
 15: (-600.0, 50.0),
 16: (-600.0, 50.0),
 23: (-50.0, 500.0),
 2: (-600.0, 50.0),
 24: (-50.0, 50.0),
 9: (-50.0, 50.0),
 17: (-600.0, 50.0),
 21: (-50.0, 50.0),
 10: (-50.0, 500.0),
 22: (-50.0, 50.0),
 13: (-600.0, 50.0),
 11: (-50.0, 500.0),
 20: (-50.0, 50.0),
 18: (-50.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT = {20: (-50.0, 500.0),
 18: (-400.0, 50.0),
 13: (-200.0, 400.0),
 21: (-50.0, 500.0),
 22: (-100.0, 400.0),
 28: (-400.0, 50.0),
 17: (-500.0, 50.0),
 11: (-50.0, 500.0),
 14: (-100.0, 100.0),
 15: (-100.0, 100.0),
 1: (-500.0, 50.0),
 2: (-500.0, 50.0),
 9: (-50.0, 500.0),
 16: (-500.0, 50.0),
 10: (-100.0, 100.0),
 23: (-100.0, 100.0),
 3: (-500.0, 50.0),
 7: (-50.0, 50.0),
 24: (-50.0, 500.0),
 8: (-50.0, 50.0),
 27: (-500.0, 50.0),
 25: (-50.0, 500.0),
 6: (-50.0, 50.0),
 4: (-50.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT_JSON = json.dumps(MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT, ensure_ascii=False)
MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT_JSON = json.dumps(MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT, ensure_ascii=False)


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
            "driver_node_name": "y_hand_left_five_finger_mapping_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_five_finger_mapping_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported hand={hand}, expected 'left' or 'right'.")


def _bool_from_launch_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean launch arg value: {value!r}")


def _resolve_finger_arg_or_default(context, hand: str, scripts_dir: Path, finger_name: str, key: str, launch_arg_name: str):
    raw = LaunchConfiguration(launch_arg_name).perform(context)
    val = str(raw).strip()
    if val:
        return val

    cfg = FINGER_DEFAULTS_BY_HAND[hand][finger_name]
    if key == "ckpt":
        return str(scripts_dir.joinpath(*cfg["ckpt_rel_parts"]))
    return str(cfg[key])


def _append_finger_args_to_mapping_cmd(mapping_cmd, context, hand: str, scripts_dir: Path):
    for finger_name in FINGER_ORDER:
        enable_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="enable",
            launch_arg_name=f"enable_{finger_name}",
        )
        ckpt_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="ckpt",
            launch_arg_name=f"{finger_name}_model_ckpt",
        )
        motor_ids_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="motor_ids",
            launch_arg_name=f"{finger_name}_model_motor_ids",
        )
        q_indices_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="q_indices",
            launch_arg_name=f"{finger_name}_q_indices",
        )
        mirror_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="mirror_right_from_left",
            launch_arg_name=f"{finger_name}_mirror_right_from_left",
        )
        mapping_cmd.extend(
            [
                f"--enable-{finger_name}",
                enable_val,
                f"--{finger_name}-model-ckpt",
                ckpt_val,
                f"--{finger_name}-model-motor-ids",
                motor_ids_val,
                f"--{finger_name}-q-indices",
                q_indices_val,
                f"--{finger_name}-mirror-right-from-left",
                mirror_val,
            ]
        )


def _declare_finger_launch_args():
    actions = []
    for finger_name in FINGER_ORDER:
        actions.extend(
            [
                # Empty means "use hand-specific default" (selected by target_hand).
                DeclareLaunchArgument(f"enable_{finger_name}", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_model_ckpt", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_model_motor_ids", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_q_indices", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_mirror_right_from_left", default_value=""),
            ]
        )
    return actions


def _launch_setup(context):
    target_hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    if target_hand == "both":
        hands = ("left", "right")
    elif target_hand in ("left", "right"):
        hands = (target_hand,)
    else:
        raise ValueError(f"Unsupported target_hand={target_hand}, expected 'left'|'right'|'both'.")

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    mapping_script = scripts_dir / "five_finger_delta_mapping_main.py"

    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")
    if not mapping_script.exists():
        raise FileNotFoundError(f"five_finger_delta_mapping_main.py not found: {mapping_script}")

    glove_process = ExecuteProcess(
        cmd=["python3", "-u", str(glove_script)],
        name="glove_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_glove_node")),
    )

    require_ready = _bool_from_launch_arg(LaunchConfiguration("require_ready").perform(context))
    terminal_mode_raw = LaunchConfiguration("mapping_in_terminal").perform(context).strip().lower()
    if terminal_mode_raw not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError("Invalid mapping_in_terminal. Expected one of: auto/true/false")
    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gnome_terminal = shutil.which("gnome-terminal")
    if terminal_mode_raw == "auto":
        use_gnome_terminal = (gnome_terminal is not None) and display_available
    else:
        use_gnome_terminal = _bool_from_launch_arg(terminal_mode_raw)
        if use_gnome_terminal and gnome_terminal is None:
            raise RuntimeError(
                "mapping_in_terminal=true but gnome-terminal not found. "
                "Please install gnome-terminal or set mapping_in_terminal:=false."
            )
        if use_gnome_terminal and (not display_available):
            raise RuntimeError(
                "mapping_in_terminal=true but DISPLAY/WAYLAND_DISPLAY is not available. "
                "Please set mapping_in_terminal:=false."
            )

    mapping_runner = "direct"
    if use_gnome_terminal:
        mapping_runner = "gnome-terminal"
    elif terminal_mode_raw == "auto":
        if gnome_terminal is None:
            mapping_runner = "direct(auto:no-gnome-terminal)"
        elif not display_available:
            mapping_runner = "direct(auto:no-display)"

    driver_processes = []
    mapping_processes = []
    hand_summaries = []

    for hand in hands:
        info = _hand_runtime_info(hand)
        dev_config = _load_yhand_dev_config(info["config_var"])

        driver_processes.append(
            ExecuteProcess(
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
                on_exit=Shutdown(reason="launch is shutting down"),
                condition=IfCondition(LaunchConfiguration("start_driver_node")),
            )
        )

        mapping_cmd = [
            "python3",
            "-u",
            str(mapping_script),
            "--hand",
            hand,
            "--cmd-topic",
            info["command_topic"],
            "--feedback-topic",
            info["feedback_topic"],
            "--glove-topic",
            LaunchConfiguration("glove_topic").perform(context),
            "--defaults-path",
            LaunchConfiguration("defaults_path").perform(context),
        ]
        glove_accept_hz = LaunchConfiguration("glove_accept_hz").perform(context).strip()
        if not glove_accept_hz:
            glove_accept_hz = LaunchConfiguration("glove_max_hz").perform(context).strip()
        _append_finger_args_to_mapping_cmd(mapping_cmd, context=context, hand=hand, scripts_dir=scripts_dir)
        mapping_cmd.extend(
            [
                "--control-period-sec",
                LaunchConfiguration("control_period_sec").perform(context),
                "--status-period-sec",
                LaunchConfiguration("status_period_sec").perform(context),
                "--glove-max-hz",
                glove_accept_hz,
                "--glove-sub-queue-size",
                LaunchConfiguration("glove_sub_queue_size").perform(context),
                "--glove-pending-maxlen",
                LaunchConfiguration("glove_pending_maxlen").perform(context),
                "--glove-raw-skeleton-topic",
                LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
                "--motor-feedback-csv-path",
                LaunchConfiguration("motor_feedback_csv_path").perform(context),
                "--motor-feedback-limits-left-json",
                LaunchConfiguration("motor_feedback_limits_left_json").perform(context),
                "--motor-feedback-limits-right-json",
                LaunchConfiguration("motor_feedback_limits_right_json").perform(context),
                "--motor-cmd-offset-map-json",
                LaunchConfiguration("motor_cmd_offset_map_json").perform(context),
                "--q-offset-map-json",
                LaunchConfiguration("q_offset_map_json").perform(context),
                "--device",
                LaunchConfiguration("device").perform(context),
                "--ready-timeout-sec",
                LaunchConfiguration("ready_timeout_sec").perform(context),
            ]
        )
        if _bool_from_launch_arg(LaunchConfiguration("right_index_q_from_raw_skeleton").perform(context)):
            mapping_cmd.append("--right-non-thumb-q-from-raw-skeleton")
        else:
            mapping_cmd.append("--no-right-non-thumb-q-from-raw-skeleton")
        if _bool_from_launch_arg(LaunchConfiguration("record_motor_feedback_csv").perform(context)):
            mapping_cmd.append("--record-motor-feedback-csv")
        if require_ready:
            mapping_cmd.append("--require-ready")

        mapping_process_cmd = mapping_cmd
        if use_gnome_terminal:
            mapping_process_cmd = [
                gnome_terminal,
                "--wait",
                "--title",
                f"five_finger_mapping_{hand}",
                "--",
                *mapping_cmd,
            ]

        mapping_processes.append(
            ExecuteProcess(
                cmd=mapping_process_cmd,
                name=f"five_finger_mapping_{hand}",
                output="screen",
                on_exit=Shutdown(reason="launch is shutting down"),
            )
        )

        thumb_ckpt = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name="thumb",
            key="ckpt",
            launch_arg_name="thumb_model_ckpt",
        )
        hand_summaries.append(f"{hand}:thumb_ckpt={thumb_ckpt}")

    return [
        LogInfo(
            msg=(
                f"[five_finger_mapping_launch] target_hand={target_hand}, hands={list(hands)}, "
                f"start_driver_node={LaunchConfiguration('start_driver_node').perform(context)}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"glove_accept_hz={LaunchConfiguration('glove_accept_hz').perform(context)}, "
                f"glove_max_hz={LaunchConfiguration('glove_max_hz').perform(context)}, "
                f"glove_sub_queue_size={LaunchConfiguration('glove_sub_queue_size').perform(context)}, "
                f"glove_pending_maxlen={LaunchConfiguration('glove_pending_maxlen').perform(context)}, "
                f"right_index_q_from_raw_skeleton={LaunchConfiguration('right_index_q_from_raw_skeleton').perform(context)}, "
                f"glove_raw_skeleton_topic={LaunchConfiguration('glove_raw_skeleton_topic').perform(context)}, "
                f"record_motor_feedback_csv={LaunchConfiguration('record_motor_feedback_csv').perform(context)}, "
                f"motor_feedback_csv_path={LaunchConfiguration('motor_feedback_csv_path').perform(context)}, "
                f"mapping_runner={mapping_runner}, "
                f"{'; '.join(hand_summaries)}"
            )
        ),
        *driver_processes,
        glove_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=mapping_processes),
    ]


def generate_launch_description():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"

    launch_args = [
        DeclareLaunchArgument(
            "target_hand",
            default_value="right",
            description="left | right | both",
        ),
        DeclareLaunchArgument("start_driver_node", default_value="true"),
        DeclareLaunchArgument("start_glove_node", default_value="true"),
        DeclareLaunchArgument("glove_topic", default_value="/glove_joint_data"),
        DeclareLaunchArgument(
            "defaults_path",
            default_value=str(scripts_dir / "muscle_adjustment_defaults.py"),
        ),
    ]

    launch_args.extend(_declare_finger_launch_args())
    launch_args.extend(
        [
            DeclareLaunchArgument("control_period_sec", default_value=CONTROL_PERIOD_SEC_DEFAULT),
            DeclareLaunchArgument("status_period_sec", default_value=STATUS_PERIOD_SEC_DEFAULT),
            DeclareLaunchArgument(
                "glove_accept_hz",
                default_value=GLOVE_ACCEPT_HZ_DEFAULT,
                description="Preferred glove acceptance rate Hz; if empty, fallback to glove_max_hz.",
            ),
            DeclareLaunchArgument("glove_max_hz", default_value=GLOVE_MAX_HZ_DEFAULT),
            DeclareLaunchArgument("glove_sub_queue_size", default_value=GLOVE_SUB_QUEUE_SIZE_DEFAULT),
            DeclareLaunchArgument("glove_pending_maxlen", default_value=GLOVE_PENDING_MAXLEN_DEFAULT),
            DeclareLaunchArgument(
                "right_index_q_from_raw_skeleton",
                default_value=RIGHT_INDEX_Q_FROM_RAW_SKELETON_DEFAULT,
                description="If true (default), right-hand non-thumb q_0..q_3 are solved from /glove_joint_data_raw_skeleton (176D); set false to use glove48 q indices.",
            ),
            DeclareLaunchArgument(
                "glove_raw_skeleton_topic",
                default_value=GLOVE_RAW_SKELETON_TOPIC_DEFAULT,
                description="Raw skeleton topic (Float64MultiArray, 176D one-hand or 352D two-hand payload).",
            ),
            DeclareLaunchArgument(
                "record_motor_feedback_csv",
                default_value=RECORD_MOTOR_FEEDBACK_CSV_DEFAULT,
                description="If true, record each motor feedback frame to CSV (24 motor pos columns).",
            ),
            DeclareLaunchArgument(
                "motor_feedback_csv_path",
                default_value=MOTOR_FEEDBACK_CSV_PATH_DEFAULT,
                description="Motor feedback CSV path. Supports placeholders: {hand}, {pid}, {timestamp}.",
            ),
            DeclareLaunchArgument(
                "mapping_in_terminal",
                default_value=MAPPING_IN_TERMINAL_DEFAULT,
                description="Run mapping script in gnome-terminal so r/p keyboard recording works: true/false/auto.",
            ),
            DeclareLaunchArgument(
                "motor_feedback_limits_left_json",
                default_value=MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT_JSON,
                description="Left-hand MOTOR_FEEDBACK_LIMITS JSON: {\"motor_id\": [lo, hi], ...}",
            ),
            DeclareLaunchArgument(
                "motor_feedback_limits_right_json",
                default_value=MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT_JSON,
                description="Right-hand MOTOR_FEEDBACK_LIMITS JSON: {\"motor_id\": [lo, hi], ...}",
            ),
            DeclareLaunchArgument(
                "motor_cmd_offset_map_json",
                default_value=MOTOR_CMD_OFFSET_MAP_DEFAULTS_JSON,
                description="Motor command offset map JSON by hand and motor_id, e.g. {\"left\": {\"8\": 5.0}, \"right\": {\"22\": 5.0}}",
            ),
            DeclareLaunchArgument(
                "q_offset_map_json",
                default_value=Q_OFFSET_MAP_DEFAULTS_JSON,
                description="Glove q offset map JSON by hand and q_index, e.g. {\"left\": {\"4\": 0.0}, \"right\": {\"8\": 10.0}}",
            ),
            DeclareLaunchArgument("device", default_value="auto"),
            DeclareLaunchArgument("require_ready", default_value="false"),
            DeclareLaunchArgument("ready_timeout_sec", default_value="12.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )

    return LaunchDescription(launch_args)
