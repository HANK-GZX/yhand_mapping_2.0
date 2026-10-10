import json
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
            "driver_node_name": "y_hand_left_thumb_delta_multihead_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_thumb_delta_multihead_node",
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


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    info = _hand_runtime_info(hand)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    mapping_script = scripts_dir / "thumb_delta_multihead_mapping_main.py"
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")
    if not mapping_script.exists():
        raise FileNotFoundError(f"thumb_delta_multihead_mapping_main.py not found: {mapping_script}")

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
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    glove_process = ExecuteProcess(
        cmd=[
            "python3",
            "-u",
            str(glove_script),
        ],
        name="glove_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_glove_node")),
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
        "--model-ckpt",
        LaunchConfiguration("model_ckpt").perform(context),
        "--model-motor-ids",
        LaunchConfiguration("model_motor_ids").perform(context),
        "--q-indices",
        LaunchConfiguration("q_indices").perform(context),
        "--mirror-right-from-left",
        LaunchConfiguration("mirror_right_from_left").perform(context),
        "--defaults-path",
        LaunchConfiguration("defaults_path").perform(context),
        "--control-period-sec",
        LaunchConfiguration("control_period_sec").perform(context),
        "--status-period-sec",
        LaunchConfiguration("status_period_sec").perform(context),
        "--index-adduction-q-local-index",
        LaunchConfiguration("index_adduction_q_local_index").perform(context),
        "--index-adduction-q-offset",
        LaunchConfiguration("index_adduction_q_offset").perform(context),
        "--right-q8-offset",
        LaunchConfiguration("right_q8_offset").perform(context),
        "--right-q12-offset",
        LaunchConfiguration("right_q12_offset").perform(context),
        "--device",
        LaunchConfiguration("device").perform(context),
        "--ready-timeout-sec",
        LaunchConfiguration("ready_timeout_sec").perform(context),
    ]

    require_ready = _bool_from_launch_arg(LaunchConfiguration("require_ready").perform(context))
    if require_ready:
        mapping_cmd.append("--require-ready")

    mapping_process = ExecuteProcess(
        cmd=mapping_cmd,
        name=f"thumb_delta_multihead_mapping_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[thumb_delta_multihead_mapping_launch] target_hand={hand}, "
                f"driver_node={info['driver_node_name']}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"glove_topic={LaunchConfiguration('glove_topic').perform(context)}, "
                f"model_ckpt={LaunchConfiguration('model_ckpt').perform(context)}, "
                f"model_motor_ids={LaunchConfiguration('model_motor_ids').perform(context)}, "
                f"q_indices={LaunchConfiguration('q_indices').perform(context)}, "
                f"mirror_right_from_left={LaunchConfiguration('mirror_right_from_left').perform(context)}"
            )
        ),
        driver_process,
        glove_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[mapping_process]),
    ]


def generate_launch_description():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="left",
                description="Target hand for mapping control: left/right",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Start minimal_example_dual_stream_commented.py glove publisher node",
            ),
            DeclareLaunchArgument(
                "glove_topic",
                default_value="/glove_joint_data",
                description="ROS topic for 48D glove data",
            ),
            DeclareLaunchArgument(
                "model_ckpt",
                default_value=str(scripts_dir / "mp_test" / "thumb" / "thumb_delta_multihead" / "best_thumb_delta_multihead.pt"),
                description="Thumb delta multihead checkpoint path",
            ),
            DeclareLaunchArgument(
                "model_motor_ids",
                default_value=model_motor_ids_csv('left', 'thumb'),
                description="Thumb output motor ids (left-hand order, m_0..m_5)",
            ),
            DeclareLaunchArgument(
                "q_indices",
                default_value="0,1,2,3",
                description="Thumb glove20 input indices (hand-local)",
            ),
            DeclareLaunchArgument(
                "mirror_right_from_left",
                default_value="true",
                description="When target_hand=right, mirror left-hand motor ids to right-hand counterparts",
            ),
            DeclareLaunchArgument(
                "defaults_path",
                default_value=str(scripts_dir / "muscle_adjustment_defaults.py"),
                description="Path to muscle_adjustment_defaults.py",
            ),
            DeclareLaunchArgument(
                "control_period_sec",
                default_value="0.01",
                description="Command publish period (sec)",
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="2.0",
                description="Status print period (sec)",
            ),
            DeclareLaunchArgument(
                "index_adduction_q_local_index",
                default_value="4",
                description="Hand-local glove20 q index for index adduction input",
            ),
            DeclareLaunchArgument(
                "index_adduction_q_offset",
                default_value="15.0",
                description="Additive glove-q offset for index adduction input (hand-local q_4) before model inference",
            ),
            DeclareLaunchArgument(
                "right_q8_offset",
                default_value="10.0",
                description="Additive glove-q offset for right-hand q_8 before model inference",
            ),
            DeclareLaunchArgument(
                "right_q12_offset",
                default_value="-10.0",
                description="Additive glove-q offset for right-hand q_12 before model inference",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="auto",
                description="Torch device: auto/cpu/cuda",
            ),
            DeclareLaunchArgument(
                "require_ready",
                default_value="false",
                description="Exit if glove+feedback are not ready before timeout",
            ),
            DeclareLaunchArgument(
                "ready_timeout_sec",
                default_value="12.0",
                description="Startup ready timeout in seconds",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
