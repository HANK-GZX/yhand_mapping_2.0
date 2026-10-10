import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

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
            "driver_node_name": "y_hand_left_data_collection_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_data_collection_node",
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
    target_finger = LaunchConfiguration("target_finger").perform(context).strip().lower()
    info = _hand_runtime_info(hand)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    collect_script = scripts_dir / "data_collection_main.py"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"

    if not collect_script.exists():
        raise FileNotFoundError(f"data_collection_main.py not found: {collect_script}")
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")

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

    collect_cmd = [
        "python3",
        "-u",
        str(collect_script),
        "--hand",
        hand,
        "--target-finger",
        target_finger,
        "--cmd-topic",
        info["command_topic"],
        "--feedback-topic",
        info["feedback_topic"],
        "--glove-topic",
        LaunchConfiguration("glove_topic").perform(context),
        "--glove-raw-skeleton-topic",
        LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
        "--manual-supplement-after-run",
        LaunchConfiguration("manual_supplement_after_run").perform(context),
    ]

    terminal_mode_raw = LaunchConfiguration("collect_in_terminal").perform(context).strip().lower()
    if terminal_mode_raw not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError(
            "Invalid collect_in_terminal. Expected one of: auto/true/false"
        )
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
            f"data_collection_{hand}",
            "--",
            *collect_cmd,
        ]
        collect_runner = "gnome-terminal"
    elif terminal_mode_raw == "auto":
        if gnome_terminal is None:
            collect_runner = "direct(auto:no-gnome-terminal)"
        elif not display_available:
            collect_runner = "direct(auto:no-display)"

    collect_process = ExecuteProcess(
        cmd=collect_process_cmd,
        name=f"data_collection_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[data_collection_launch] target_hand={hand}, "
                f"target_finger={target_finger}, "
                f"driver_node={info['driver_node_name']}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"manual_supplement_after_run={LaunchConfiguration('manual_supplement_after_run').perform(context)}, "
                f"collect_runner={collect_runner}"
            )
        ),
        driver_process,
        glove_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[collect_process]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="left",
                description="Target hand for data collection: left/right",
            ),
            DeclareLaunchArgument(
                "target_finger",
                default_value="all",
                description="Target finger to collect: all/thumb/index/middle/ring/little",
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
                "glove_raw_skeleton_topic",
                default_value="/glove_joint_data_raw_skeleton",
                description="ROS topic for raw glove skeleton data",
            ),
            DeclareLaunchArgument(
                "collect_in_terminal",
                default_value="true",
                description=(
                    "Run data_collection_main.py in gnome-terminal (true/false/auto). "
                    "Default true enables interactive manual supplement after auto collection."
                ),
            ),
            DeclareLaunchArgument(
                "manual_supplement_after_run",
                default_value="true",
                description="Enter manual supplement loop after auto collection (true/false).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
