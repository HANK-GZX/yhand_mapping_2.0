import json
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration

# Optional fixed action: "", "0", "1", "2", "3".
# Empty string means interactive selection in terminal.
ACTION_INDEX_DEFAULT = ""

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
            "driver_node_name": "y_hand_left_manus_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_manus_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported target_hand={hand}, expected 'left' or 'right'.")


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    action_index = LaunchConfiguration("action_index").perform(context).strip()
    info = _hand_runtime_info(hand)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    calibration_script = scripts_dir / "manus_calibration.py"

    if not calibration_script.exists():
        raise FileNotFoundError(f"manus_calibration.py not found: {calibration_script}")

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

    gnome_terminal = shutil.which("gnome-terminal")
    if gnome_terminal is None:
        raise RuntimeError(
            "gnome-terminal not found. Please install gnome-terminal, "
            "or run manus_calibration.py in a separate interactive terminal."
        )

    calibration_cmd = [
        "python3",
        "-u",
        str(calibration_script),
        "--hand",
        hand,
        "--cmd-topic",
        info["command_topic"],
        "--feedback-topic",
        info["feedback_topic"],
    ]
    if action_index in {"0", "1", "2", "3"}:
        calibration_cmd.extend(["--action", action_index])

    calibration_process = ExecuteProcess(
        cmd=[
            gnome_terminal,
            "--wait",
            "--title",
            f"manus_calibration_{hand}",
            "--",
            *calibration_cmd,
        ],
        name=f"manus_calibration_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[manus_calibration_launch] target_hand={hand}, action_index={action_index!r}, "
                f"driver_node={info['driver_node_name']}"
            )
        ),
        driver_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[calibration_process]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="left",
                description="Target hand for manus calibration: left/right",
            ),
            DeclareLaunchArgument(
                "action_index",
                default_value=ACTION_INDEX_DEFAULT,
                description='Optional fixed action index: "", "0", "1", "2", or "3".',
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
