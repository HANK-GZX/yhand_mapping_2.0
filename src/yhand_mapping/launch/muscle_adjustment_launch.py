import json
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction

# Manual hand selection (edit this value before launch): "left" or "right"
TARGET_HAND = "left"

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
            "driver_node_name": "y_hand_left_adjust_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_adjust_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported TARGET_HAND={hand}, expected 'left' or 'right'.")


def _launch_setup(context):
    del context

    hand = TARGET_HAND.strip().lower()
    info = _hand_runtime_info(hand)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    adjustment_script = scripts_dir / "muscle_adjustment.py"
    defaults_file = scripts_dir / "muscle_adjustment_defaults.py"
    excel_path = scripts_dir / "motor_test_logs" / f"muscle_adjustment_{hand}.xlsx"

    for required_path, desc in (
        (adjustment_script, "muscle_adjustment.py"),
        (defaults_file, "muscle_adjustment_defaults.py"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"{desc} not found: {required_path}")

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
            "or run muscle_adjustment.py in a separate interactive terminal."
        )

    adjustment_cmd = [
        "python3",
        "-u",
        str(adjustment_script),
        "--hand",
        hand,
        "--defaults-path",
        str(defaults_file),
        "--excel-path",
        str(excel_path),
        "--cmd-topic",
        info["command_topic"],
        "--feedback-topic",
        info["feedback_topic"],
    ]

    adjustment_process = ExecuteProcess(
        cmd=[
            gnome_terminal,
            "--wait",
            "--title",
            f"muscle_adjustment_{hand}",
            "--",
            *adjustment_cmd,
        ],
        name=f"muscle_adjustment_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[muscle_adjustment_launch] TARGET_HAND={hand}, "
                f"driver_node={info['driver_node_name']}, "
                "test_y_hand_node is intentionally not launched."
            )
        ),
        driver_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[adjustment_process]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            OpaqueFunction(function=_launch_setup),
        ]
    )
