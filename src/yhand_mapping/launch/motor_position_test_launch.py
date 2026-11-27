import json
import math
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import ExecutableInPackage

CONTROL_SCRIPT_START_DELAY_SEC = 1.5
MOTOR_COUNT = 28


def _load_yhand_dev_config(config_var_name: str) -> str:
    hand = {"config_dev1": "left", "config_dev2": "right"}[config_var_name]
    config_file = Path(__file__).resolve().parent.parent / "config" / "devices" / f"{hand}_hand_device_config.json"
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
            "driver_node_name": "y_hand_left_position_test_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_position_test_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported target_hand={hand}, expected 'left' or 'right'.")


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    assist_enable = LaunchConfiguration("assist_enable").perform(context).strip().lower()
    info = _hand_runtime_info(hand)

    package_dir = Path(__file__).resolve().parent.parent
    test_script = package_dir / "scripts" / "motor_position_test.py"
    defaults_file = package_dir / "config" / "motor_parameter.json"

    for required_path, desc in (
        (test_script, "motor_position_test.py"),
        (defaults_file, "motor_parameter.json"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"{desc} not found: {required_path}")

    motor_parameters = json.loads(defaults_file.read_text(encoding="utf-8"))
    current_limit = motor_parameters.get("max_current_limit_a") if isinstance(motor_parameters, dict) else None
    if type(current_limit) not in (int, float) or not math.isfinite(current_limit) or current_limit <= 0:
        raise ValueError(f"{defaults_file}: max_current_limit_a must be a finite positive number")
    current_limits = json.dumps([float(current_limit)] * MOTOR_COUNT)
    dev_config = _load_yhand_dev_config(info["config_var"])

    driver_process = ExecuteProcess(
        cmd=[
            ExecutableInPackage(package="devices_pkg", executable="Y_Hand_Node"),
            info["driver_node_name"],
            info["feedback_topic"],
            info["command_topic"],
            dev_config,
            "--ros-args",
            "-p",
            f"max_current_limit:={current_limits}",
        ],
        name=info["driver_node_name"],
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    gnome_terminal = shutil.which("gnome-terminal")
    if gnome_terminal is None:
        raise RuntimeError(
            "gnome-terminal not found. Please install gnome-terminal, "
            "or run motor_position_test.py in a separate interactive terminal."
        )

    test_cmd = [
        "python3",
        "-u",
        str(test_script),
        "--hand",
        hand,
        "--defaults-path",
        str(defaults_file),
        "--cmd-topic",
        info["command_topic"],
        "--feedback-topic",
        info["feedback_topic"],
        "--assist-enable",
        assist_enable,
    ]

    test_process = ExecuteProcess(
        cmd=[
            gnome_terminal,
            "--wait",
            "--title",
            f"motor_position_test_{hand}",
            "--",
            *test_cmd,
        ],
        name=f"motor_position_test_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[motor_position_test_launch] target_hand={hand}, "
                f"driver_node={info['driver_node_name']}"
            )
        ),
        driver_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[test_process]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="left",
                description="Target hand for motor position test: left/right",
            ),
            DeclareLaunchArgument(
                "assist_enable",
                default_value="false",
                description="Legacy compatibility argument for single-motor tests.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
