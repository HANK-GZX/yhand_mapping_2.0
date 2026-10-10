import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration

CONTROL_SCRIPT_START_DELAY_SEC = 1.5

# Left-hand hard isolation
HAND = "left"
DRIVER_NODE_NAME = "y_hand_left_data_collection_node"
FEEDBACK_TOPIC = "/yhand_left_feedback"
COMMAND_TOPIC = "/yhand_left_cmd"


def _load_hand_device_config() -> str:
    config_file = Path(__file__).resolve().parent.parent / "config" / f"{HAND}_hand_device_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    lines = [f"PC_IP: {config['pc_ip']}", f"SN: {config['sn']}", "Boards:"]
    for board in config["boards"]:
        lines.extend([f"  - Id: {board['id']}", f"    IP: {board['ip']}"])
        if "port" in board:
            lines.append(f"    Port: {board['port']}")
    return "\n".join(lines) + "\n"


def _bool_from_launch_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean launch arg value: {value!r}")


def _launch_setup(context):
    target_finger = LaunchConfiguration("target_finger").perform(context).strip().lower()
    start_glove_node = _bool_from_launch_arg(LaunchConfiguration("start_glove_node").perform(context))
    disable_glove_required = _bool_from_launch_arg(
        LaunchConfiguration("disable_glove_required").perform(context)
    )

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    collect_script = scripts_dir / "data_collection_main.py"
    monitor_script = scripts_dir / "data_collection_health_monitor.py"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"

    if not collect_script.exists():
        raise FileNotFoundError(f"data_collection_main.py not found: {collect_script}")
    if not monitor_script.exists():
        raise FileNotFoundError(f"data_collection_health_monitor.py not found: {monitor_script}")
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")

    runtime_dir = scripts_dir / "motor_test_logs" / "data_collection"
    resume_file = LaunchConfiguration("resume_file").perform(context).strip()
    if not resume_file:
        resume_file = str(runtime_dir / f"{HAND}_resume_latest.json")
    live_status_file = LaunchConfiguration("live_status_file").perform(context).strip()
    if not live_status_file:
        live_status_file = str(runtime_dir / f"{HAND}_live_status.json")

    driver_process = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "devices_pkg",
            "Y_Hand_Node",
            DRIVER_NODE_NAME,
            FEEDBACK_TOPIC,
            COMMAND_TOPIC,
            _load_hand_device_config(),
        ],
        name=DRIVER_NODE_NAME,
        output="screen",
        on_exit=Shutdown(reason="left launch is shutting down"),
    )
    glove_process = ExecuteProcess(
        cmd=[
            "python3",
            "-u",
            str(glove_script),
        ],
        name="glove_publisher",
        output="screen",
    )

    collect_cmd = [
        "python3",
        "-u",
        str(collect_script),
        "--run-mode",
        "controller",
        "--hand",
        HAND,
        "--target-finger",
        target_finger,
        "--cmd-topic",
        COMMAND_TOPIC,
        "--feedback-topic",
        FEEDBACK_TOPIC,
        "--glove-topic",
        LaunchConfiguration("glove_topic").perform(context),
        "--glove-raw-skeleton-topic",
        LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
        "--manual-supplement-after-run",
        LaunchConfiguration("manual_supplement_after_run").perform(context),
        "--resume",
        LaunchConfiguration("resume").perform(context),
        "--resume-file",
        resume_file,
        "--live-status-file",
        live_status_file,
    ]
    if disable_glove_required:
        collect_cmd.append("--disable-glove-required")

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
            "data_collection_left",
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
        name="data_collection_left",
        output="screen",
        on_exit=Shutdown(reason="left launch is shutting down"),
    )

    start_health_monitor = _bool_from_launch_arg(LaunchConfiguration("start_health_monitor").perform(context))
    monitor_process = None
    monitor_runner = "disabled"
    if start_health_monitor:
        monitor_cmd = [
            "python3",
            "-u",
            str(monitor_script),
            "--status-file",
            live_status_file,
            "--label",
            "left-health",
            "--refresh-sec",
            LaunchConfiguration("health_refresh_sec").perform(context),
        ]

        monitor_terminal_mode = LaunchConfiguration("health_monitor_in_terminal").perform(context).strip().lower()
        if monitor_terminal_mode not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
            raise ValueError("Invalid health_monitor_in_terminal. Expected one of: auto/true/false")

        if monitor_terminal_mode == "auto":
            use_monitor_terminal = (gnome_terminal is not None) and display_available
        else:
            use_monitor_terminal = _bool_from_launch_arg(monitor_terminal_mode)
            if use_monitor_terminal and gnome_terminal is None:
                raise RuntimeError(
                    "health_monitor_in_terminal=true but gnome-terminal not found."
                )
            if use_monitor_terminal and (not display_available):
                raise RuntimeError(
                    "health_monitor_in_terminal=true but DISPLAY/WAYLAND_DISPLAY is not available."
                )

        monitor_cmd_run = monitor_cmd
        if use_monitor_terminal:
            monitor_cmd_run = [
                gnome_terminal,
                "--title",
                "data_collection_left_health",
                "--",
                *monitor_cmd,
            ]
            monitor_runner = "gnome-terminal"
        elif monitor_terminal_mode == "auto":
            if gnome_terminal is None:
                monitor_runner = "direct(auto:no-gnome-terminal)"
            elif not display_available:
                monitor_runner = "direct(auto:no-display)"
            else:
                monitor_runner = "direct"
        else:
            monitor_runner = "direct"

        monitor_process = ExecuteProcess(
            cmd=monitor_cmd_run,
            name="data_collection_left_health",
            output="screen",
        )

    actions = [
        LogInfo(
            msg=(
                "[data_collection_left_independent] "
                f"hand={HAND}, target_finger={target_finger}, "
                f"driver_node={DRIVER_NODE_NAME}, "
                f"cmd_topic={COMMAND_TOPIC}, feedback_topic={FEEDBACK_TOPIC}, "
                f"start_glove_node={start_glove_node}, "
                f"disable_glove_required={disable_glove_required}, "
                f"collect_runner={collect_runner}, monitor_runner={monitor_runner}, "
                f"resume_file={resume_file}, live_status_file={live_status_file}"
            )
        )
    ]
    if monitor_process is not None:
        actions.append(monitor_process)
    if start_glove_node:
        actions.append(glove_process)
    actions.extend([
        driver_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=[collect_process]),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_finger",
                default_value="all",
                description="Target finger to collect: all/thumb/index/middle/ring/little",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Start glove publisher (minimal_example_dual_stream_commented.py).",
            ),
            DeclareLaunchArgument(
                "disable_glove_required",
                default_value="true",
                description="Allow collection startup even if glove topics are temporarily not ready.",
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
                description="Run data_collection_main.py in gnome-terminal (true/false/auto).",
            ),
            DeclareLaunchArgument(
                "manual_supplement_after_run",
                default_value="true",
                description="Enter manual supplement loop after auto collection (true/false).",
            ),
            DeclareLaunchArgument(
                "resume",
                default_value="false",
                description="Resume from checkpoint (true/false).",
            ),
            DeclareLaunchArgument(
                "resume_file",
                default_value="",
                description="Resume checkpoint json path. Empty means default under motor_test_logs/data_collection.",
            ),
            DeclareLaunchArgument(
                "live_status_file",
                default_value="",
                description="Live health status json path. Empty means default under motor_test_logs/data_collection.",
            ),
            DeclareLaunchArgument(
                "start_health_monitor",
                default_value="false",
                description="Start live health monitor process (true/false).",
            ),
            DeclareLaunchArgument(
                "health_monitor_in_terminal",
                default_value="auto",
                description="Run health monitor in gnome-terminal (true/false/auto).",
            ),
            DeclareLaunchArgument(
                "health_refresh_sec",
                default_value="0.5",
                description="Health monitor refresh interval in seconds.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
