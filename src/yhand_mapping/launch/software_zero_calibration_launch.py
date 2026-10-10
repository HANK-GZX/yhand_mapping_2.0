import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration


HAND_CHOICES = ("left", "right", "both")
MODE_CHOICES = ("collect-reference", "calibrate-auto", "show-reference", "show-state")
TERMINAL_ARG_VALUES = ("auto", "1", "0", "true", "false", "yes", "no", "on", "off")
RAW_CMD_TOPIC_TEMPLATE = "/yhand_{hand}_raw_cmd"
RAW_FEEDBACK_TOPIC_TEMPLATE = "/yhand_{hand}_raw_feedback"


def _bool_from_launch_arg(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid boolean launch arg: {value!r}")


def _declare_arg(name: str, default_value: str, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default_value, description=description)


def _resolve_path(raw: str, base: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _find_project_root(launch_file: Path) -> Path:
    source_like = launch_file.parent.parent
    if (source_like / "scripts").is_dir() and (source_like / "config").is_dir():
        return source_like
    for parent in launch_file.parents:
        candidate = parent / "src" / "yhand_mapping"
        if (candidate / "scripts").is_dir() and (candidate / "config").is_dir():
            return candidate
    return source_like


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"top-level JSON must be object: {path}")
    return obj


def _select_hand_cfg(root_cfg: dict, hand: str) -> dict:
    hands = root_cfg.get("hands")
    if isinstance(hands, dict):
        hand_cfg = hands.get(hand)
        if not isinstance(hand_cfg, dict):
            raise ValueError(f"config missing hands.{hand}")
        common = {k: v for k, v in root_cfg.items() if k not in ("hands", "default_hand")}
        out = dict(common)
        out.update(hand_cfg)
        return out
    return dict(root_cfg)


def _build_dev_config_text(cfg: dict) -> str:
    pc_ip = str(cfg.get("pc_ip", "")).strip()
    sn = str(cfg.get("sn", "")).strip()
    boards = cfg.get("boards", [])
    if not pc_ip:
        raise ValueError("config missing `pc_ip`")
    if not sn:
        raise ValueError("config missing `sn`")
    if not isinstance(boards, list) or not boards:
        raise ValueError("config missing non-empty `boards` list")

    lines = [f"PC_IP: {pc_ip}", f"SN: {sn}", "Boards:"]
    for board in boards:
        if not isinstance(board, dict):
            raise ValueError("each item in `boards` must be an object")
        board_id = board.get("id")
        board_ip = board.get("ip")
        board_port = board.get("port", None)
        if board_id is None or board_ip is None:
            raise ValueError("board item must include `id` and `ip`")
        lines.append(f"  - Id: {int(board_id)}")
        lines.append(f"    IP: {str(board_ip)}")
        if board_port is not None and str(board_port).strip() != "":
            lines.append(f"    Port: {int(board_port)}")
    return "\n".join(lines)


def _raw_topics(hand: str, cmd_override: str, feedback_override: str) -> tuple[str, str]:
    cmd_topic = cmd_override.strip() or RAW_CMD_TOPIC_TEMPLATE.format(hand=hand)
    feedback_topic = feedback_override.strip() or RAW_FEEDBACK_TOPIC_TEMPLATE.format(hand=hand)
    return cmd_topic, feedback_topic


def _launch_setup(context):
    launch_file = Path(__file__).resolve()
    project_root = _find_project_root(launch_file)
    scripts_dir = project_root / "scripts"

    hand = LaunchConfiguration("hand_side").perform(context).strip().lower()
    if hand not in HAND_CHOICES:
        raise ValueError("hand_side must be left, right, or both")
    target_hands = ["left", "right"] if hand == "both" else [hand]

    mode = LaunchConfiguration("mode").perform(context).strip()
    if mode not in MODE_CHOICES:
        raise ValueError("mode must be one of collect-reference/calibrate-auto/show-reference/show-state")

    start_driver = _bool_from_launch_arg(LaunchConfiguration("start_driver").perform(context))
    start_delay_sec = float(LaunchConfiguration("calibration_start_delay_sec").perform(context))
    in_terminal_raw = LaunchConfiguration("calibration_in_terminal").perform(context).strip().lower()
    if in_terminal_raw not in TERMINAL_ARG_VALUES:
        raise ValueError("Invalid calibration_in_terminal. Expected one of: auto/true/false")

    device_config_file = _resolve_path(LaunchConfiguration("device_config_file").perform(context), project_root)
    zero_config_file = _resolve_path(LaunchConfiguration("zero_config_file").perform(context), project_root)
    defaults_path = _resolve_path(LaunchConfiguration("defaults_path").perform(context), project_root)
    cmd_topic_override = LaunchConfiguration("cmd_topic").perform(context)
    feedback_topic_override = LaunchConfiguration("feedback_topic").perform(context)
    if hand == "both" and (cmd_topic_override.strip() or feedback_topic_override.strip()):
        raise ValueError("hand_side:=both does not support single cmd_topic/feedback_topic override")
    cmd_topic, feedback_topic = (
        ("/yhand_<hand>_raw_cmd", "/yhand_<hand>_raw_feedback")
        if hand == "both"
        else _raw_topics(hand, cmd_topic_override, feedback_topic_override)
    )
    reference_path = LaunchConfiguration("reference_path").perform(context).strip()
    state_path = LaunchConfiguration("state_path").perform(context).strip()

    calibration_script = scripts_dir / "software_zero_calibration.py"
    if not calibration_script.exists():
        raise FileNotFoundError(f"software_zero_calibration.py not found: {calibration_script}")
    if not defaults_path.exists():
        raise FileNotFoundError(f"defaults file not found: {defaults_path}")
    if not zero_config_file.exists():
        raise FileNotFoundError(f"software-zero config file not found: {zero_config_file}")

    actions = [
        LogInfo(
            msg=(
                "[software_zero_calibration_launch] "
                f"mode={mode}, hand={hand}, start_driver={start_driver}, "
                f"device_config={device_config_file}, zero_config={zero_config_file}, "
                f"defaults_path={defaults_path}, cmd_topic={cmd_topic}, feedback_topic={feedback_topic}, "
                f"calibration_in_terminal={in_terminal_raw}"
            )
        )
    ]

    if start_driver and mode not in ("show-reference", "show-state"):
        root_cfg = _load_json(device_config_file)
        for driver_hand in target_hands:
            hand_cfg = _select_hand_cfg(root_cfg, driver_hand)
            driver_cmd_topic, driver_feedback_topic = _raw_topics(driver_hand, "", "")
            dev_config_text = _build_dev_config_text(hand_cfg)
            driver_node_name = f"y_hand_{driver_hand}_software_zero_raw_node"
            actions.append(
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "run",
                        "devices_pkg",
                        "Y_Hand_Node",
                        driver_node_name,
                        driver_feedback_topic,
                        driver_cmd_topic,
                        dev_config_text,
                    ],
                    name=driver_node_name,
                    output="screen",
                    on_exit=Shutdown(reason="software_zero_calibration_launch is shutting down"),
                )
            )

    calibration_cmd = [
        "python3",
        "-u",
        str(calibration_script),
        "--mode",
        mode,
        "--hand",
        hand,
        "--config",
        str(zero_config_file),
        "--defaults-path",
        str(defaults_path),
    ]
    if hand != "both":
        calibration_cmd.extend(["--cmd-topic", cmd_topic, "--feedback-topic", feedback_topic])
    if reference_path:
        calibration_cmd.extend(["--reference-path", str(_resolve_path(reference_path, project_root))])
    if state_path:
        calibration_cmd.extend(["--state-path", str(_resolve_path(state_path, project_root))])

    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gnome_terminal = shutil.which("gnome-terminal")
    if in_terminal_raw == "auto":
        use_gnome_terminal = (gnome_terminal is not None) and display_available
    else:
        use_gnome_terminal = _bool_from_launch_arg(in_terminal_raw)
        if use_gnome_terminal and gnome_terminal is None:
            raise RuntimeError("calibration_in_terminal=true but gnome-terminal not found.")
        if use_gnome_terminal and (not display_available):
            raise RuntimeError("calibration_in_terminal=true but DISPLAY/WAYLAND_DISPLAY is unavailable.")
    if use_gnome_terminal:
        calibration_cmd = [
            gnome_terminal,
            "--wait",
            "--title",
            f"software_zero_calibration_{hand}",
            "--",
            *calibration_cmd,
        ]

    calibration_process = ExecuteProcess(
        cmd=calibration_cmd,
        name=f"software_zero_calibration_{hand}",
        output="screen",
        on_exit=Shutdown(reason="software_zero_calibration_launch is shutting down"),
    )
    if start_driver and mode not in ("show-reference", "show-state") and start_delay_sec > 0.0:
        actions.append(TimerAction(period=start_delay_sec, actions=[calibration_process]))
    else:
        actions.append(calibration_process)
    return actions


def generate_launch_description():
    launch_file = Path(__file__).resolve()
    project_root = _find_project_root(launch_file)
    default_device_config = project_root / "config" / "bimanual_hand_device_config.json"
    default_zero_config = project_root / "config" / "software_zero_calibration_config.json"
    default_defaults_path = project_root / "scripts" / "muscle_adjustment_defaults.py"

    return LaunchDescription(
        [
            _declare_arg("mode", "show-reference", "collect-reference/calibrate-auto/show-reference/show-state"),
            _declare_arg("hand_side", "right", "Target hand: left/right/both."),
            _declare_arg("start_driver", "true", "Start raw Y_Hand_Node driver before calibration."),
            _declare_arg("calibration_start_delay_sec", "1.0", "Delay calibration startup after raw driver startup."),
            _declare_arg("device_config_file", str(default_device_config), "Bimanual device config JSON."),
            _declare_arg("zero_config_file", str(default_zero_config), "Software-zero config JSON."),
            _declare_arg("defaults_path", str(default_defaults_path), "Path to muscle_adjustment_defaults.py."),
            _declare_arg("cmd_topic", "", "Raw command topic override."),
            _declare_arg("feedback_topic", "", "Raw feedback topic override."),
            _declare_arg("reference_path", "", "Back-delta reference JSON path override."),
            _declare_arg("state_path", "", "Back-delta state JSON path override."),
            _declare_arg("calibration_in_terminal", "auto", "Run calibration script in gnome-terminal (true/false/auto)."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
