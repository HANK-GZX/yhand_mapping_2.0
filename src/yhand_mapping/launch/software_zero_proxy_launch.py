import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration

HAND_CHOICES = ("left", "right")
RAW_CMD_TOPIC_TEMPLATE = "/yhand_{hand}_raw_cmd"
RAW_FEEDBACK_TOPIC_TEMPLATE = "/yhand_{hand}_raw_feedback"
CMD_TOPIC_TEMPLATE = "/yhand_{hand}_cmd"
FEEDBACK_TOPIC_TEMPLATE = "/yhand_{hand}_feedback"


# ---------------------------------------------------------------------------
# Launch/config helpers
# ---------------------------------------------------------------------------


def _bool_from_launch_arg(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid boolean launch arg: {value!r}")


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


def _hands_from_arg(value: str) -> list[str]:
    v = str(value).strip().lower()
    if v == "both":
        return ["left", "right"]
    if v in HAND_CHOICES:
        return [v]
    raise ValueError("hand_side must be left/right/both")


def _hand_topics(hand: str) -> tuple[str, str, str, str]:
    return (
        RAW_CMD_TOPIC_TEMPLATE.format(hand=hand),
        RAW_FEEDBACK_TOPIC_TEMPLATE.format(hand=hand),
        CMD_TOPIC_TEMPLATE.format(hand=hand),
        FEEDBACK_TOPIC_TEMPLATE.format(hand=hand),
    )


def _declare_arg(name: str, default_value: str, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default_value, description=description)


# ---------------------------------------------------------------------------
# Launch assembly
# ---------------------------------------------------------------------------


def _launch_setup(context):
    launch_file = Path(__file__).resolve()
    project_root = _find_project_root(launch_file)
    scripts_dir = project_root / "scripts"

    config_file = _resolve_path(LaunchConfiguration("config_file").perform(context), project_root)
    zero_config_file = _resolve_path(LaunchConfiguration("zero_config_file").perform(context), project_root)
    hand_side = LaunchConfiguration("hand_side").perform(context)
    start_driver = _bool_from_launch_arg(LaunchConfiguration("start_driver").perform(context))
    start_proxy = _bool_from_launch_arg(LaunchConfiguration("start_proxy").perform(context))
    reload_state_period_sec = LaunchConfiguration("reload_state_period_sec").perform(context).strip()
    proxy_start_delay_sec = float(LaunchConfiguration("proxy_start_delay_sec").perform(context))

    root_cfg = _load_json(config_file)
    proxy_script = scripts_dir / "software_zero_proxy.py"
    if not proxy_script.exists():
        raise FileNotFoundError(f"software_zero_proxy.py not found: {proxy_script}")

    actions = [
        LogInfo(
            msg=(
                "[software_zero_proxy_launch] "
                f"hand_side={hand_side}, start_driver={start_driver}, start_proxy={start_proxy}, "
                f"device_config={config_file}, zero_config={zero_config_file}"
            )
        )
    ]

    proxy_actions = []
    for hand in _hands_from_arg(hand_side):
        hand_cfg = _select_hand_cfg(root_cfg, hand)
        dev_config_text = _build_dev_config_text(hand_cfg)
        raw_cmd_topic, raw_feedback_topic, cmd_topic, feedback_topic = _hand_topics(hand)
        driver_node_name = f"y_hand_{hand}_software_zero_raw_node"

        if start_driver:
            actions.append(
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "run",
                        "devices_pkg",
                        "Y_Hand_Node",
                        driver_node_name,
                        raw_feedback_topic,
                        raw_cmd_topic,
                        dev_config_text,
                    ],
                    name=driver_node_name,
                    output="screen",
                    on_exit=Shutdown(reason="software_zero_proxy_launch is shutting down"),
                )
            )

        if start_proxy:
            proxy_actions.append(
                ExecuteProcess(
                    cmd=[
                        "python3",
                        "-u",
                        str(proxy_script),
                        "--hand",
                        hand,
                        "--config",
                        str(zero_config_file),
                        "--raw-cmd-topic",
                        raw_cmd_topic,
                        "--raw-feedback-topic",
                        raw_feedback_topic,
                        "--cmd-topic",
                        cmd_topic,
                        "--feedback-topic",
                        feedback_topic,
                        "--reload-state-period-sec",
                        reload_state_period_sec,
                    ],
                    name=f"software_zero_proxy_{hand}",
                    output="screen",
                    on_exit=Shutdown(reason="software_zero_proxy_launch is shutting down"),
                )
            )

    if start_proxy:
        if start_driver and proxy_start_delay_sec > 0.0:
            actions.append(TimerAction(period=proxy_start_delay_sec, actions=proxy_actions))
        else:
            actions.extend(proxy_actions)

    return actions


def generate_launch_description():
    launch_file = Path(__file__).resolve()
    project_root = _find_project_root(launch_file)
    default_device_config = project_root / "config" / "bimanual_hand_device_config.json"
    default_zero_config = project_root / "config" / "software_zero_calibration_config.json"

    return LaunchDescription(
        [
            _declare_arg(
                "hand_side",
                "right",
                "Which hand to launch: left/right/both.",
            ),
            _declare_arg(
                "config_file",
                str(default_device_config),
                "Bimanual device config JSON.",
            ),
            _declare_arg(
                "zero_config_file",
                str(default_zero_config),
                "Software-zero calibration config JSON.",
            ),
            _declare_arg(
                "start_driver",
                "true",
                "Start raw Y_Hand_Node driver(s).",
            ),
            _declare_arg(
                "start_proxy",
                "true",
                "Start software-zero proxy node(s).",
            ),
            _declare_arg(
                "proxy_start_delay_sec",
                "1.0",
                "Delay proxy startup after raw driver startup.",
            ),
            _declare_arg(
                "reload_state_period_sec",
                "0.0",
                "0 disables state reload; positive value reloads zero_offset periodically.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
