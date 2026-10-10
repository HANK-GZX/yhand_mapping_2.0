import json
import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration


def _bool_from_launch_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean launch arg value: {value!r}")


def _bool_from_cfg(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return _bool_from_launch_arg(value)
    return bool(value)


def _normalize_target_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value not in ("left", "right"):
        raise ValueError(f"Unsupported target_hand={hand!r}, expected 'left' or 'right'.")
    return value


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


def _load_runtime_cfg(config_file: Path) -> dict:
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        obj = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON config: {config_file}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Top-level JSON must be an object")
    return obj


def _resolve_hand_runtime_cfg(runtime_cfg: dict) -> dict:
    cfg = dict(runtime_cfg)
    hand = str(cfg.get("hand", cfg.get("default_hand", "right"))).strip().lower() or "right"
    if hand not in ("left", "right"):
        raise ValueError(f"Unsupported hand in config: {hand!r}")

    hands_cfg = cfg.get("hands")
    if isinstance(hands_cfg, dict):
        selected = hands_cfg.get(hand)
        if selected is None:
            raise ValueError(f"Config hands missing selected hand entry: {hand!r}")
        if not isinstance(selected, dict):
            raise ValueError(f"Config hands[{hand!r}] must be an object")
        cfg.pop("hands", None)
        cfg.update(selected)

    cfg["hand"] = hand
    return cfg


def _launch_setup(context):
    launch_file = Path(__file__).resolve()
    project_root = launch_file.parent.parent

    target_hand = _normalize_target_hand(LaunchConfiguration("target_hand").perform(context))
    config_file_raw = LaunchConfiguration("config_file").perform(context).strip()
    if config_file_raw:
        config_file = Path(config_file_raw).expanduser()
        if not config_file.is_absolute():
            config_file = (project_root / config_file).resolve()
    else:
        config_file = (project_root / "config" / f"{target_hand}_hand_device_config.json").resolve()

    start_driver = _bool_from_launch_arg(LaunchConfiguration("start_driver").perform(context))
    start_glove_node = _bool_from_launch_arg(LaunchConfiguration("start_glove_node").perform(context))
    start_controller = _bool_from_launch_arg(LaunchConfiguration("start_controller").perform(context))
    dry_run = _bool_from_launch_arg(LaunchConfiguration("dry_run").perform(context))
    split_process = _bool_from_launch_arg(LaunchConfiguration("split_process").perform(context))
    controller_in_terminal_raw = LaunchConfiguration("controller_in_terminal").perform(context).strip().lower()
    if controller_in_terminal_raw not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError("Invalid controller_in_terminal. Expected one of: auto/true/false")

    runtime_cfg_raw = _load_runtime_cfg(config_file)
    runtime_cfg_hands = runtime_cfg_raw.get("hands")
    runtime_cfg_declared_hand = str(runtime_cfg_raw.get("hand", "")).strip().lower()
    if isinstance(runtime_cfg_hands, dict):
        runtime_cfg_raw["hand"] = target_hand
    else:
        if runtime_cfg_declared_hand and runtime_cfg_declared_hand != target_hand:
            raise ValueError(
                f"Config hand={runtime_cfg_declared_hand!r} does not match target_hand={target_hand!r}: "
                f"{config_file}"
            )
        runtime_cfg_raw["hand"] = target_hand

    runtime_cfg = _resolve_hand_runtime_cfg(runtime_cfg_raw)
    dev_config_text = _build_dev_config_text(runtime_cfg)

    hand = str(runtime_cfg.get("hand", "right")).strip().lower()
    driver_node_name = str(runtime_cfg.get("driver_node_name", f"y_hand_{hand}_wrist_angle_node"))
    cmd_topic = str(runtime_cfg.get("cmd_topic", f"/yhand_{hand}_cmd"))
    feedback_topic = str(runtime_cfg.get("feedback_topic", f"/yhand_{hand}_feedback"))
    glove_topic = str(runtime_cfg.get("glove_topic", "/glove_joint_data_raw_skeleton"))
    input_source = str(runtime_cfg.get("input_source", "manus_raw_skeleton")).strip().lower() or "manus_raw_skeleton"
    mocap_tf_topic = str(runtime_cfg.get("mocap_tf_topic", "/tf_static"))
    mocap_hand_frame = str(runtime_cfg.get("mocap_hand_frame", "")).strip()
    target_topic = str(runtime_cfg.get("target_topic", "/wrist_glove_target_angle"))
    default_side_code = 1.0 if hand == "right" else 0.0
    glove_side_code = float(runtime_cfg.get("glove_side_code", default_side_code))
    alpha_scale = float(runtime_cfg.get("alpha_scale", 1.0))
    beta_scale = float(runtime_cfg.get("beta_scale", 1.0))
    alpha_offset_deg = float(runtime_cfg.get("alpha_offset_deg", 0.0))
    beta_offset_deg = float(runtime_cfg.get("beta_offset_deg", 0.0))
    alpha_limit_deg = float(runtime_cfg.get("alpha_limit_deg", 5.0))
    beta_limit_deg = abs(float(runtime_cfg.get("beta_limit_deg", 30.0)))
    beta_min_deg = float(runtime_cfg.get("beta_min_deg", -beta_limit_deg))
    beta_max_deg = float(runtime_cfg.get("beta_max_deg", beta_limit_deg))
    max_command_step_alpha_deg = float(runtime_cfg.get("max_command_step_alpha_deg", 2.0))
    max_command_step_beta_deg = float(runtime_cfg.get("max_command_step_beta_deg", 6.0))
    output_publish_rate_hz = float(runtime_cfg.get("output_publish_rate_hz", 500.0))
    status_print_period_sec = float(runtime_cfg.get("status_print_period_sec", 1.0))
    deadband_deg = float(runtime_cfg.get("deadband_deg", 0.1))
    allow_single_chunk_fallback = _bool_from_cfg(
        runtime_cfg.get("allow_single_chunk_fallback", True), True
    )
    fallback_chunk_index = int(runtime_cfg.get("fallback_chunk_index", -1))
    controller_start_delay_sec = float(runtime_cfg.get("controller_start_delay_sec", 1.5))
    zero_on_exit = "true" if _bool_from_cfg(runtime_cfg.get("zero_on_exit", True), True) else "false"

    motor_offsets_cfg = runtime_cfg.get("motor_position_offsets_deg", [0.0] * 28)
    motor_offsets_json = json.dumps(motor_offsets_cfg, ensure_ascii=False)

    script_path = project_root / "scripts" / "wrist_glove_motor_control.py"
    target_publisher_script_path = project_root / "scripts" / "wrist_glove_target_publisher.py"
    target_motor_script_path = project_root / "scripts" / "wrist_target_motor_control.py"
    defaults_path = project_root / "scripts" / "muscle_adjustment_defaults.py"
    glove_script_path = project_root / "scripts" / "minimal_example_dual_stream_commented.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Controller script not found: {script_path}")
    if not defaults_path.exists():
        raise FileNotFoundError(f"Defaults file not found: {defaults_path}")
    if split_process and (not target_publisher_script_path.exists()):
        raise FileNotFoundError(f"Target publisher script not found: {target_publisher_script_path}")
    if split_process and (not target_motor_script_path.exists()):
        raise FileNotFoundError(f"Target motor script not found: {target_motor_script_path}")
    if start_glove_node and (not glove_script_path.exists()):
        raise FileNotFoundError(f"Glove script not found: {glove_script_path}")

    actions = [
        LogInfo(
            msg=(
                "[wrist_glove_control_launch] "
                f"target_hand={target_hand}, "
                f"config_file={config_file}, "
                f"hand={hand}, "
                f"driver_node={driver_node_name}, "
                f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}, glove_topic={glove_topic}, "
                f"input_source={input_source}, mocap_tf_topic={mocap_tf_topic}, mocap_hand_frame={mocap_hand_frame}, "
                f"target_topic={target_topic}, "
                f"start_driver={start_driver}, start_glove_node={start_glove_node}, start_controller={start_controller}, "
                f"dry_run={dry_run}, split_process={split_process}, "
                f"controller_start_delay_sec={controller_start_delay_sec:.3f}, "
                f"controller_in_terminal={controller_in_terminal_raw}, "
                f"alpha_limit_deg={alpha_limit_deg:.2f}, beta_range_deg=[{beta_min_deg:.2f},{beta_max_deg:.2f}], "
                f"max_command_step_alpha_deg={max_command_step_alpha_deg:.3f}, "
                f"max_command_step_beta_deg={max_command_step_beta_deg:.3f}, "
                f"output_publish_rate_hz={output_publish_rate_hz:.2f}, "
                f"deadband_deg={deadband_deg:.3f}, "
                f"allow_single_chunk_fallback={allow_single_chunk_fallback}, "
                f"fallback_chunk_index={fallback_chunk_index}, "
                f"motor_offsets_cfg_type={type(motor_offsets_cfg).__name__}"
            )
        )
    ]

    if start_driver:
        driver_process = ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "devices_pkg",
                "Y_Hand_Node",
                driver_node_name,
                feedback_topic,
                cmd_topic,
                dev_config_text,
            ],
            name=driver_node_name,
            output="screen",
            on_exit=Shutdown(reason="wrist_glove_control_launch is shutting down"),
        )
        actions.append(driver_process)

    if start_glove_node:
        if input_source == "mocap_tf":
            actions.append(
                LogInfo(
                    msg=(
                        "[wrist_glove_control_launch] input_source=mocap_tf but start_glove_node=true. "
                        "The Manus glove publisher is not required for mocap TF input."
                    )
                )
            )
        glove_process = ExecuteProcess(
            cmd=[
                "python3",
                "-u",
                str(glove_script_path),
                "--log-wrist-node0",
            ],
            name="glove_publisher",
            output="screen",
            on_exit=Shutdown(reason="wrist_glove_control_launch is shutting down"),
        )
        actions.append(glove_process)

    if start_controller:
        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        gnome_terminal = shutil.which("gnome-terminal")
        if controller_in_terminal_raw == "auto":
            use_gnome_terminal = (gnome_terminal is not None) and display_available
        else:
            use_gnome_terminal = _bool_from_launch_arg(controller_in_terminal_raw)
            if use_gnome_terminal and gnome_terminal is None:
                raise RuntimeError(
                    "controller_in_terminal=true but gnome-terminal not found. "
                    "Install gnome-terminal or set controller_in_terminal:=false."
                )
            if use_gnome_terminal and (not display_available):
                raise RuntimeError(
                    "controller_in_terminal=true but DISPLAY/WAYLAND_DISPLAY is unavailable. "
                    "Set controller_in_terminal:=false for headless mode."
                )

        if split_process and (not dry_run):
            if input_source != "manus_raw_skeleton":
                raise RuntimeError(
                    "split_process=true currently supports only input_source=manus_raw_skeleton. "
                    "Use split_process:=false for mocap_tf input."
                )
            target_publisher_cmd = [
                "python3",
                "-u",
                str(target_publisher_script_path),
                "--hand",
                hand,
                "--glove-topic",
                glove_topic,
                "--target-topic",
                target_topic,
                "--glove-side-code",
                str(glove_side_code),
                "--alpha-scale",
                str(alpha_scale),
                "--beta-scale",
                str(beta_scale),
                "--alpha-offset-deg",
                str(alpha_offset_deg),
                "--beta-offset-deg",
                str(beta_offset_deg),
                "--deadband-deg",
                str(deadband_deg),
                "--status-print-period-sec",
                str(status_print_period_sec),
                "--allow-single-chunk-fallback",
                ("true" if allow_single_chunk_fallback else "false"),
                "--fallback-chunk-index",
                str(fallback_chunk_index),
            ]
            target_motor_cmd = [
                "python3",
                "-u",
                str(target_motor_script_path),
                "--hand",
                hand,
                "--defaults-path",
                str(defaults_path),
                "--cmd-topic",
                cmd_topic,
                "--feedback-topic",
                feedback_topic,
                "--target-topic",
                target_topic,
                "--output-publish-rate-hz",
                str(output_publish_rate_hz),
                "--alpha-limit-deg",
                str(alpha_limit_deg),
                "--beta-min-deg",
                str(beta_min_deg),
                "--beta-max-deg",
                str(beta_max_deg),
                "--max-command-step-alpha-deg",
                str(max_command_step_alpha_deg),
                "--max-command-step-beta-deg",
                str(max_command_step_beta_deg),
                "--status-print-period-sec",
                str(status_print_period_sec),
                "--motor-offsets-json",
                motor_offsets_json,
                "--zero-on-exit",
                zero_on_exit,
            ]
            if use_gnome_terminal:
                target_publisher_cmd = [
                    gnome_terminal,
                    "--wait",
                    "--title",
                    f"wrist_glove_target_publisher_{hand}",
                    "--",
                    *target_publisher_cmd,
                ]
                target_motor_cmd = [
                    gnome_terminal,
                    "--wait",
                    "--title",
                    f"wrist_target_motor_control_{hand}",
                    "--",
                    *target_motor_cmd,
                ]

            target_publisher_process = ExecuteProcess(
                cmd=target_publisher_cmd,
                name="wrist_glove_target_publisher",
                output="screen",
                on_exit=Shutdown(reason="wrist_glove_control_launch is shutting down"),
            )
            target_motor_process = ExecuteProcess(
                cmd=target_motor_cmd,
                name="wrist_target_motor_control",
                output="screen",
                on_exit=Shutdown(reason="wrist_glove_control_launch is shutting down"),
            )
            controller_processes = [target_publisher_process, target_motor_process]
        else:
            controller_cmd = [
                "python3",
                "-u",
                str(script_path),
                "--hand",
                hand,
                "--defaults-path",
                str(defaults_path),
                "--cmd-topic",
                cmd_topic,
                "--feedback-topic",
                feedback_topic,
                "--glove-topic",
                glove_topic,
                "--input-source",
                input_source,
                "--mocap-tf-topic",
                mocap_tf_topic,
                "--mocap-hand-frame",
                mocap_hand_frame,
                "--glove-side-code",
                str(glove_side_code),
                "--alpha-scale",
                str(alpha_scale),
                "--beta-scale",
                str(beta_scale),
                "--alpha-offset-deg",
                str(alpha_offset_deg),
                "--beta-offset-deg",
                str(beta_offset_deg),
                "--output-publish-rate-hz",
                str(output_publish_rate_hz),
                "--alpha-limit-deg",
                str(alpha_limit_deg),
                "--beta-min-deg",
                str(beta_min_deg),
                "--beta-max-deg",
                str(beta_max_deg),
                "--max-command-step-alpha-deg",
                str(max_command_step_alpha_deg),
                "--max-command-step-beta-deg",
                str(max_command_step_beta_deg),
                "--status-print-period-sec",
                str(status_print_period_sec),
                "--deadband-deg",
                str(deadband_deg),
                "--allow-single-chunk-fallback",
                ("true" if allow_single_chunk_fallback else "false"),
                "--fallback-chunk-index",
                str(fallback_chunk_index),
                "--motor-offsets-json",
                motor_offsets_json,
                "--zero-on-exit",
                zero_on_exit,
                "--dry-run",
                ("true" if dry_run else "false"),
            ]
            if use_gnome_terminal:
                controller_cmd = [
                    gnome_terminal,
                    "--wait",
                    "--title",
                    f"wrist_glove_motor_control_{hand}",
                    "--",
                    *controller_cmd,
                ]

            controller_processes = [
                ExecuteProcess(
                    cmd=controller_cmd,
                    name="wrist_glove_motor_control",
                    output="screen",
                    on_exit=Shutdown(reason="wrist_glove_control_launch is shutting down"),
                )
            ]

        if start_driver or start_glove_node:
            actions.append(TimerAction(period=controller_start_delay_sec, actions=controller_processes))
        else:
            actions.extend(controller_processes)

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Target hand for wrist glove control: left/right",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="",
                description=(
                    "Optional wrist glove-control runtime JSON config. "
                    "Leave empty to auto-select config/left_hand_device_config.json or "
                    "config/right_hand_device_config.json from target_hand."
                ),
            ),
            DeclareLaunchArgument(
                "start_driver",
                default_value="true",
                description="Start Y_Hand_Node driver process (true/false).",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Start minimal_example_dual_stream_commented.py glove publisher (true/false).",
            ),
            DeclareLaunchArgument(
                "start_controller",
                default_value="true",
                description="Start wrist_glove_motor_control.py process (true/false).",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="Run glove controller without publishing wrist motor commands (true/false).",
            ),
            DeclareLaunchArgument(
                "split_process",
                default_value="false",
                description=(
                    "Run glove target publisher and motor command controller as separate processes "
                    "for isolation testing (true/false)."
                ),
            ),
            DeclareLaunchArgument(
                "controller_in_terminal",
                default_value="false",
                description="Run controller in gnome-terminal (true/false/auto).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
