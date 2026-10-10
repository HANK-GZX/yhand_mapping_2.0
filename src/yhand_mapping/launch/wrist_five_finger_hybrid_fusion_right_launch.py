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
    if value not in ("left", "right", "both"):
        raise ValueError(f"Unsupported target_hand={hand!r}, expected 'left', 'right', or 'both'.")
    return value


def _resolve_target_hands(target_hand: str):
    if target_hand == "both":
        return ("left", "right")
    return (target_hand,)


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
    target_hands = _resolve_target_hands(target_hand)
    config_file_raw = LaunchConfiguration("config_file").perform(context).strip()
    if config_file_raw:
        config_file = Path(config_file_raw).expanduser()
        if not config_file.is_absolute():
            config_file = (project_root / config_file).resolve()
    else:
        if target_hand == "both":
            config_file = (project_root / "config" / "bimanual_hand_device_config.json").resolve()
        else:
            config_file = (project_root / "config" / f"{target_hand}_hand_device_config.json").resolve()

    start_driver = _bool_from_launch_arg(LaunchConfiguration("start_driver").perform(context))
    start_glove_node = _bool_from_launch_arg(LaunchConfiguration("start_glove_node").perform(context))
    start_mocap_node = _bool_from_launch_arg(LaunchConfiguration("start_mocap_node").perform(context))
    start_controller = _bool_from_launch_arg(LaunchConfiguration("start_controller").perform(context))
    dry_run = _bool_from_launch_arg(LaunchConfiguration("dry_run").perform(context))
    controller_in_terminal_raw = LaunchConfiguration("controller_in_terminal").perform(context).strip().lower()
    if controller_in_terminal_raw not in ("auto", "1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ValueError("Invalid controller_in_terminal. Expected one of: auto/true/false")

    runtime_cfg_raw = _load_runtime_cfg(config_file)

    def _resolve_runtime_cfg_for_hand(selected_hand: str) -> dict:
        cfg = dict(runtime_cfg_raw)
        runtime_cfg_hands = cfg.get("hands")
        runtime_cfg_declared_hand = str(cfg.get("hand", "")).strip().lower()
        if isinstance(runtime_cfg_hands, dict):
            cfg["hand"] = selected_hand
        else:
            if target_hand == "both":
                raise ValueError(
                    f"target_hand='both' requires a config with hands.left/hands.right entries: {config_file}"
                )
            if runtime_cfg_declared_hand and runtime_cfg_declared_hand != selected_hand:
                raise ValueError(
                    f"Config hand={runtime_cfg_declared_hand!r} does not match target_hand={selected_hand!r}: "
                    f"{config_file}"
                )
            cfg["hand"] = selected_hand
        return _resolve_hand_runtime_cfg(cfg)

    def _json_text_if_any(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)

    controller_script_path = project_root / "scripts" / "wrist_five_finger_hybrid_fusion_mapping_main.py"
    defaults_path = project_root / "scripts" / "muscle_adjustment_defaults.py"
    glove_script_path = project_root / "scripts" / "minimal_example_dual_stream_commented.py"
    mocap_script_path = project_root / "scripts" / "mocapapi_python-main" / "mocap_main_stickman_ros2.py"
    if not controller_script_path.exists():
        raise FileNotFoundError(f"Hybrid fusion controller script not found: {controller_script_path}")
    if not defaults_path.exists():
        raise FileNotFoundError(f"Defaults file not found: {defaults_path}")
    if start_glove_node and (not glove_script_path.exists()):
        raise FileNotFoundError(f"Glove script not found: {glove_script_path}")
    if start_mocap_node and (not mocap_script_path.exists()):
        raise FileNotFoundError(f"Mocap script not found: {mocap_script_path}")

    actions = [
        LogInfo(
            msg=(
                "[wrist_five_finger_hybrid_fusion_launch] "
                f"target_hand={target_hand}, hands={list(target_hands)}, config_file={config_file}, "
                f"start_driver={start_driver}, start_glove_node={start_glove_node}, "
                f"start_mocap_node={start_mocap_node}, start_controller={start_controller}, "
                f"dry_run={dry_run}, controller_in_terminal={controller_in_terminal_raw}"
            )
        )
    ]

    if start_glove_node:
        glove_process = ExecuteProcess(
            cmd=["python3", "-u", str(glove_script_path)],
            name="glove_publisher",
            output="screen",
            on_exit=Shutdown(reason="wrist_five_finger_hybrid_fusion_launch is shutting down"),
        )
        actions.append(glove_process)

    if start_mocap_node:
        mocap_process = ExecuteProcess(
            cmd=["python3", "-u", str(mocap_script_path)],
            name="mocap_tf_publisher",
            output="screen",
            on_exit=Shutdown(reason="wrist_five_finger_hybrid_fusion_launch is shutting down"),
        )
        actions.append(mocap_process)

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

    controller_processes = []
    controller_start_delay_sec = 1.5
    hand_summaries = []

    for hand in target_hands:
        runtime_cfg = _resolve_runtime_cfg_for_hand(hand)
        dev_config_text = _build_dev_config_text(runtime_cfg)

        driver_node_name = str(runtime_cfg.get("driver_node_name", f"y_hand_{hand}_wrist_hybrid_fusion_node"))
        cmd_topic = str(runtime_cfg.get("cmd_topic", f"/yhand_{hand}_cmd"))
        feedback_topic = str(runtime_cfg.get("feedback_topic", f"/yhand_{hand}_feedback"))

        raw_glove_topic = str(runtime_cfg.get("glove_topic", "/glove_joint_data_raw_skeleton"))
        glove48_topic = str(runtime_cfg.get("five_finger_glove_topic", "/glove_joint_data"))
        glove_raw_skeleton_topic = str(
            runtime_cfg.get("five_finger_glove_raw_skeleton_topic", raw_glove_topic)
        )
        non_thumb_q_from_raw_skeleton = _bool_from_cfg(
            runtime_cfg.get(
                "five_finger_non_thumb_q_from_raw_skeleton",
                runtime_cfg.get(
                    f"five_finger_{hand}_non_thumb_q_from_raw_skeleton",
                    runtime_cfg.get(
                        f"{hand}_index_q_from_raw_skeleton",
                        runtime_cfg.get("right_index_q_from_raw_skeleton", True),
                    ),
                ),
            ),
            True,
        )

        default_side_code = 1.0 if hand == "right" else 0.0
        glove_side_code = float(
            runtime_cfg.get(
                "glove_side_code",
                runtime_cfg.get(
                    f"{hand}_glove_side_code",
                    runtime_cfg.get("right_glove_side_code", default_side_code),
                ),
            )
        )
        alpha_scale = float(runtime_cfg.get("alpha_scale", 1.0))
        beta_scale = float(runtime_cfg.get("beta_scale", 1.0))
        alpha_offset_deg = float(runtime_cfg.get("alpha_offset_deg", 0.0))
        beta_offset_deg = float(runtime_cfg.get("beta_offset_deg", 0.0))
        alpha_limit_deg = float(runtime_cfg.get("alpha_limit_deg", 5.0))
        beta_limit_deg = abs(float(runtime_cfg.get("beta_limit_deg", 30.0)))
        beta_min_deg = float(runtime_cfg.get("beta_min_deg", -beta_limit_deg))
        beta_max_deg = float(runtime_cfg.get("beta_max_deg", beta_limit_deg))
        max_command_step_alpha_deg = float(
            runtime_cfg.get(
                "five_finger_max_command_step_alpha_deg",
                runtime_cfg.get("max_command_step_alpha_deg", 2.0),
            )
        )
        max_command_step_beta_deg = float(
            runtime_cfg.get(
                "five_finger_max_command_step_beta_deg",
                runtime_cfg.get("max_command_step_beta_deg", 6.0),
            )
        )
        deadband_deg = float(runtime_cfg.get("deadband_deg", 0.1))
        output_publish_rate_hz = max(float(runtime_cfg.get("output_publish_rate_hz", 100.0)), 1e-6)
        status_print_period_sec = float(runtime_cfg.get("five_finger_status_period_sec", 1.0))
        controller_start_delay_sec = max(
            controller_start_delay_sec,
            float(runtime_cfg.get("controller_start_delay_sec", 1.5)),
        )
        zero_on_exit = "true" if _bool_from_cfg(runtime_cfg.get("zero_on_exit", True), True) else "false"
        device = str(runtime_cfg.get("five_finger_device", "auto"))

        motor_offsets_cfg = runtime_cfg.get("motor_position_offsets_deg", [0.0] * 28)
        motor_offsets_json = json.dumps(motor_offsets_cfg, ensure_ascii=False)

        q_offset_map_cfg = runtime_cfg.get("q_offset_map_json", None)
        if q_offset_map_cfg is None:
            q_offset_map_cfg = runtime_cfg.get("q_offset_map", None)
        motor_cmd_offset_map_cfg = runtime_cfg.get("motor_cmd_offset_map_json", None)
        if motor_cmd_offset_map_cfg is None:
            motor_cmd_offset_map_cfg = runtime_cfg.get("motor_cmd_offset_map", None)
        motor_feedback_limits_left_cfg = runtime_cfg.get("motor_feedback_limits_left_json", None)
        if motor_feedback_limits_left_cfg is None:
            motor_feedback_limits_left_cfg = runtime_cfg.get("motor_feedback_limits_left", None)
        motor_feedback_limits_right_cfg = runtime_cfg.get("motor_feedback_limits_right_json", None)
        if motor_feedback_limits_right_cfg is None:
            motor_feedback_limits_right_cfg = runtime_cfg.get("motor_feedback_limits_right", None)

        q_offset_map_json = _json_text_if_any(q_offset_map_cfg)
        motor_cmd_offset_map_json = _json_text_if_any(motor_cmd_offset_map_cfg)
        motor_feedback_limits_left_json = _json_text_if_any(motor_feedback_limits_left_cfg)
        motor_feedback_limits_right_json = _json_text_if_any(motor_feedback_limits_right_cfg)

        hand_summaries.append(
            f"{hand}:driver_node={driver_node_name},cmd_topic={cmd_topic},feedback_topic={feedback_topic},"
            f"glove48_topic={glove48_topic},raw_topic={glove_raw_skeleton_topic},"
            f"non_thumb_raw_skeleton={non_thumb_q_from_raw_skeleton},"
            f"alpha_limit_deg={alpha_limit_deg:.2f},beta_range_deg=[{beta_min_deg:.2f},{beta_max_deg:.2f}],"
            f"deadband_deg={deadband_deg:.3f},max_step_alpha_deg={max_command_step_alpha_deg:.3f},"
            f"max_step_beta_deg={max_command_step_beta_deg:.3f},output_publish_rate_hz={output_publish_rate_hz:.2f},"
            f"status_print_period_sec={status_print_period_sec:.3f},motor_offsets_cfg_type={type(motor_offsets_cfg).__name__}"
        )

        if start_driver:
            actions.append(
                ExecuteProcess(
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
                    on_exit=Shutdown(reason="wrist_five_finger_hybrid_fusion_launch is shutting down"),
                )
            )

        if start_controller:
            controller_cmd = [
                "python3",
                "-u",
                str(controller_script_path),
                "--hand",
                hand,
                "--config",
                str(config_file),
                "--defaults-path",
                str(defaults_path),
                "--cmd-topic",
                cmd_topic,
                "--feedback-topic",
                feedback_topic,
                "--glove48-topic",
                glove48_topic,
                "--glove-raw-topic",
                glove_raw_skeleton_topic,
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
                "--alpha-limit-deg",
                str(alpha_limit_deg),
                "--beta-min-deg",
                str(beta_min_deg),
                "--beta-max-deg",
                str(beta_max_deg),
                "--max-step-alpha-deg",
                str(max_command_step_alpha_deg),
                "--max-step-beta-deg",
                str(max_command_step_beta_deg),
                "--wrist-deadband-deg",
                str(deadband_deg),
                "--output-publish-rate-hz",
                str(output_publish_rate_hz),
                "--status-period-sec",
                str(status_print_period_sec),
                "--motor-offsets-json",
                motor_offsets_json,
                "--device",
                device,
                "--zero-on-exit",
                zero_on_exit,
            ]
            if not non_thumb_q_from_raw_skeleton:
                controller_cmd.append("--no-non-thumb-q-from-raw-skeleton")
            if dry_run:
                controller_cmd.append("--dry-run")
            if q_offset_map_json.strip():
                controller_cmd.extend(["--q-offset-map-json", q_offset_map_json])
            if motor_cmd_offset_map_json.strip():
                controller_cmd.extend(["--motor-cmd-offset-map-json", motor_cmd_offset_map_json])
            if motor_feedback_limits_left_json.strip():
                controller_cmd.extend(["--motor-feedback-limits-left-json", motor_feedback_limits_left_json])
            if motor_feedback_limits_right_json.strip():
                controller_cmd.extend(["--motor-feedback-limits-right-json", motor_feedback_limits_right_json])

            if use_gnome_terminal:
                controller_cmd = [
                    gnome_terminal,
                    "--wait",
                    "--title",
                    f"wrist_five_finger_hybrid_fusion_{hand}",
                    "--",
                    *controller_cmd,
                ]

            controller_processes.append(
                ExecuteProcess(
                    cmd=controller_cmd,
                    name=f"wrist_five_finger_hybrid_fusion_control_{hand}",
                    output="screen",
                    on_exit=Shutdown(reason="wrist_five_finger_hybrid_fusion_launch is shutting down"),
                )
            )

    actions.append(
        LogInfo(
            msg="[wrist_five_finger_hybrid_fusion_launch] " + "; ".join(hand_summaries)
        )
    )

    if start_controller:
        if start_driver or start_glove_node or start_mocap_node:
            actions.append(TimerAction(period=controller_start_delay_sec, actions=controller_processes))
        else:
            actions.extend(controller_processes)

    return actions


def generate_launch_description():
    launch_file = Path(__file__).resolve()
    project_root = launch_file.parent.parent

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Target hand for wrist five-finger hybrid fusion: left/right/both",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="",
                description=(
                    "Optional hybrid fusion runtime JSON config. "
                    "Leave empty to auto-select left/right hand config, or "
                    "config/bimanual_hand_device_config.json when target_hand=both."
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
                "start_mocap_node",
                default_value="true",
                description="Start mocap_main_stickman_ros2.py TF publisher for wrist mocap input (true/false).",
            ),
            DeclareLaunchArgument(
                "start_controller",
                default_value="true",
                description="Start wrist_five_finger_hybrid_fusion_mapping_main.py process (true/false).",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="Run hybrid fusion controller without publishing command topic (true/false).",
            ),
            DeclareLaunchArgument(
                "controller_in_terminal",
                default_value="false",
                description="Run controller in gnome-terminal (true/false/auto).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
