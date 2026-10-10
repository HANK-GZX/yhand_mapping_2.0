import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


CONTROL_SCRIPT_START_DELAY_SEC = 1.5
NON_THUMB_FINGERS = ("index", "middle", "ring", "little")

MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT = {6: (-50.0, 500.0),
 4: (-600.0, 50.0),
 27: (-150.0, 100.0),
 7: (-50.0, 500.0),
 8: (-100.0, 150.0),
 14: (-600.0, 50.0),
 3: (-600.0, 50.0),
 25: (-50.0, 500.0),
 28: (-50.0, 50.0),
 1: (-50.0, 50.0),
 15: (-600.0, 50.0),
 16: (-600.0, 50.0),
 23: (-50.0, 500.0),
 2: (-600.0, 50.0),
 24: (-50.0, 50.0),
 9: (-50.0, 50.0),
 17: (-600.0, 50.0),
 21: (-50.0, 50.0),
 10: (-50.0, 500.0),
 22: (-50.0, 50.0),
 13: (-600.0, 50.0),
 11: (-50.0, 500.0),
 20: (-50.0, 50.0),
 18: (-50.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT = {20: (-50.0, 500.0),
 18: (-400.0, 50.0),
 13: (-200.0, 400.0),
 21: (-50.0, 500.0),
 22: (-100.0, 400.0),
 28: (-400.0, 50.0),
 17: (-500.0, 50.0),
 11: (-50.0, 500.0),
 14: (-100.0, 100.0),
 15: (-100.0, 100.0),
 1: (-500.0, 50.0),
 2: (-500.0, 50.0),
 9: (-50.0, 500.0),
 16: (-500.0, 50.0),
 10: (-100.0, 100.0),
 23: (-100.0, 100.0),
 3: (-500.0, 50.0),
 7: (-50.0, 50.0),
 24: (-50.0, 500.0),
 8: (-50.0, 50.0),
 27: (-500.0, 50.0),
 25: (-50.0, 500.0),
 6: (-50.0, 50.0),
 4: (-50.0, 50.0)}

Q_OFFSET_MAP_DEFAULT = {
    "left": {i: 0.0 for i in range(20)},
    "right": {i: 0.0 for i in range(20)},
}


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
            "driver_node_name": "y_hand_left_five_finger_hybrid_mapping_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_five_finger_hybrid_mapping_node",
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
    thumb_enable_raw = LaunchConfiguration("enable_thumb").perform(context).strip()
    thumb_enable_legacy_raw = LaunchConfiguration("enable_model").perform(context).strip()
    thumb_enable_summary = thumb_enable_raw or thumb_enable_legacy_raw or "true"

    def _append_arg_if_nonempty(cmd_list, arg_name: str, arg_value: str):
        value = str(arg_value).strip()
        if value:
            cmd_list.extend([arg_name, value])

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    mapping_script = scripts_dir / "five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py"
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")
    if not mapping_script.exists():
        raise FileNotFoundError(
            "five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py "
            f"not found: {mapping_script}"
        )

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
        condition=IfCondition(LaunchConfiguration("start_driver_node")),
    )

    glove_process = ExecuteProcess(
        cmd=["python3", "-u", str(glove_script)],
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
        "--glove-raw-skeleton-topic",
        LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
        "--defaults-path",
        LaunchConfiguration("defaults_path").perform(context),
        "--control-period-sec",
        LaunchConfiguration("control_period_sec").perform(context),
        "--status-period-sec",
        LaunchConfiguration("status_period_sec").perform(context),
        "--glove-max-hz",
        LaunchConfiguration("glove_max_hz").perform(context),
        "--glove-sub-queue-size",
        LaunchConfiguration("glove_sub_queue_size").perform(context),
        "--glove-pending-maxlen",
        LaunchConfiguration("glove_pending_maxlen").perform(context),
        "--motor-feedback-limits-left-json",
        LaunchConfiguration("motor_feedback_limits_left_json").perform(context),
        "--motor-feedback-limits-right-json",
        LaunchConfiguration("motor_feedback_limits_right_json").perform(context),
        "--q-offset-map-json",
        LaunchConfiguration("q_offset_map_json").perform(context),
        "--device",
        LaunchConfiguration("device").perform(context),
        "--ready-timeout-sec",
        LaunchConfiguration("ready_timeout_sec").perform(context),
    ]

    _append_arg_if_nonempty(mapping_cmd, "--enable-model", LaunchConfiguration("enable_model").perform(context))
    _append_arg_if_nonempty(mapping_cmd, "--model-ckpt", LaunchConfiguration("model_ckpt").perform(context))
    _append_arg_if_nonempty(mapping_cmd, "--model-motor-ids", LaunchConfiguration("model_motor_ids").perform(context))
    _append_arg_if_nonempty(mapping_cmd, "--q-indices", LaunchConfiguration("q_indices").perform(context))
    _append_arg_if_nonempty(
        mapping_cmd,
        "--mirror-right-from-left",
        LaunchConfiguration("mirror_right_from_left").perform(context),
    )
    _append_arg_if_nonempty(mapping_cmd, "--enable-thumb", LaunchConfiguration("enable_thumb").perform(context))
    _append_arg_if_nonempty(
        mapping_cmd,
        "--thumb-model-ckpt",
        LaunchConfiguration("thumb_model_ckpt").perform(context),
    )
    _append_arg_if_nonempty(
        mapping_cmd,
        "--thumb-model-motor-ids",
        LaunchConfiguration("thumb_model_motor_ids").perform(context),
    )
    _append_arg_if_nonempty(
        mapping_cmd,
        "--thumb-q-indices",
        LaunchConfiguration("thumb_q_indices").perform(context),
    )
    _append_arg_if_nonempty(
        mapping_cmd,
        "--thumb-mirror-right-from-left",
        LaunchConfiguration("thumb_mirror_right_from_left").perform(context),
    )

    for finger in NON_THUMB_FINGERS:
        mapping_cmd.extend(
            [
                f"--enable-{finger}",
                LaunchConfiguration(f"enable_{finger}").perform(context),
            ]
        )
        _append_arg_if_nonempty(
            mapping_cmd,
            f"--{finger}-model-ckpt",
            LaunchConfiguration(f"{finger}_model_ckpt").perform(context),
        )
        _append_arg_if_nonempty(
            mapping_cmd,
            f"--{finger}-curl-meta",
            LaunchConfiguration(f"{finger}_curl_meta").perform(context),
        )
        _append_arg_if_nonempty(
            mapping_cmd,
            f"--{finger}-model-motor-ids",
            LaunchConfiguration(f"{finger}_model_motor_ids").perform(context),
        )
        _append_arg_if_nonempty(
            mapping_cmd,
            f"--{finger}-q-indices",
            LaunchConfiguration(f"{finger}_q_indices").perform(context),
        )
        _append_arg_if_nonempty(
            mapping_cmd,
            f"--{finger}-mirror-left-from-right",
            LaunchConfiguration(f"{finger}_mirror_left_from_right").perform(context),
        )

    use_raw = _bool_from_launch_arg(
        LaunchConfiguration("right_non_thumb_q_from_raw_skeleton").perform(context)
    )
    if use_raw:
        mapping_cmd.append("--right-non-thumb-q-from-raw-skeleton")
    else:
        mapping_cmd.append("--no-right-non-thumb-q-from-raw-skeleton")

    require_ready = _bool_from_launch_arg(LaunchConfiguration("require_ready").perform(context))
    if require_ready:
        mapping_cmd.append("--require-ready")

    mapping_process = ExecuteProcess(
        cmd=mapping_cmd,
        name=f"five_finger_hybrid_mapping_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch] target_hand={hand}, "
                f"driver_node={info['driver_node_name']}, "
                f"start_driver_node={LaunchConfiguration('start_driver_node').perform(context)}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"glove_topic={LaunchConfiguration('glove_topic').perform(context)}, "
                f"glove_raw_skeleton_topic={LaunchConfiguration('glove_raw_skeleton_topic').perform(context)}, "
                f"right_non_thumb_q_from_raw_skeleton={LaunchConfiguration('right_non_thumb_q_from_raw_skeleton').perform(context)}, "
                f"thumb_enable={thumb_enable_summary}, "
                f"enable_index={LaunchConfiguration('enable_index').perform(context)}, "
                f"enable_middle={LaunchConfiguration('enable_middle').perform(context)}, "
                f"enable_ring={LaunchConfiguration('enable_ring').perform(context)}, "
                f"enable_little={LaunchConfiguration('enable_little').perform(context)}"
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
                default_value="right",
                description="Target hand for mapping control: left/right",
            ),
            DeclareLaunchArgument("start_driver_node", default_value="true"),
            DeclareLaunchArgument("start_glove_node", default_value="true"),
            DeclareLaunchArgument("glove_topic", default_value="/glove_joint_data"),
            DeclareLaunchArgument(
                "glove_raw_skeleton_topic",
                default_value="/glove_joint_data_raw_skeleton",
            ),
            DeclareLaunchArgument(
                "right_non_thumb_q_from_raw_skeleton",
                default_value="true",
                description="Solve non-thumb q_0..q_3 from selected hand raw skeleton",
            ),
            DeclareLaunchArgument(
                "defaults_path",
                default_value=str(scripts_dir / "muscle_adjustment_defaults.py"),
            ),
            DeclareLaunchArgument("enable_model", default_value=""),
            DeclareLaunchArgument("model_ckpt", default_value=""),
            DeclareLaunchArgument("model_motor_ids", default_value=""),
            DeclareLaunchArgument("q_indices", default_value=""),
            DeclareLaunchArgument("mirror_right_from_left", default_value=""),
            DeclareLaunchArgument("enable_thumb", default_value=""),
            DeclareLaunchArgument("thumb_model_ckpt", default_value=""),
            DeclareLaunchArgument("thumb_model_motor_ids", default_value=""),
            DeclareLaunchArgument("thumb_q_indices", default_value=""),
            DeclareLaunchArgument("thumb_mirror_right_from_left", default_value=""),
            DeclareLaunchArgument("enable_index", default_value="true"),
            DeclareLaunchArgument("index_model_ckpt", default_value=""),
            DeclareLaunchArgument("index_curl_meta", default_value=""),
            DeclareLaunchArgument("index_model_motor_ids", default_value=""),
            DeclareLaunchArgument("index_q_indices", default_value=""),
            DeclareLaunchArgument("index_mirror_left_from_right", default_value=""),
            DeclareLaunchArgument("enable_middle", default_value="true"),
            DeclareLaunchArgument("middle_model_ckpt", default_value=""),
            DeclareLaunchArgument("middle_curl_meta", default_value=""),
            DeclareLaunchArgument("middle_model_motor_ids", default_value=""),
            DeclareLaunchArgument("middle_q_indices", default_value=""),
            DeclareLaunchArgument("middle_mirror_left_from_right", default_value=""),
            DeclareLaunchArgument("enable_ring", default_value="true"),
            DeclareLaunchArgument("ring_model_ckpt", default_value=""),
            DeclareLaunchArgument("ring_curl_meta", default_value=""),
            DeclareLaunchArgument("ring_model_motor_ids", default_value=""),
            DeclareLaunchArgument("ring_q_indices", default_value=""),
            DeclareLaunchArgument("ring_mirror_left_from_right", default_value=""),
            DeclareLaunchArgument("enable_little", default_value="true"),
            DeclareLaunchArgument("little_model_ckpt", default_value=""),
            DeclareLaunchArgument("little_curl_meta", default_value=""),
            DeclareLaunchArgument("little_model_motor_ids", default_value=""),
            DeclareLaunchArgument("little_q_indices", default_value=""),
            DeclareLaunchArgument("little_mirror_left_from_right", default_value=""),
            DeclareLaunchArgument("control_period_sec", default_value="0.01"),
            DeclareLaunchArgument("status_period_sec", default_value="2.0"),
            DeclareLaunchArgument("glove_max_hz", default_value="100.0"),
            DeclareLaunchArgument("glove_sub_queue_size", default_value="60"),
            DeclareLaunchArgument("glove_pending_maxlen", default_value="60"),
            DeclareLaunchArgument(
                "motor_feedback_limits_left_json",
                default_value=json.dumps(MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT, ensure_ascii=False),
            ),
            DeclareLaunchArgument(
                "motor_feedback_limits_right_json",
                default_value=json.dumps(MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT, ensure_ascii=False),
            ),
            DeclareLaunchArgument(
                "q_offset_map_json",
                default_value=json.dumps(Q_OFFSET_MAP_DEFAULT, ensure_ascii=False),
            ),
            DeclareLaunchArgument("device", default_value="auto"),
            DeclareLaunchArgument("require_ready", default_value="false"),
            DeclareLaunchArgument("ready_timeout_sec", default_value="12.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
