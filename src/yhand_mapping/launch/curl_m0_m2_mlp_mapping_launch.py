import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from hand_motor_map import (
    model_motor_ids_csv,
)

CONTROL_SCRIPT_START_DELAY_SEC = 1.5

FINGER_RUNTIME_PRESETS = {
    "index": {
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "index",
            "best_index_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "index",
            "index_target_q0_supervised_curl_1570.meta.json",
        ),
        "model_motor_ids": model_motor_ids_csv('right', 'index'),
        "q_indices": "4,5,6,7",
    },
    "middle": {
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "middle",
            "best_middle_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "middle",
            "middle_target_q0_supervised_curl_1357.meta.json",
        ),
        "model_motor_ids": model_motor_ids_csv('right', 'middle'),
        "q_indices": "8,9,10,11",
    },
    "ring": {
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "ring",
            "best_ring_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "ring",
            "ring_target_q0_supervised_curl_1704.meta.json",
        ),
        "model_motor_ids": model_motor_ids_csv('right', 'ring'),
        "q_indices": "12,13,14,15",
    },
    "little": {
        "model_ckpt_rel_parts": (
            "mp_curve_q0_right",
            "little",
            "best_little_q0_curl_multihead_mlp.pt",
        ),
        "curl_meta_rel_parts": (
            "curve_function",
            "little",
            "little_target_q0_supervised_curl_1320.meta.json",
        ),
        "model_motor_ids": model_motor_ids_csv('right', 'little'),
        "q_indices": "16,17,18,19",
    },
}
DEFAULT_FINGER_ORDER = ("index", "middle", "ring", "little")


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
            "driver_node_name": "y_hand_left_ring_curl_m0_m2_mlp_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_ring_curl_m0_m2_mlp_node",
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


def _finger_runtime_defaults(scripts_dir: Path, finger: str):
    value = finger.strip().lower()
    if value not in FINGER_RUNTIME_PRESETS:
        raise ValueError(
            f"Unsupported finger={finger!r}, expected one of {sorted(FINGER_RUNTIME_PRESETS.keys())}"
        )
    preset = FINGER_RUNTIME_PRESETS[value]
    return {
        "model_ckpt": str(scripts_dir.joinpath(*preset["model_ckpt_rel_parts"])),
        "curl_meta": str(scripts_dir.joinpath(*preset["curl_meta_rel_parts"])),
        "model_motor_ids": str(preset["model_motor_ids"]),
        "q_indices": str(preset["q_indices"]),
    }


def _resolve_enabled_finger_configs(context, scripts_dir: Path, selected_finger: str):
    generic_model_ckpt = LaunchConfiguration("model_ckpt").perform(context).strip()
    generic_curl_meta = LaunchConfiguration("curl_meta").perform(context).strip()
    generic_model_motor_ids = LaunchConfiguration("model_motor_ids").perform(context).strip()
    generic_q_indices = LaunchConfiguration("q_indices").perform(context).strip()
    generic_mirror = LaunchConfiguration("mirror_left_from_right").perform(context).strip()

    explicit_enable = any(
        LaunchConfiguration(f"enable_{finger}").perform(context).strip()
        for finger in DEFAULT_FINGER_ORDER
    )
    resolved = {}
    for finger in DEFAULT_FINGER_ORDER:
        defaults = _finger_runtime_defaults(scripts_dir, finger)
        enable_raw = LaunchConfiguration(f"enable_{finger}").perform(context).strip()
        if enable_raw:
            enable = enable_raw
        else:
            enable = "true" if ((not explicit_enable) and finger == selected_finger) else "false"

        finger_model_ckpt = LaunchConfiguration(f"{finger}_model_ckpt").perform(context).strip()
        finger_curl_meta = LaunchConfiguration(f"{finger}_curl_meta").perform(context).strip()
        finger_model_motor_ids = LaunchConfiguration(f"{finger}_model_motor_ids").perform(context).strip()
        finger_q_indices = LaunchConfiguration(f"{finger}_q_indices").perform(context).strip()
        finger_mirror = LaunchConfiguration(f"{finger}_mirror_left_from_right").perform(context).strip()

        if finger == selected_finger:
            finger_model_ckpt = finger_model_ckpt or generic_model_ckpt
            finger_curl_meta = finger_curl_meta or generic_curl_meta
            finger_model_motor_ids = finger_model_motor_ids or generic_model_motor_ids
            finger_q_indices = finger_q_indices or generic_q_indices
            finger_mirror = finger_mirror or generic_mirror

        resolved[finger] = {
            "enable": enable,
            "model_ckpt": finger_model_ckpt or defaults["model_ckpt"],
            "curl_meta": finger_curl_meta or defaults["curl_meta"],
            "model_motor_ids": finger_model_motor_ids or defaults["model_motor_ids"],
            "q_indices": finger_q_indices or defaults["q_indices"],
            "mirror_left_from_right": finger_mirror or "true",
        }
    return resolved


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    finger = LaunchConfiguration("finger").perform(context).strip().lower()
    info = _hand_runtime_info(hand)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    resolved_finger_configs = _resolve_enabled_finger_configs(context, scripts_dir, finger)
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    mapping_script = scripts_dir / "curl_m0_m2_mlp_mapping_main.py"
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")
    if not mapping_script.exists():
        raise FileNotFoundError(f"curl_m0_m2_mlp_mapping_main.py not found: {mapping_script}")

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
        cmd=[
            "python3",
            "-u",
            str(glove_script),
        ],
        name="glove_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_glove_node")),
    )

    mapping_cmd = [
        "python3",
        "-u",
        str(mapping_script),
        "--finger",
        finger,
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
        "--device",
        LaunchConfiguration("device").perform(context),
        "--ready-timeout-sec",
        LaunchConfiguration("ready_timeout_sec").perform(context),
    ]
    for resolved_finger in DEFAULT_FINGER_ORDER:
        cfg = resolved_finger_configs[resolved_finger]
        mapping_cmd.extend(
            [
                f"--enable-{resolved_finger}",
                cfg["enable"],
                f"--{resolved_finger}-model-ckpt",
                cfg["model_ckpt"],
                f"--{resolved_finger}-curl-meta",
                cfg["curl_meta"],
                f"--{resolved_finger}-model-motor-ids",
                cfg["model_motor_ids"],
                f"--{resolved_finger}-q-indices",
                cfg["q_indices"],
                f"--{resolved_finger}-mirror-left-from-right",
                cfg["mirror_left_from_right"],
            ]
        )

    use_raw = _bool_from_launch_arg(LaunchConfiguration("right_non_thumb_q_from_raw_skeleton").perform(context))
    if use_raw:
        mapping_cmd.append("--right-non-thumb-q-from-raw-skeleton")
    else:
        mapping_cmd.append("--no-right-non-thumb-q-from-raw-skeleton")

    require_ready = _bool_from_launch_arg(LaunchConfiguration("require_ready").perform(context))
    if require_ready:
        mapping_cmd.append("--require-ready")

    mapping_process = ExecuteProcess(
        cmd=mapping_cmd,
        name=f"{finger}_curl_multihead_mlp_mapping_{hand}",
        output="screen",
        on_exit=Shutdown(reason="launch is shutting down"),
    )

    return [
        LogInfo(
            msg=(
                f"[curl_m0_m2_mlp_mapping_launch] target_hand={hand}, "
                f"finger={finger}, "
                f"driver_node={info['driver_node_name']}, "
                f"start_driver_node={LaunchConfiguration('start_driver_node').perform(context)}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"glove_topic={LaunchConfiguration('glove_topic').perform(context)}, "
                f"glove_raw_skeleton_topic={LaunchConfiguration('glove_raw_skeleton_topic').perform(context)}, "
                f"right_non_thumb_q_from_raw_skeleton={LaunchConfiguration('right_non_thumb_q_from_raw_skeleton').perform(context)}, "
                f"index_enable={resolved_finger_configs['index']['enable']}, "
                f"index_model_ckpt={resolved_finger_configs['index']['model_ckpt']}, "
                f"index_curl_meta={resolved_finger_configs['index']['curl_meta']}, "
                f"index_model_motor_ids={resolved_finger_configs['index']['model_motor_ids']}, "
                f"index_q_indices={resolved_finger_configs['index']['q_indices']}, "
                f"middle_enable={resolved_finger_configs['middle']['enable']}, "
                f"middle_model_ckpt={resolved_finger_configs['middle']['model_ckpt']}, "
                f"middle_curl_meta={resolved_finger_configs['middle']['curl_meta']}, "
                f"middle_model_motor_ids={resolved_finger_configs['middle']['model_motor_ids']}, "
                f"middle_q_indices={resolved_finger_configs['middle']['q_indices']}, "
                f"ring_enable={resolved_finger_configs['ring']['enable']}, "
                f"ring_model_ckpt={resolved_finger_configs['ring']['model_ckpt']}, "
                f"ring_curl_meta={resolved_finger_configs['ring']['curl_meta']}, "
                f"ring_model_motor_ids={resolved_finger_configs['ring']['model_motor_ids']}, "
                f"ring_q_indices={resolved_finger_configs['ring']['q_indices']}, "
                f"little_enable={resolved_finger_configs['little']['enable']}, "
                f"little_model_ckpt={resolved_finger_configs['little']['model_ckpt']}, "
                f"little_curl_meta={resolved_finger_configs['little']['curl_meta']}, "
                f"little_model_motor_ids={resolved_finger_configs['little']['model_motor_ids']}, "
                f"little_q_indices={resolved_finger_configs['little']['q_indices']}"
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
                "finger",
                default_value="ring",
                description="Target finger for mapping control: index/middle/ring/little",
            ),
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Target hand for mapping control: left/right",
            ),
            DeclareLaunchArgument(
                "start_driver_node",
                default_value="true",
                description="Start Y_Hand_Node driver",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Start minimal_example_dual_stream_commented.py glove publisher node",
            ),
            DeclareLaunchArgument(
                "glove_topic",
                default_value="/glove_joint_data",
                description="ROS topic for 48D glove data, 20D glove20, or 4D finger joint angles",
            ),
            DeclareLaunchArgument(
                "glove_raw_skeleton_topic",
                default_value="/glove_joint_data_raw_skeleton",
                description="176D raw-skeleton topic for finger q solving on the selected hand",
            ),
            DeclareLaunchArgument(
                "right_non_thumb_q_from_raw_skeleton",
                default_value="true",
                description="Solve the enabled non-thumb finger q_0..q_3 from raw skeleton instead of glove48 q indices",
            ),
            DeclareLaunchArgument(
                "model_ckpt",
                default_value="",
                description="Legacy selected-finger checkpoint path. Empty uses the selected finger default.",
            ),
            DeclareLaunchArgument(
                "curl_meta",
                default_value="",
                description="Legacy selected-finger curl meta path. Empty uses the selected finger default.",
            ),
            DeclareLaunchArgument(
                "model_motor_ids",
                default_value="",
                description="Legacy selected-finger motor order. Empty uses the selected finger default.",
            ),
            DeclareLaunchArgument(
                "q_indices",
                default_value="",
                description="Legacy selected-finger glove20 q indices. Empty uses the selected finger default.",
            ),
            DeclareLaunchArgument(
                "mirror_left_from_right",
                default_value="true",
                description="Legacy selected-finger mirror setting when target_hand=left",
            ),
            DeclareLaunchArgument("enable_index", default_value="", description="Enable index finger model: true/false"),
            DeclareLaunchArgument("index_model_ckpt", default_value="", description="Index finger checkpoint path"),
            DeclareLaunchArgument("index_curl_meta", default_value="", description="Index finger curl meta path"),
            DeclareLaunchArgument("index_model_motor_ids", default_value="", description="Index finger motor ids csv"),
            DeclareLaunchArgument("index_q_indices", default_value="", description="Index finger q indices csv"),
            DeclareLaunchArgument(
                "index_mirror_left_from_right",
                default_value="",
                description="Index finger mirror setting for left hand",
            ),
            DeclareLaunchArgument("enable_middle", default_value="", description="Enable middle finger model: true/false"),
            DeclareLaunchArgument("middle_model_ckpt", default_value="", description="Middle finger checkpoint path"),
            DeclareLaunchArgument("middle_curl_meta", default_value="", description="Middle finger curl meta path"),
            DeclareLaunchArgument("middle_model_motor_ids", default_value="", description="Middle finger motor ids csv"),
            DeclareLaunchArgument("middle_q_indices", default_value="", description="Middle finger q indices csv"),
            DeclareLaunchArgument(
                "middle_mirror_left_from_right",
                default_value="",
                description="Middle finger mirror setting for left hand",
            ),
            DeclareLaunchArgument("enable_ring", default_value="", description="Enable ring finger model: true/false"),
            DeclareLaunchArgument("ring_model_ckpt", default_value="", description="Ring finger checkpoint path"),
            DeclareLaunchArgument("ring_curl_meta", default_value="", description="Ring finger curl meta path"),
            DeclareLaunchArgument("ring_model_motor_ids", default_value="", description="Ring finger motor ids csv"),
            DeclareLaunchArgument("ring_q_indices", default_value="", description="Ring finger q indices csv"),
            DeclareLaunchArgument(
                "ring_mirror_left_from_right",
                default_value="",
                description="Ring finger mirror setting for left hand",
            ),
            DeclareLaunchArgument("enable_little", default_value="", description="Enable little finger model: true/false"),
            DeclareLaunchArgument("little_model_ckpt", default_value="", description="Little finger checkpoint path"),
            DeclareLaunchArgument("little_curl_meta", default_value="", description="Little finger curl meta path"),
            DeclareLaunchArgument("little_model_motor_ids", default_value="", description="Little finger motor ids csv"),
            DeclareLaunchArgument("little_q_indices", default_value="", description="Little finger q indices csv"),
            DeclareLaunchArgument(
                "little_mirror_left_from_right",
                default_value="",
                description="Little finger mirror setting for left hand",
            ),
            DeclareLaunchArgument(
                "defaults_path",
                default_value=str(scripts_dir / "muscle_adjustment_defaults.py"),
                description="Path to muscle_adjustment_defaults.py",
            ),
            DeclareLaunchArgument(
                "control_period_sec",
                default_value="0.01",
                description="Command publish period (sec)",
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="2.0",
                description="Status print period (sec)",
            ),
            DeclareLaunchArgument(
                "glove_max_hz",
                default_value="100.0",
                description="Max accepted glove callback rate. <=0 disables rate limiting",
            ),
            DeclareLaunchArgument(
                "glove_sub_queue_size",
                default_value="60",
                description="ROS subscription queue size for glove topic",
            ),
            DeclareLaunchArgument(
                "glove_pending_maxlen",
                default_value="60",
                description="Internal queue size for buffered glove frames",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="auto",
                description="Torch device: auto/cpu/cuda",
            ),
            DeclareLaunchArgument(
                "require_ready",
                default_value="false",
                description="If true, mapping script exits when startup topics are not ready within timeout",
            ),
            DeclareLaunchArgument(
                "ready_timeout_sec",
                default_value="12.0",
                description="Ready wait timeout passed to mapping script",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
