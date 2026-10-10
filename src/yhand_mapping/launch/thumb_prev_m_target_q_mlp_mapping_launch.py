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
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")

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
    "left": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
    "right": {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0,
        16: 0.0,
        17: 0.0,
        18: 0.0,
        19: 0.0,
    },
}

MOTOR_CMD_OFFSET_MAP_DEFAULT = {
    "left": {mid: 0.0 for mid in sorted(MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT.keys())},
    "right": {mid: 0.0 for mid in sorted(MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT.keys())},
}

FINGER_DEFAULTS_BY_HAND = {
    "left": {
        "thumb": {
            "enable": "false",
            "ckpt_rel_parts": None,
            "motor_ids": model_motor_ids_csv('left', 'thumb'),
            "q_indices": "0,1,2,3",
            "mirror_right_from_left": "false",
        },
        "index": {
            "enable": "false",
            "ckpt_rel_parts": None,
            "motor_ids": model_motor_ids_csv('left', 'index'),
            "q_indices": "4,5,6,7",
            "mirror_right_from_left": "false",
        },
        "middle": {
            "enable": "false",
            "ckpt_rel_parts": None,
            "motor_ids": model_motor_ids_csv('left', 'middle'),
            "q_indices": "8,9,10,11",
            "mirror_right_from_left": "false",
        },
        "ring": {
            "enable": "false",
            "ckpt_rel_parts": None,
            "motor_ids": model_motor_ids_csv('left', 'ring'),
            "q_indices": "12,13,14,15",
            "mirror_right_from_left": "false",
        },
        "little": {
            "enable": "false",
            "ckpt_rel_parts": None,
            "motor_ids": model_motor_ids_csv('left', 'little'),
            "q_indices": "16,17,18,19",
            "mirror_right_from_left": "false",
        },
    },
    "right": {
        "thumb": {
            "enable": "true",
            "ckpt_rel_parts": (
                "mp_right",
                "thumb_delta_multihead_masked",
                "best_thumb_prev_m_target_q_mlp.pt",
            ),
            "motor_ids": model_motor_ids_csv('right', 'thumb'),
            "q_indices": "0,1,2,3",
            "mirror_right_from_left": "false",
        },
        "index": {
            "enable": "false",
            "ckpt_rel_parts": (
                "mp_right",
                "index_delta_multihead_masked",
                "best_index_prev_m_target_q_mlp.pt",
            ),
            "motor_ids": model_motor_ids_csv('right', 'index'),
            "q_indices": "4,5,6,7",
            "mirror_right_from_left": "false",
        },
        "middle": {
            "enable": "false",
            "ckpt_rel_parts": (
                "mp_right",
                "middle_delta_multihead_masked",
                "best_lstm_mlp.pt",
            ),
            "motor_ids": model_motor_ids_csv('right', 'middle'),
            "q_indices": "8,9,10,11",
            "mirror_right_from_left": "false",
        },
        "ring": {
            "enable": "false",
            "ckpt_rel_parts": (
                "mp_right",
                "ring_delta_multihead_masked",
                "best_ring_prev_m_target_q_mlp.pt",
            ),
            "motor_ids": model_motor_ids_csv('right', 'ring'),
            "q_indices": "12,13,14,15",
            "mirror_right_from_left": "false",
        },
        "little": {
            "enable": "false",
            "ckpt_rel_parts": (
                "mp_right",
                "little_delta_multihead_masked",
                "best_little_prev_m_target_q_mlp.pt",
            ),
            "motor_ids": model_motor_ids_csv('right', 'little'),
            "q_indices": "16,17,18,19",
            "mirror_right_from_left": "false",
        },
    },
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
            "driver_node_name": "y_hand_left_prev_m_target_q_mlp_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_prev_m_target_q_mlp_node",
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


def _resolve_finger_arg_or_default(context, hand: str, scripts_dir: Path, finger_name: str, key: str, launch_arg_name: str):
    raw = LaunchConfiguration(launch_arg_name).perform(context)
    val = str(raw).strip()
    if val:
        return val

    cfg = FINGER_DEFAULTS_BY_HAND[hand][finger_name]
    if key == "ckpt":
        rel_parts = cfg["ckpt_rel_parts"]
        return "" if rel_parts is None else str(scripts_dir.joinpath(*rel_parts))
    return str(cfg[key])


def _legacy_thumb_arg(context, legacy_name: str):
    return LaunchConfiguration(legacy_name).perform(context).strip()


def _append_finger_args_to_mapping_cmd(mapping_cmd, context, hand: str, scripts_dir: Path):
    for finger_name in FINGER_ORDER:
        enable_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="enable",
            launch_arg_name=f"enable_{finger_name}",
        )
        ckpt_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="ckpt",
            launch_arg_name=f"{finger_name}_model_ckpt",
        )
        motor_ids_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="motor_ids",
            launch_arg_name=f"{finger_name}_model_motor_ids",
        )
        q_indices_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="q_indices",
            launch_arg_name=f"{finger_name}_q_indices",
        )
        mirror_val = _resolve_finger_arg_or_default(
            context=context,
            hand=hand,
            scripts_dir=scripts_dir,
            finger_name=finger_name,
            key="mirror_right_from_left",
            launch_arg_name=f"{finger_name}_mirror_right_from_left",
        )

        if finger_name == "thumb":
            enable_val = _legacy_thumb_arg(context, "enable_model") or enable_val
            ckpt_val = _legacy_thumb_arg(context, "model_ckpt") or ckpt_val
            motor_ids_val = _legacy_thumb_arg(context, "model_motor_ids") or motor_ids_val
            q_indices_val = _legacy_thumb_arg(context, "q_indices") or q_indices_val
            mirror_val = _legacy_thumb_arg(context, "mirror_right_from_left") or mirror_val

        mapping_cmd.extend(
            [
                f"--enable-{finger_name}",
                enable_val,
                f"--{finger_name}-model-ckpt",
                ckpt_val,
                f"--{finger_name}-model-motor-ids",
                motor_ids_val,
                f"--{finger_name}-q-indices",
                q_indices_val,
                f"--{finger_name}-mirror-right-from-left",
                mirror_val,
            ]
        )


def _declare_finger_launch_args():
    actions = []
    for finger_name in FINGER_ORDER:
        actions.extend(
            [
                DeclareLaunchArgument(f"enable_{finger_name}", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_model_ckpt", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_model_motor_ids", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_q_indices", default_value=""),
                DeclareLaunchArgument(f"{finger_name}_mirror_right_from_left", default_value=""),
            ]
        )
    return actions


def _launch_setup(context):
    target_hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    if target_hand == "both":
        hands = ("left", "right")
    elif target_hand in ("left", "right"):
        hands = (target_hand,)
    else:
        raise ValueError(f"Unsupported target_hand={target_hand}, expected 'left'|'right'|'both'.")

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    mapping_script = scripts_dir / "thumb_prev_m_target_q_mlp_mapping_main.py"
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")
    if not mapping_script.exists():
        raise FileNotFoundError(f"thumb_prev_m_target_q_mlp_mapping_main.py not found: {mapping_script}")

    glove_process = ExecuteProcess(
        cmd=["python3", "-u", str(glove_script)],
        name="glove_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_glove_node")),
    )

    require_ready = _bool_from_launch_arg(LaunchConfiguration("require_ready").perform(context))
    driver_processes = []
    mapping_processes = []
    hand_summaries = []

    for hand in hands:
        info = _hand_runtime_info(hand)
        dev_config = _load_yhand_dev_config(info["config_var"])

        driver_processes.append(
            ExecuteProcess(
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
            "--defaults-path",
            LaunchConfiguration("defaults_path").perform(context),
        ]
        _append_finger_args_to_mapping_cmd(mapping_cmd, context=context, hand=hand, scripts_dir=scripts_dir)
        mapping_cmd.extend(
            [
                "--control-period-sec",
                LaunchConfiguration("control_period_sec").perform(context),
                "--status-period-sec",
                LaunchConfiguration("status_period_sec").perform(context),
                "--glove-raw-skeleton-topic",
                LaunchConfiguration("glove_raw_skeleton_topic").perform(context),
                "--glove-max-hz",
                LaunchConfiguration("glove_max_hz").perform(context),
                "--glove-sub-queue-size",
                LaunchConfiguration("glove_sub_queue_size").perform(context),
                "--glove-pending-maxlen",
                LaunchConfiguration("glove_pending_maxlen").perform(context),
                "--right-non-thumb-q-from-raw-skeleton",
                LaunchConfiguration("right_index_q_from_raw_skeleton").perform(context),
                "--motor-feedback-limits-left-json",
                LaunchConfiguration("motor_feedback_limits_left_json").perform(context),
                "--motor-feedback-limits-right-json",
                LaunchConfiguration("motor_feedback_limits_right_json").perform(context),
                "--q-offset-map-json",
                LaunchConfiguration("q_offset_map_json").perform(context),
                "--motor-cmd-offset-map-json",
                LaunchConfiguration("motor_cmd_offset_map_json").perform(context),
                "--device",
                LaunchConfiguration("device").perform(context),
                "--ready-timeout-sec",
                LaunchConfiguration("ready_timeout_sec").perform(context),
            ]
        )
        if require_ready:
            mapping_cmd.append("--require-ready")

        mapping_processes.append(
            ExecuteProcess(
                cmd=mapping_cmd,
                name=f"prev_m_target_q_mlp_mapping_{hand}",
                output="screen",
                on_exit=Shutdown(reason="launch is shutting down"),
            )
        )

        enabled = []
        for finger_name in FINGER_ORDER:
            enable_val = _resolve_finger_arg_or_default(
                context=context,
                hand=hand,
                scripts_dir=scripts_dir,
                finger_name=finger_name,
                key="enable",
                launch_arg_name=f"enable_{finger_name}",
            )
            if finger_name == "thumb":
                enable_val = _legacy_thumb_arg(context, "enable_model") or enable_val
            if _bool_from_launch_arg(enable_val):
                enabled.append(finger_name)
        hand_summaries.append(f"{hand}:enabled={enabled}")

    return [
        LogInfo(
            msg=(
                f"[thumb_prev_m_target_q_mlp_mapping_launch] target_hand={target_hand}, hands={list(hands)}, "
                f"start_driver_node={LaunchConfiguration('start_driver_node').perform(context)}, "
                f"start_glove_node={LaunchConfiguration('start_glove_node').perform(context)}, "
                f"glove_topic={LaunchConfiguration('glove_topic').perform(context)}, "
                f"glove_raw_skeleton_topic={LaunchConfiguration('glove_raw_skeleton_topic').perform(context)}, "
                f"glove_max_hz={LaunchConfiguration('glove_max_hz').perform(context)}, "
                f"glove_sub_queue_size={LaunchConfiguration('glove_sub_queue_size').perform(context)}, "
                f"glove_pending_maxlen={LaunchConfiguration('glove_pending_maxlen').perform(context)}, "
                f"right_index_q_from_raw_skeleton={LaunchConfiguration('right_index_q_from_raw_skeleton').perform(context)}, "
                f"{'; '.join(hand_summaries)}"
            )
        ),
        *driver_processes,
        glove_process,
        TimerAction(period=CONTROL_SCRIPT_START_DELAY_SEC, actions=mapping_processes),
    ]


def generate_launch_description():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    launch_args = [
        DeclareLaunchArgument(
            "target_hand",
            default_value="right",
            description="left | right | both",
        ),
        DeclareLaunchArgument("start_driver_node", default_value="true"),
        DeclareLaunchArgument("start_glove_node", default_value="true"),
        DeclareLaunchArgument("glove_topic", default_value="/glove_joint_data"),
        DeclareLaunchArgument("glove_raw_skeleton_topic", default_value="/glove_joint_data_raw_skeleton"),
        DeclareLaunchArgument(
            "defaults_path",
            default_value=str(scripts_dir / "muscle_adjustment_defaults.py"),
        ),
        # Backward-compatible thumb aliases from the thumb-only launch.
        DeclareLaunchArgument("enable_model", default_value=""),
        DeclareLaunchArgument("model_ckpt", default_value=""),
        DeclareLaunchArgument("model_motor_ids", default_value=""),
        DeclareLaunchArgument("q_indices", default_value=""),
        DeclareLaunchArgument("mirror_right_from_left", default_value=""),
    ]

    launch_args.extend(_declare_finger_launch_args())
    launch_args.extend(
        [
            DeclareLaunchArgument("control_period_sec", default_value="0.01"),
            DeclareLaunchArgument("status_period_sec", default_value="2.0"),
            DeclareLaunchArgument(
                "glove_max_hz",
                default_value="100.0",
                description="Max accepted glove callback rate. <=0 disables rate limiting",
            ),
            DeclareLaunchArgument("glove_sub_queue_size", default_value="60"),
            DeclareLaunchArgument("glove_pending_maxlen", default_value="60"),
            DeclareLaunchArgument(
                "right_index_q_from_raw_skeleton",
                default_value="true",
                description=(
                    "If true, right-hand non-thumb fingers (index/middle/ring/little) q_0..q_3 "
                    "are solved from raw skeleton; thumb still uses glove q_indices."
                ),
            ),
            DeclareLaunchArgument(
                "motor_feedback_limits_left_json",
                default_value=json.dumps(MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT, ensure_ascii=False),
                description="Left-hand motor position limits JSON: {\"motor_id\": [lo, hi], ...}",
            ),
            DeclareLaunchArgument(
                "motor_feedback_limits_right_json",
                default_value=json.dumps(MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT, ensure_ascii=False),
                description="Right-hand motor position limits JSON: {\"motor_id\": [lo, hi], ...}",
            ),
            DeclareLaunchArgument(
                "q_offset_map_json",
                default_value=json.dumps(Q_OFFSET_MAP_DEFAULT, ensure_ascii=False),
                description="Glove q offset map JSON by hand and q_index",
            ),
            DeclareLaunchArgument(
                "motor_cmd_offset_map_json",
                default_value=json.dumps(MOTOR_CMD_OFFSET_MAP_DEFAULT, ensure_ascii=False),
                description="Motor command offset map JSON by hand and motor_id",
            ),
            DeclareLaunchArgument("device", default_value="auto"),
            DeclareLaunchArgument("require_ready", default_value="false"),
            DeclareLaunchArgument("ready_timeout_sec", default_value="12.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
    return LaunchDescription(launch_args)
