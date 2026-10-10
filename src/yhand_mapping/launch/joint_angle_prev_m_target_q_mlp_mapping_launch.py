import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    Shutdown,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
INPUT_TERMINAL_START_DELAY_SEC = 2.0


def _bool_from_launch_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean launch argument: {value!r}")


def _launch_input_terminal(context):
    enabled = _bool_from_launch_arg(
        LaunchConfiguration("start_angle_input_terminal").perform(context)
    )
    if not enabled:
        return []

    gnome_terminal = shutil.which("gnome-terminal")
    if gnome_terminal is None:
        raise RuntimeError(
            "start_angle_input_terminal=true but gnome-terminal was not found."
        )

    input_script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "joint_angle_terminal_input.py"
    )
    if not input_script.exists():
        raise FileNotFoundError(f"Joint-angle input script not found: {input_script}")

    hand = LaunchConfiguration("target_hand").perform(context)
    finger = LaunchConfiguration("target_finger").perform(context)
    topic = LaunchConfiguration("joint_angle_topic").perform(context)
    publish_hz = LaunchConfiguration("input_publish_hz").perform(context)
    burst_count = LaunchConfiguration("input_publish_burst_count").perform(context)

    input_process = ExecuteProcess(
        cmd=[
            gnome_terminal,
            "--wait",
            "--title",
            f"joint_angle_input_{hand}_{finger}",
            "--",
            "python3",
            "-u",
            str(input_script),
            "--hand",
            hand,
            "--finger",
            finger,
            "--topic",
            topic,
            "--publish-hz",
            publish_hz,
            "--burst-count",
            burst_count,
        ],
        name=f"joint_angle_input_{hand}_{finger}",
        output="screen",
        on_exit=Shutdown(reason="joint-angle input terminal closed"),
    )
    return [
        TimerAction(
            period=INPUT_TERMINAL_START_DELAY_SEC,
            actions=[input_process],
        )
    ]


def generate_launch_description():
    base_launch = Path(__file__).resolve().parent / "thumb_prev_m_target_q_mlp_mapping_launch.py"
    if not base_launch.exists():
        raise FileNotFoundError(f"Base mapping launch not found: {base_launch}")

    target_hand = LaunchConfiguration("target_hand")
    target_finger = LaunchConfiguration("target_finger")
    joint_angle_topic = LaunchConfiguration("joint_angle_topic")
    selected_model_ckpt = LaunchConfiguration("selected_model_ckpt")
    start_driver_node = LaunchConfiguration("start_driver_node")
    right_non_thumb_q_from_raw_skeleton = LaunchConfiguration(
        "right_non_thumb_q_from_raw_skeleton"
    )

    finger_enable_args = {
        f"enable_{finger_name}": PythonExpression(
            ["'", target_finger, f"' == '{finger_name}'"]
        )
        for finger_name in FINGER_NAMES
    }
    finger_model_args = {
        f"{finger_name}_model_ckpt": selected_model_ckpt
        for finger_name in FINGER_NAMES
    }
    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(base_launch)),
        launch_arguments={
            "target_hand": target_hand,
            "start_driver_node": start_driver_node,
            "start_glove_node": "false",
            "glove_topic": joint_angle_topic,
            "right_index_q_from_raw_skeleton": right_non_thumb_q_from_raw_skeleton,
            # Disable the legacy thumb alias so target_finger is authoritative.
            "enable_model": "",
            "model_ckpt": "",
            **finger_enable_args,
            **finger_model_args,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Hand controlled by the selected finger's 4D joint-angle input",
                choices=["left", "right"],
            ),
            DeclareLaunchArgument(
                "target_finger",
                default_value="thumb",
                description="Single finger controlled by the 4D joint-angle input",
                choices=list(FINGER_NAMES),
            ),
            DeclareLaunchArgument(
                "joint_angle_topic",
                default_value="/target_joint_angles",
                description="std_msgs/msg/Float64MultiArray topic containing four angles for target_finger",
            ),
            DeclareLaunchArgument(
                "selected_model_ckpt",
                default_value="",
                description="Checkpoint for target_finger; empty uses the base launch default",
            ),
            DeclareLaunchArgument(
                "start_driver_node",
                default_value="true",
                description="Whether to start the Y Hand driver node",
            ),
            DeclareLaunchArgument(
                "right_non_thumb_q_from_raw_skeleton",
                default_value="false",
                description=(
                    "Whether right-hand non-thumb fingers (index/middle/ring/little) "
                    "should ignore the 4D joint-angle input and instead use raw skeleton q."
                ),
            ),
            DeclareLaunchArgument(
                "start_angle_input_terminal",
                default_value="true",
                description="Open an interactive gnome-terminal for entering four joint angles",
            ),
            DeclareLaunchArgument(
                "input_publish_hz",
                default_value="50.0",
                description="Rate used for the short message burst after each terminal input",
            ),
            DeclareLaunchArgument(
                "input_publish_burst_count",
                default_value="5",
                description="Number of frames sent for each entered joint-angle target",
            ),
            LogInfo(
                msg=[
                    "[joint_angle_prev_m_target_q_mlp_mapping_launch] hand=",
                    target_hand,
                    ", finger=",
                    target_finger,
                    ", joint_angle_topic=",
                    joint_angle_topic,
                    ", right_non_thumb_q_from_raw_skeleton=",
                    right_non_thumb_q_from_raw_skeleton,
                    ", expected_layout=selected-finger-4D, other_joint_angles=0",
                ]
            ),
            mapping_launch,
            OpaqueFunction(function=_launch_input_terminal),
        ]
    )
