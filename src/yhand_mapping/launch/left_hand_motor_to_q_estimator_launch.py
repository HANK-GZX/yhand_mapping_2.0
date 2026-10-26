"""Read-only five-finger state estimation alongside the existing teleoperator."""
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def setup(context):
    package = Path(get_package_share_directory('yhand_mapping'))
    value = lambda name: LaunchConfiguration(name).perform(context)
    cmd = [sys.executable, str(package/'scripts/q_estimator/left_hand_motor_to_q_estimator.py'), 'run']
    for name in ['finger', 'feedback_topic', 'model_dir', 'model', 'publish_hz', 'feedback_timeout']:
        cmd += ['--'+name.replace('_', '-'), value(name)]
    for name in ['q_topic', 'relative_q_topic', 'joint_state_topic', 'valid_topic', 'status_topic']:
        if value(name).strip():
            cmd += ['--'+name.replace('_', '-'), value(name)]
    for name in ['q_zero', 'motor_offset']:
        if value(name).strip():
            cmd += ['--'+name.replace('_', '-'), *value(name).replace(',', ' ').split()]
    if value('allow_extrapolation') == 'true':
        cmd += ['--allow-extrapolation']
    for name in ['q_cmd_topic', 'glove_topic', 'glove_raw_topic', 'robot_glove_topic', 'log_dir', 'log_max_age']:
        setting = LaunchConfiguration(name, default='').perform(context)
        if setting.strip() or name in ['glove_raw_topic', 'robot_glove_topic']:
            cmd += ['--'+name.replace('_', '-'), setting]
    if LaunchConfiguration('log_enabled', default='true').perform(context) == 'false':
        cmd += ['--no-log']
    return [ExecuteProcess(cmd=cmd, output='screen', emulate_tty=True)]


def generate_launch_description():
    package = Path(get_package_share_directory('yhand_mapping'))
    return LaunchDescription([
        DeclareLaunchArgument('finger', default_value='five', choices=['five', 'thumb', 'index', 'middle', 'ring', 'little']),
        DeclareLaunchArgument('feedback_topic', default_value='/yhand_left_feedback'),
        DeclareLaunchArgument('model_dir', default_value=str(package/'config')),
        DeclareLaunchArgument('model', default_value='auto', choices=['auto', 'linear', 'quadratic']),
        DeclareLaunchArgument('publish_hz', default_value='50'),
        DeclareLaunchArgument('feedback_timeout', default_value='0.5'),
        *[DeclareLaunchArgument(name, default_value='') for name in ['q_topic', 'relative_q_topic',
            'joint_state_topic', 'valid_topic', 'status_topic', 'q_zero', 'motor_offset']],
        DeclareLaunchArgument('allow_extrapolation', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('log_enabled', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('log_dir', default_value=''),
        DeclareLaunchArgument('q_cmd_topic', default_value=''),
        DeclareLaunchArgument('glove_topic', default_value='/glove_joint_data'),
        DeclareLaunchArgument('glove_raw_topic', default_value='/glove_joint_data_raw_skeleton'),
        DeclareLaunchArgument('robot_glove_topic', default_value=''),
        DeclareLaunchArgument('log_max_age', default_value='0.1'),
        OpaqueFunction(function=setup)])
