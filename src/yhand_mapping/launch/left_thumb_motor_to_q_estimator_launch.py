"""Launch only the independent feedback estimator; reuse the running hand driver."""
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def setup(context):
    package = Path(get_package_share_directory('yhand_mapping'))
    value = lambda name: LaunchConfiguration(name).perform(context)
    command = [sys.executable, str(package/'scripts/q_estimator/left_thumb_motor_to_q_estimator.py'), 'run']
    for name in ['feedback_topic', 'model_file', 'model', 'q_topic', 'relative_q_topic',
                 'joint_state_topic', 'valid_topic', 'status_topic', 'publish_hz', 'feedback_timeout']:
        command += ['--'+name.replace('_', '-'), value(name)]
    if value('allow_extrapolation') == 'true':
        command += ['--allow-extrapolation']
    return [ExecuteProcess(cmd=command, output='screen', emulate_tty=True)]


def generate_launch_description():
    package = Path(get_package_share_directory('yhand_mapping'))
    defaults = {
        'feedback_topic': '/yhand_left_feedback',
        'model_file': str(package/'config/left_thumb_motor_to_q_estimator.json'),
        'model': 'quadratic', 'q_topic': '/left_thumb_estimated_q',
        'relative_q_topic': '/left_thumb_estimated_q_relative',
        'joint_state_topic': '/left_thumb_estimated_joint_states',
        'valid_topic': '/left_thumb_estimate_valid', 'status_topic': '/left_thumb_estimate_status',
        'publish_hz': '50', 'feedback_timeout': '0.5'}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        DeclareLaunchArgument('allow_extrapolation', default_value='false', choices=['true', 'false'],
                              description='Publish out-of-range diagnostic q with valid=false'),
        OpaqueFunction(function=setup)])
