import json
from pathlib import Path
import signal

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, Shutdown,
)
from launch.event_handlers import OnShutdown
from launch.events import matches_action
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration


def _load_yhand_dev_config(hand: str) -> str:
    config_file = Path(__file__).resolve().parent.parent / "config" / "devices" / f"{hand}_hand_device_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    lines = [f"PC_IP: {config['pc_ip']}", f"SN: {config['sn']}", "Boards:"]
    for board in config["boards"]:
        lines.extend([f"  - Id: {board['id']}", f"    IP: {board['ip']}"])
        if "port" in board:
            lines.append(f"    Port: {board['port']}")
    return "\n".join(lines) + "\n"


def _launch_setup(context):
    target_hand = LaunchConfiguration('target_hand').perform(context)
    hands = ('left', 'right') if target_hand == 'both' else (target_hand,)
    script = Path(get_package_share_directory('yhand_mapping')) / 'scripts' / 'yhand_pretighten.py'
    demo = Path(get_package_prefix('yhand_mapping')) / 'lib' / 'yhand_mapping' / 'Y_Hand_M2_Demo'
    sdk_config = Path(get_package_share_directory('devices_pkg')) / 'YAML' / 'Y_Hand_M2' / 'out' / 'TOP.yaml'
    node_names = {'left': 'y_hand_node', 'right': 'y_hand_node_2'}
    node_list = []
    for hand in hands:
        node_name = node_names[hand]
        node_list.append(ExecuteProcess(
            cmd=['python3', str(script), '--demo-executable', str(demo),
                 '--sdk-config', str(sdk_config), '--hand', hand,
                 '--zero-tolerance', LaunchConfiguration('zero_tolerance'),
                 '--zero-timeout', LaunchConfiguration('zero_timeout'),
                 '--dev-config', _load_yhand_dev_config(hand)],
            name=f'{node_name}_pretighten',
            output='screen',
            # Allow the SDK demo to release position control before terminating.
            sigterm_timeout='6.0',
            sigkill_timeout='1.0',
            on_exit=Shutdown(reason=f"{node_name} pretighten controller exited"),
        ))

    def interrupt_controllers(event, context):
        # Also handle SIGINT sent to the launch parent alone in interactive mode.
        if not event.due_to_sigint or context.noninteractive:
            return []
        return [EmitEvent(event=SignalProcess(
            signal_number=signal.SIGINT, process_matcher=matches_action(node),
        )) for node in node_list if node.process_details and node.process_details.get('pid')]

    return [
        RegisterEventHandler(OnShutdown(on_shutdown=interrupt_controllers)),
        *node_list,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'target_hand',
            default_value='both',
            choices=['left', 'right', 'both'],
            description='Hand whose 28 motors should be commanded to absolute position zero.',
        ),
        DeclareLaunchArgument('zero_tolerance', default_value='1.0',
                              description='Maximum absolute position error in degrees.'),
        DeclareLaunchArgument('zero_timeout', default_value='10.0',
                              description='Seconds from SDK startup to confirm all motors at zero.'),
        OpaqueFunction(function=_launch_setup),
    ])
