import json
from pathlib import Path
import signal

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnShutdown
from launch.events import matches_action
from launch.events.process import SignalProcess


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


def generate_launch_description():
    config_dev1 = _load_yhand_dev_config("config_dev1")
    config_dev2 = _load_yhand_dev_config("config_dev2")
    script = Path(get_package_share_directory('yhand_mapping')) / 'scripts' / 'yhand_pretighten.py'
    driver = Path(get_package_prefix('devices_pkg')) / 'lib' / 'devices_pkg' / 'Y_Hand_Node'
    node_mapping = {
        'y_hand_node': ('yhand_left_cmd', 'yhand_left_feedback', config_dev1),
        # 'y_hand_node_2': ('yhand_right_cmd', 'yhand_right_feedback', config_dev2),
    }
    node_list = []
    for node_name, (cmd_topic, feedback_topic, dev_config) in node_mapping.items():
        node_list.append(ExecuteProcess(
            cmd=['python3', str(script), '--driver-executable', str(driver),
                 '--driver-node-name', node_name, '--cmd-topic', cmd_topic,
                 '--feedback-topic', feedback_topic, '--dev-config', dev_config],
            name=f'{node_name}_pretighten',
            output='screen',
            # SDK Set_Zero 最坏逐路等待约 84 秒；留出反馈检查及驱动清理时间。
            sigterm_timeout='100.0',
            sigkill_timeout='1.0',
            on_exit=Shutdown(reason=f"{node_name} pretighten controller exited"),
        ))
    def interrupt_controllers(event, context):
        # Also cover SIGINT delivered only to the launch parent (for example kill -INT).
        if not event.due_to_sigint:
            return []
        return [EmitEvent(event=SignalProcess(
            signal_number=signal.SIGINT, process_matcher=matches_action(node),
        )) for node in node_list if node.process_details and node.process_details.get('pid')]

    return LaunchDescription([
        RegisterEventHandler(OnShutdown(on_shutdown=interrupt_controllers)),
        *node_list,
    ])
