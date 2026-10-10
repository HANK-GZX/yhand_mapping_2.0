from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    glove_script = Path(__file__).resolve().parent.parent / "scripts" / "minimal_example_dual_stream_commented.py"
    if not glove_script.exists():
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")

    glove_process = ExecuteProcess(
        cmd=["python3", "-u", str(glove_script)],
        name="glove_publisher",
        output="screen",
    )

    return LaunchDescription([glove_process])
