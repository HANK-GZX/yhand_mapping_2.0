import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration


CONTROL_SCRIPT_START_DELAY_SEC = 1.5
REPLAY_VERIFY_GLOVE_Q_XLSX_WRAPPER_TEMPLATE = r"""
import importlib.util
import pathlib
import sys
import time

import rclpy

script_path = pathlib.Path(sys.argv[1]).resolve()
glove_topic = str(sys.argv[2]).strip()
glove_q_output_dir_raw = str(sys.argv[3]).strip()
args = list(sys.argv[4:])
sys.path.insert(0, str(script_path.parent))

spec = importlib.util.spec_from_file_location("replay_runtime", str(script_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_ACTIVE_NODE = {"node": None}
_NAN = float("nan")


def _normalize48(raw_vals):
    out = []
    vals = list(raw_vals) if raw_vals is not None else []
    for i in range(48):
        try:
            out.append(float(vals[i]))
        except Exception:
            out.append(_NAN)
    return out


_original_init = module.ReplayVerifyNode.__init__


def _patched_init(self, *call_args, **call_kwargs):
    _original_init(self, *call_args, **call_kwargs)
    self.glove48_topic = glove_topic
    self.glove48_received = False
    self.glove48_seq = 0
    self.last_glove48 = [_NAN] * 48
    self.last_glove48_stamp = 0.0
    self.last_point_glove48_avg = [_NAN] * 48
    self.glove48_subscriber = self.create_subscription(
        module.Float64MultiArray, self.glove48_topic, self._on_glove48, 10
    )
    _ACTIVE_NODE["node"] = self


def _on_glove48(self, msg):
    data = list(msg.data) if getattr(msg, "data", None) is not None else []
    vals = _normalize48(data[:48])
    with self._lock:
        self.last_glove48 = vals
        self.glove48_received = True
        self.glove48_seq += 1
        self.last_glove48_stamp = time.monotonic()


def _collect_glove48_samples(self, sample_count: int, timeout_sec: float):
    need = max(1, int(sample_count))
    samples = []
    with self._lock:
        last_seq = int(self.glove48_seq)
        had_glove = bool(self.glove48_received)
        latest = list(self.last_glove48)
    if need <= 1 and had_glove:
        return [latest]
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while len(samples) < need and time.monotonic() < deadline:
        with self._lock:
            cur_seq = int(self.glove48_seq)
            cur = list(self.last_glove48)
            cur_has = bool(self.glove48_received)
        if cur_has and cur_seq > last_seq:
            samples.append(cur)
            last_seq = cur_seq
        time.sleep(0.005)
    if (not samples) and had_glove:
        samples.append(latest)
    return samples


_original_collect_glove_samples = module.ReplayVerifyNode.collect_glove_samples


def _patched_collect_glove_samples(self, *call_args, **call_kwargs):
    raw_samples = _original_collect_glove_samples(self, *call_args, **call_kwargs)
    try:
        if len(call_args) >= 1:
            sample_count = int(call_args[0])
        else:
            sample_count = int(call_kwargs.get("sample_count", 1))
        if len(call_args) >= 2:
            timeout_sec = float(call_args[1])
        else:
            timeout_sec = float(call_kwargs.get("timeout_sec", 1.0))
        q_samples = self.collect_glove48_samples(sample_count=sample_count, timeout_sec=timeout_sec)
        q_avg = module._average_vectors(q_samples, expected_len=48)
        with self._lock:
            self.last_point_glove48_avg = _normalize48(q_avg)
    except Exception:
        with self._lock:
            self.last_point_glove48_avg = [_NAN] * 48
    return raw_samples


_original_write_single_row_xlsx = module._write_single_row_xlsx


def _patched_write_single_row_xlsx(path, row):
    _original_write_single_row_xlsx(path, row)
    try:
        p = pathlib.Path(path).resolve()
        if p.parent.name != "recorded_glove_skeleton":
            return
        node = _ACTIVE_NODE.get("node")
        if node is None:
            return
        if glove_q_output_dir_raw:
            q_dir = pathlib.Path(glove_q_output_dir_raw).expanduser().resolve()
        else:
            q_dir = p.parent.parent / "glove_data"
        q_dir.mkdir(parents=True, exist_ok=True)
        with node._lock:
            q_row = list(getattr(node, "last_point_glove48_avg", [_NAN] * 48))
        q_row = _normalize48(q_row)
        q_path = q_dir / p.name
        _original_write_single_row_xlsx(q_path, q_row)
    except Exception as exc:
        print(f"[replay-record] WARN failed to write glove q xlsx for {path}: {exc}")


module.ReplayVerifyNode.__init__ = _patched_init
module.ReplayVerifyNode._on_glove48 = _on_glove48
module.ReplayVerifyNode.collect_glove48_samples = _collect_glove48_samples
module.ReplayVerifyNode.collect_glove_samples = _patched_collect_glove_samples
module._write_single_row_xlsx = _patched_write_single_row_xlsx

sys.argv = [str(script_path)] + args
module.main()
"""


def _find_workspace_root() -> Path:
    launch_file = Path(__file__).resolve()
    for parent in launch_file.parents:
        if (parent / "src" / "yhand_mapping").is_dir():
            return parent
    raise RuntimeError(f"Cannot locate ROS2 workspace root from launch file: {launch_file}")


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
            "driver_node_name": "y_hand_left_24_motor_replay_node",
            "feedback_topic": "/yhand_left_feedback",
            "command_topic": "/yhand_left_cmd",
        }
    if hand == "right":
        return {
            "config_var": "config_dev2",
            "driver_node_name": "y_hand_right_24_motor_replay_node",
            "feedback_topic": "/yhand_right_feedback",
            "command_topic": "/yhand_right_cmd",
        }
    raise ValueError(f"Unsupported target_hand={hand}, expected 'left' or 'right'.")


def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (workspace_root / p).resolve()


def _launch_setup(context):
    hand = LaunchConfiguration("target_hand").perform(context).strip().lower()
    start_driver_node_raw = LaunchConfiguration("start_driver_node").perform(context).strip()
    start_glove_node_raw = LaunchConfiguration("start_glove_node").perform(context).strip()
    folder_raw = LaunchConfiguration("folder").perform(context).strip()
    single_file = LaunchConfiguration("single_file").perform(context).strip()
    defaults_path_raw = LaunchConfiguration("defaults_path").perform(context).strip()
    cmd_topic_override = LaunchConfiguration("cmd_topic").perform(context).strip()
    feedback_topic_override = LaunchConfiguration("feedback_topic").perform(context).strip()
    exclude_motor_ids = LaunchConfiguration("exclude_motor_ids").perform(context).strip()
    frame_interval_sec = LaunchConfiguration("frame_interval_sec").perform(context).strip()
    control_period_sec = LaunchConfiguration("control_period_sec").perform(context).strip()
    status_period_sec = LaunchConfiguration("status_period_sec").perform(context).strip()
    reach_threshold = LaunchConfiguration("reach_threshold").perform(context).strip()
    arrival_confirm_frames = LaunchConfiguration("arrival_confirm_frames").perform(context).strip()
    arrival_timeout_sec = LaunchConfiguration("arrival_timeout_sec").perform(context).strip()
    feedback_stale_sec = LaunchConfiguration("feedback_stale_sec").perform(context).strip()
    unreached_max_log = LaunchConfiguration("unreached_max_log").perform(context).strip()
    glove_topic = LaunchConfiguration("glove_topic").perform(context).strip()
    glove_raw_skeleton_topic = LaunchConfiguration("glove_raw_skeleton_topic").perform(context).strip()
    record_glove_q_raw = LaunchConfiguration("record_glove_q").perform(context).strip()
    glove_q_csv_path_raw = LaunchConfiguration("glove_q_csv_path").perform(context).strip()
    log_publish_full_raw = LaunchConfiguration("log_publish_full").perform(context).strip()
    publish_idle_on_exit_raw = LaunchConfiguration("publish_idle_on_exit").perform(context).strip()
    verify_mode_raw = LaunchConfiguration("verify_mode").perform(context).strip()
    verify_motor_feedback_dir_raw = LaunchConfiguration("verify_motor_feedback_dir").perform(context).strip()
    verify_glove_skeleton_dir_raw = LaunchConfiguration("verify_glove_skeleton_dir").perform(context).strip()
    verify_output_dir_raw = LaunchConfiguration("verify_output_dir").perform(context).strip()
    ready_timeout_sec = LaunchConfiguration("ready_timeout_sec").perform(context).strip()
    verify_glove_sample_count = LaunchConfiguration("verify_glove_sample_count").perform(context).strip()
    verify_glove_sample_timeout_sec = LaunchConfiguration("verify_glove_sample_timeout_sec").perform(context).strip()
    verify_post_reach_wait_sec = LaunchConfiguration("verify_post_reach_wait_sec").perform(context).strip()
    shutdown_on_replay_exit_raw = LaunchConfiguration("shutdown_on_replay_exit").perform(context).strip()
    loop_raw = LaunchConfiguration("loop").perform(context).strip()
    verbose_raw = LaunchConfiguration("verbose").perform(context).strip()
    start_delay_sec = LaunchConfiguration("control_script_start_delay_sec").perform(context).strip()

    info = _hand_runtime_info(hand)
    start_driver_node = _parse_bool(start_driver_node_raw)
    start_glove_node = _parse_bool(start_glove_node_raw)
    loop = _parse_bool(loop_raw)
    verbose = _parse_bool(verbose_raw)
    log_publish_full = _parse_bool(log_publish_full_raw)
    publish_idle_on_exit = _parse_bool(publish_idle_on_exit_raw)
    verify_mode = _parse_bool(verify_mode_raw)
    record_glove_q = _parse_bool(record_glove_q_raw)
    shutdown_on_replay_exit = _parse_bool(shutdown_on_replay_exit_raw)

    workspace_root = _find_workspace_root()
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    replay_script = scripts_dir / "right_24_motor_position_replay_from_xlsx.py"
    glove_script = scripts_dir / "minimal_example_dual_stream_commented.py"
    if not replay_script.exists():
        raise FileNotFoundError(f"right_24_motor_position_replay_from_xlsx.py not found: {replay_script}")
    if start_glove_node and (not glove_script.exists()):
        raise FileNotFoundError(f"minimal_example_dual_stream_commented.py not found: {glove_script}")

    defaults_path = (
        _resolve_path(defaults_path_raw, workspace_root)
        if defaults_path_raw
        else (scripts_dir / "muscle_adjustment_defaults.py").resolve()
    )
    if not defaults_path.exists():
        raise FileNotFoundError(f"muscle_adjustment_defaults.py not found: {defaults_path}")

    folder_path = (
        _resolve_path(folder_raw, workspace_root)
        if folder_raw
        else (scripts_dir / "2026.5.18").resolve()
    )
    # In verify mode with explicit verify_motor_feedback_dir, script no longer depends on --folder.
    require_folder_exists = (not verify_mode) or (not verify_motor_feedback_dir_raw)
    if require_folder_exists and (not folder_path.exists()):
        raise FileNotFoundError(f"xlsx folder not found: {folder_path}")

    verify_motor_feedback_dir = ""
    if verify_motor_feedback_dir_raw:
        verify_motor_feedback_dir = str(_resolve_path(verify_motor_feedback_dir_raw, workspace_root))
        if not Path(verify_motor_feedback_dir).exists():
            raise FileNotFoundError(f"verify_motor_feedback_dir not found: {verify_motor_feedback_dir}")

    verify_glove_skeleton_dir = ""
    if verify_glove_skeleton_dir_raw:
        verify_glove_skeleton_dir = str(_resolve_path(verify_glove_skeleton_dir_raw, workspace_root))
        if not Path(verify_glove_skeleton_dir).exists():
            raise FileNotFoundError(f"verify_glove_skeleton_dir not found: {verify_glove_skeleton_dir}")

    verify_output_dir = ""
    if verify_output_dir_raw:
        verify_output_dir = str(_resolve_path(verify_output_dir_raw, workspace_root))

    glove_q_output_dir = ""
    if verify_mode and record_glove_q:
        if glove_q_csv_path_raw:
            resolved = _resolve_path(glove_q_csv_path_raw, workspace_root)
            glove_q_output_dir = str((resolved.parent if resolved.suffix else resolved).resolve())

    cmd_topic = cmd_topic_override if cmd_topic_override else info["command_topic"]
    feedback_topic = feedback_topic_override if feedback_topic_override else info["feedback_topic"]

    actions = [
        LogInfo(
            msg=(
                f"[right_24_motor_position_replay_launch] target_hand={hand}, "
                f"start_driver_node={start_driver_node}, folder={folder_path}, "
                f"start_glove_node={start_glove_node}, "
                f"single_file={single_file if single_file else '<all .xlsx>'}, "
                f"frame_interval_sec={frame_interval_sec}, control_period_sec={control_period_sec}, "
                f"status_period_sec={status_period_sec}, loop={loop}, verbose={verbose}, "
                f"reach_threshold={reach_threshold}, arrival_confirm_frames={arrival_confirm_frames}, "
                f"arrival_timeout_sec={arrival_timeout_sec}, feedback_stale_sec={feedback_stale_sec}, "
                f"unreached_max_log={unreached_max_log}, "
                f"log_publish_full={log_publish_full}, "
                f"publish_idle_on_exit={publish_idle_on_exit}, "
                f"verify_mode={verify_mode}, "
                f"verify_motor_feedback_dir={verify_motor_feedback_dir if verify_motor_feedback_dir else '<auto>'}, "
                f"verify_glove_skeleton_dir={verify_glove_skeleton_dir if verify_glove_skeleton_dir else '<auto>'}, "
                f"verify_output_dir={verify_output_dir if verify_output_dir else '<auto>'}, "
                f"record_glove_q={record_glove_q}, "
                f"glove_topic={glove_topic}, "
                f"glove_q_xlsx_dir={glove_q_output_dir if glove_q_output_dir else '<auto: replay_output/glove_data>'}, "
                f"glove_raw_skeleton_topic={glove_raw_skeleton_topic}, "
                f"shutdown_on_replay_exit={shutdown_on_replay_exit}, "
                f"exclude_motor_ids={exclude_motor_ids if exclude_motor_ids else '<hand wrist default>'}, "
                f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}"
            )
        )
    ]

    if start_driver_node:
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
        )
        actions.append(driver_process)

    if start_glove_node:
        glove_process = ExecuteProcess(
            cmd=["python3", "-u", str(glove_script)],
            name="glove_publisher",
            output="screen",
        )
        actions.append(glove_process)

    replay_args = [
        "--source-motor-map",
        LaunchConfiguration("source_motor_map").perform(context),
        "--hand",
        hand,
        "--folder",
        str(folder_path),
        "--defaults-path",
        str(defaults_path),
        "--cmd-topic",
        cmd_topic,
        "--feedback-topic",
        feedback_topic,
        "--glove-raw-skeleton-topic",
        glove_raw_skeleton_topic,
        "--frame-interval-sec",
        frame_interval_sec,
        "--control-period-sec",
        control_period_sec,
        "--status-period-sec",
        status_period_sec,
        "--reach-threshold",
        reach_threshold,
        "--arrival-confirm-frames",
        arrival_confirm_frames,
        "--arrival-timeout-sec",
        arrival_timeout_sec,
        "--feedback-stale-sec",
        feedback_stale_sec,
        "--unreached-max-log",
        unreached_max_log,
    ]
    if single_file:
        replay_args.extend(["--single-file", single_file])
    if exclude_motor_ids:
        replay_args.extend(["--exclude-motor-ids", exclude_motor_ids])
    if loop:
        replay_args.append("--loop")
    if verbose:
        replay_args.append("--verbose")
    if log_publish_full:
        replay_args.append("--log-publish-full")
    if publish_idle_on_exit:
        replay_args.append("--publish-idle-on-exit")
    if verify_mode:
        replay_args.append("--verify-mode")
        replay_args.extend(["--ready-timeout-sec", ready_timeout_sec])
        replay_args.extend(["--verify-glove-sample-count", verify_glove_sample_count])
        replay_args.extend(["--verify-glove-sample-timeout-sec", verify_glove_sample_timeout_sec])
        replay_args.extend(["--verify-post-reach-wait-sec", verify_post_reach_wait_sec])
        if verify_motor_feedback_dir:
            replay_args.extend(["--verify-motor-feedback-dir", verify_motor_feedback_dir])
        if verify_glove_skeleton_dir:
            replay_args.extend(["--verify-glove-skeleton-dir", verify_glove_skeleton_dir])
        if verify_output_dir:
            replay_args.extend(["--verify-output-dir", verify_output_dir])

    replay_cmd = ["python3", "-u", str(replay_script), *replay_args]
    if verify_mode and record_glove_q:
        replay_cmd = [
            "python3",
            "-u",
            "-c",
            REPLAY_VERIFY_GLOVE_Q_XLSX_WRAPPER_TEMPLATE,
            str(replay_script),
            glove_topic,
            glove_q_output_dir,
            *replay_args,
        ]

    replay_process_kwargs = {
        "cmd": replay_cmd,
        "name": f"{hand}_24_motor_position_replay",
        "output": "screen",
    }
    if shutdown_on_replay_exit:
        replay_process_kwargs["on_exit"] = Shutdown(reason="launch is shutting down")
    replay_process = ExecuteProcess(**replay_process_kwargs)

    if start_driver_node:
        actions.append(TimerAction(period=float(start_delay_sec), actions=[replay_process]))
    else:
        actions.append(replay_process)
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "source_motor_map", default_value="current", choices=["current", "legacy"],
                description="Motor numbering used in the source recording.",
            ),
            DeclareLaunchArgument(
                "target_hand",
                default_value="right",
                description="Target hand for 24-motor xlsx replay: left/right",
            ),
            DeclareLaunchArgument(
                "start_driver_node",
                default_value="true",
                description="Whether to start Y_Hand_Node in this launch: true/false",
            ),
            DeclareLaunchArgument(
                "start_glove_node",
                default_value="true",
                description="Whether to start minimal_example_dual_stream_commented.py glove publisher: true/false",
            ),
            DeclareLaunchArgument(
                "folder",
                default_value="src/yhand_mapping/scripts/2026.5.18",
                description="Folder containing xlsx frames (relative to workspace root or absolute path).",
            ),
            DeclareLaunchArgument(
                "single_file",
                default_value="",
                description="Replay only one xlsx file (e.g. 04978.xlsx). Empty means replay all files.",
            ),
            DeclareLaunchArgument(
                "frame_interval_sec",
                default_value="0.2",
                description="Deprecated in single-file hold mode.",
            ),
            DeclareLaunchArgument(
                "control_period_sec",
                default_value="0.02",
                description="Command publish period while holding target.",
            ),
            DeclareLaunchArgument(
                "status_period_sec",
                default_value="0.5",
                description="Reached-check and status log period.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="false",
                description="Deprecated in single-file hold mode.",
            ),
            DeclareLaunchArgument(
                "verbose",
                default_value="true",
                description="Print holding status details: true/false",
            ),
            DeclareLaunchArgument(
                "exclude_motor_ids",
                default_value="",
                description="Comma separated motor IDs to exclude. Empty uses hand wrist default.",
            ),
            DeclareLaunchArgument(
                "cmd_topic",
                default_value="",
                description="Override command topic. Empty uses hand default.",
            ),
            DeclareLaunchArgument(
                "feedback_topic",
                default_value="",
                description="Override feedback topic. Empty uses hand default.",
            ),
            DeclareLaunchArgument(
                "glove_topic",
                default_value="/glove_joint_data",
                description="ROS topic for 48D glove joint-angle(q) data.",
            ),
            DeclareLaunchArgument(
                "glove_raw_skeleton_topic",
                default_value="/glove_joint_data_raw_skeleton",
                description="Raw glove skeleton topic used in verify mode.",
            ),
            DeclareLaunchArgument(
                "record_glove_q",
                default_value="true",
                description="In verify mode, record /glove_joint_data q stream to per-point xlsx: true/false",
            ),
            DeclareLaunchArgument(
                "glove_q_csv_path",
                default_value="",
                description=(
                    "Output directory for recorded glove q xlsx files (compat arg name). "
                    "If a file path is given, its parent directory is used. "
                    "Empty uses replay output sibling folder: glove_data."
                ),
            ),
            DeclareLaunchArgument(
                "defaults_path",
                default_value="",
                description="Path to muscle_adjustment_defaults.py. Empty uses script default.",
            ),
            DeclareLaunchArgument(
                "reach_threshold",
                default_value="7.0",
                description="Absolute position error threshold for unreached check.",
            ),
            DeclareLaunchArgument(
                "arrival_confirm_frames",
                default_value="30",
                description="Consecutive reached frames required.",
            ),
            DeclareLaunchArgument(
                "arrival_timeout_sec",
                default_value="6.0",
                description="Arrival timeout in seconds.",
            ),
            DeclareLaunchArgument(
                "feedback_stale_sec",
                default_value="1.0",
                description="Feedback stale threshold in seconds.",
            ),
            DeclareLaunchArgument(
                "unreached_max_log",
                default_value="24",
                description="Maximum unreached motors printed in one line.",
            ),
            DeclareLaunchArgument(
                "log_publish_full",
                default_value="true",
                description="Log all controlled published positions: true/false",
            ),
            DeclareLaunchArgument(
                "publish_idle_on_exit",
                default_value="false",
                description="Publish idle command before replay exits: true/false",
            ),
            DeclareLaunchArgument(
                "verify_mode",
                default_value="false",
                description="Enable verify mode (motor_feedback replay + glove skeleton comparison): true/false",
            ),
            DeclareLaunchArgument(
                "verify_motor_feedback_dir",
                default_value="",
                description=(
                    "Directory containing motor_feedback xlsx/csv files. "
                    "Empty uses script auto-resolve based on --folder."
                ),
            ),
            DeclareLaunchArgument(
                "verify_glove_skeleton_dir",
                default_value="",
                description=(
                    "Directory containing reference Glove Skeleton Data xlsx/csv files. "
                    "Empty uses script auto-resolve."
                ),
            ),
            DeclareLaunchArgument(
                "verify_output_dir",
                default_value="",
                description="Output directory for verify summaries. Empty uses script auto timestamp directory.",
            ),
            DeclareLaunchArgument(
                "ready_timeout_sec",
                default_value="20.0",
                description="Verify mode startup wait timeout for feedback and glove raw readiness.",
            ),
            DeclareLaunchArgument(
                "verify_glove_sample_count",
                default_value="5",
                description="Verify mode glove_raw sample count per point for averaging.",
            ),
            DeclareLaunchArgument(
                "verify_glove_sample_timeout_sec",
                default_value="1.5",
                description="Verify mode glove_raw sample wait timeout per point.",
            ),
            DeclareLaunchArgument(
                "verify_post_reach_wait_sec",
                default_value="0.2",
                description="Verify mode extra wait after reached and before glove sampling.",
            ),
            DeclareLaunchArgument(
                "shutdown_on_replay_exit",
                default_value="false",
                description="Shutdown whole launch when replay process exits: true/false",
            ),
            DeclareLaunchArgument(
                "control_script_start_delay_sec",
                default_value=str(CONTROL_SCRIPT_START_DELAY_SEC),
                description="Delay before replay script starts when driver node is enabled.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
