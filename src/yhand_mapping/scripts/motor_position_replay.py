#!/usr/bin/env python3
import argparse
import copy
import select
import sys
import threading
import time
from pathlib import Path

import rclpy

import motor_position_test as mpt

FAST_ZERO_SWAY_RELEASE_SEC = 0.0
FAST_ZERO_FLEXOR_RELEASE_SEC = 0.2
FAST_ZERO_ARRIVAL_CONFIRM_FRAMES = 12
FAST_ZERO_TIMEOUT_SEC = 4.0


def _parse_args():
    parser = argparse.ArgumentParser(description="Replay recorded motor snapshots by index.")
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument("--defaults-path", default="")
    parser.add_argument("--snapshot-path", default="")
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Replay snapshot index (1-based). Default 0 means prompt in terminal.",
    )
    parser.add_argument(
        "--auto-loop-mode",
        default="",
        help="Auto-start loop mode without prompt: 1/mode_1 or 2/mode_2.",
    )
    parser.add_argument(
        "--auto-loop-interval",
        type=float,
        default=1.0,
        help="Step interval seconds for auto loop mode. Default: 1.0",
    )
    return parser.parse_args()


def _resolve_snapshot_index(snapshot_history: list[mpt.MotorSnapshot], index_arg: int):
    total = len(snapshot_history)
    if total <= 0:
        print("暂无可回放快照：请先在 motor_position_test.py 中按 h 记录快照。")
        return None

    if int(index_arg) > 0:
        if 1 <= int(index_arg) <= total:
            return int(index_arg)
        print(f"命令行索引超范围: {index_arg}，有效范围是 1..{total}")
        return None

    return mpt._prompt_replay_index(snapshot_history)


def _prompt_replay_index_or_zero(snapshot_history: list[mpt.MotorSnapshot]):
    if not snapshot_history:
        print("暂无可回放快照：请先在 motor_position_test.py 中按 h 记录快照。")
        return "cancel", None
    mpt._print_snapshot_catalog(snapshot_history)
    total = len(snapshot_history)
    while True:
        raw = input("输入回放索引号（回车=最新，z=全部回0，o=循环播放，c=取消）: ").strip().lower()
        if raw in ("o", "auto", "loop"):
            return "auto_loop", None
        if raw in ("z", "zero"):
            return "zero", None
        if raw in ("c", "cancel", "q", "quit", "x"):
            return "cancel", None
        if raw == "":
            return "replay", total
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入整数索引号，或回车/z/c。")
            continue
        if 1 <= idx <= total:
            return "replay", idx
        print(f"索引超范围，请输入 1..{total}。")


def _prompt_auto_loop_mode():
    while True:
        raw = input(
            "选择 o 循环模式（1=对指循环，2=对指回0循环，c=取消）: "
        ).strip().lower()
        if raw == "1":
            return "mode_1"
        if raw == "2":
            return "mode_2"
        if raw in ("c", "cancel", "q", "quit", "x"):
            return None
        print("输入无效，请输入 1 / 2 / c。")


def _run_auto_loop(
    node: mpt.MotorPositionTestNode,
    snapshot_history: list[mpt.MotorSnapshot],
    mode: str,
    interval_sec: float = 0.5,
):
    required_indices = [1, 2, 3, 4]
    total = len(snapshot_history)
    missing_indices = [idx for idx in required_indices if idx > total]
    if missing_indices:
        print(
            f"无法启动 o 自动循环：当前快照只有 {total} 条，"
            f"缺少索引 {missing_indices}。"
        )
        return

    if mode == "mode_1":
        mode_label = "对指循环"
        sequence = ["1", "2", "3", "4", "z"]
    elif mode == "mode_2":
        mode_label = "对指回0循环"
        sequence = ["1", "z", "2", "z", "3", "z", "4", "z"]
    else:
        print(f"未知循环模式: {mode}")
        return

    def _apply_auto_loop_command(cmd: str):
        if cmd == "zero":
            print("检测到 z 指令：打断 o 循环并执行回0。")
            zero_result = _execute_all_zero_fast(node)
            print(f"回0结果: {zero_result}")
            return True
        if cmd == "cancel":
            print("检测到 c 指令：停止 o 循环并返回菜单。")
            return True
        return False

    def _poll_auto_loop_command(timeout_sec: float = 0.0):
        wait_sec = max(0.0, float(timeout_sec))
        try:
            readable, _, _ = select.select([sys.stdin], [], [], wait_sec)
        except Exception:
            if wait_sec > 0.0:
                time.sleep(wait_sec)
            return None
        if not readable:
            return None
        raw = sys.stdin.readline()
        if raw is None:
            return None
        cmd = raw.strip().lower()
        if not cmd:
            return None
        if cmd in ("z", "zero"):
            return "zero"
        if cmd in ("c", "cancel", "q", "quit", "x"):
            return "cancel"
        print(f"运行中忽略输入: {cmd}（仅支持 z/c）")
        return None

    def _wait_interval_with_interrupt(wait_sec: float):
        remaining = max(0.0, float(wait_sec))
        while remaining > 0.0:
            slice_sec = min(0.05, remaining)
            cmd = _poll_auto_loop_command(timeout_sec=slice_sec)
            if cmd is not None:
                return cmd
            remaining -= slice_sec
        return None

    print(
        f"已启动 o 自动循环（模式 {mode_label}），"
        f"每步间隔 {float(interval_sec):.1f}s。"
        "运行中输入 z+回车 可打断并回0，输入 c+回车 可停止循环返回菜单。"
    )
    round_idx = 0
    while True:
        pending_cmd = _poll_auto_loop_command(timeout_sec=0.0)
        if pending_cmd is not None and _apply_auto_loop_command(pending_cmd):
            return

        round_idx += 1
        print(f"自动循环轮次 #{round_idx}")
        for step in sequence:
            pending_cmd = _poll_auto_loop_command(timeout_sec=0.0)
            if pending_cmd is not None and _apply_auto_loop_command(pending_cmd):
                return

            if step == "z":
                print("自动动作: z (全部回0)")
                zero_result = _execute_all_zero_fast(node)
                print(f"回0结果: {zero_result}")
            else:
                replay_idx = int(step)
                print(f"自动动作: 回放索引 #{replay_idx}")
                replay_result = _execute_replay_snapshot_fast(
                    node,
                    snapshot_history[replay_idx - 1],
                    snapshot_index=replay_idx,
                )
                print(
                    f"回放结果: {replay_result.get('result')}, "
                    f"elapsed={float(replay_result.get('elapsed_sec', 0.0)):.2f}s, "
                    f"failed_ids={replay_result.get('failed_ids', [])}"
                )
            pending_cmd = _wait_interval_with_interrupt(float(interval_sec))
            if pending_cmd is not None and _apply_auto_loop_command(pending_cmd):
                return


def _execute_all_zero_fast(node: mpt.MotorPositionTestNode):
    original_values = (
        float(mpt.ZERO_SWAY_RELEASE_SEC),
        float(mpt.ZERO_FLEXOR_RELEASE_SEC),
        int(mpt.ARRIVAL_CONFIRM_FRAMES),
        float(mpt.RETURN_ZERO_TIMEOUT_SEC),
    )
    try:
        mpt.ZERO_SWAY_RELEASE_SEC = float(FAST_ZERO_SWAY_RELEASE_SEC)
        mpt.ZERO_FLEXOR_RELEASE_SEC = float(FAST_ZERO_FLEXOR_RELEASE_SEC)
        mpt.ARRIVAL_CONFIRM_FRAMES = int(FAST_ZERO_ARRIVAL_CONFIRM_FRAMES)
        mpt.RETURN_ZERO_TIMEOUT_SEC = float(FAST_ZERO_TIMEOUT_SEC)
        return mpt._execute_all_zero(node)
    finally:
        (
            mpt.ZERO_SWAY_RELEASE_SEC,
            mpt.ZERO_FLEXOR_RELEASE_SEC,
            mpt.ARRIVAL_CONFIRM_FRAMES,
            mpt.RETURN_ZERO_TIMEOUT_SEC,
        ) = original_values


def _execute_replay_snapshot_fast(
    node: mpt.MotorPositionTestNode,
    snapshot: mpt.MotorSnapshot,
    snapshot_index: int | None = None,
):
    snapshot_label = f"#{int(snapshot_index)}" if snapshot_index is not None else "<unknown>"
    snapshot_time = mpt._format_wall_time(snapshot.created_ts)
    target_positions, invalid_ids = mpt._snapshot_target_positions(snapshot)
    if invalid_ids:
        print(f"快照{snapshot_label}不可回放：以下电机pos无效 {invalid_ids}")
        return {
            "result": "snapshot_invalid",
            "elapsed_sec": 0.0,
            "failed_ids": invalid_ids,
        }

    print(
        f"开始同步位控28电机到快照{snapshot_label}"
        f"(time={snapshot_time}, source={snapshot.source})，快速模式：不做到位/超时等待..."
    )
    node.set_all_motor_position_targets(target_positions)
    return {
        "result": "command_sent",
        "elapsed_sec": 0.0,
        "failed_ids": [],
    }


def _run_replay_loop(
    node: mpt.MotorPositionTestNode,
    snapshot_history: list[mpt.MotorSnapshot],
    index_arg: int,
    auto_loop_mode: str = "",
    auto_loop_interval: float = 1.0,
):
    mode_text = str(auto_loop_mode).strip().lower()
    if mode_text:
        if mode_text in ("1", "mode_1"):
            mode = "mode_1"
        elif mode_text in ("2", "mode_2"):
            mode = "mode_2"
        else:
            print(f"无效 --auto-loop-mode={auto_loop_mode}，仅支持 1/mode_1 或 2/mode_2")
            return
        _run_auto_loop(node, snapshot_history, mode=mode, interval_sec=float(auto_loop_interval))
        return

    if int(index_arg) > 0:
        replay_idx = _resolve_snapshot_index(snapshot_history, index_arg)
        if replay_idx is None:
            print("未执行回放，退出。")
            return
        replay_result = _execute_replay_snapshot_fast(
            node,
            snapshot_history[replay_idx - 1],
            snapshot_index=replay_idx,
        )
        print(
            f"回放结果: {replay_result.get('result')}, "
            f"elapsed={float(replay_result.get('elapsed_sec', 0.0)):.2f}s, "
            f"failed_ids={replay_result.get('failed_ids', [])}"
        )
        return

    while True:
        action, replay_idx = _prompt_replay_index_or_zero(snapshot_history)
        if action == "cancel":
            print("退出回放模式。")
            return
        if action == "auto_loop":
            auto_mode = _prompt_auto_loop_mode()
            if auto_mode is None:
                print("已取消 o 循环。")
                continue
            _run_auto_loop(node, snapshot_history, mode=auto_mode, interval_sec=1.0)
            continue
        if action == "zero":
            zero_result = _execute_all_zero_fast(node)
            print(f"回0结果: {zero_result}")
            continue
        replay_result = _execute_replay_snapshot_fast(
            node,
            snapshot_history[replay_idx - 1],
            snapshot_index=replay_idx,
        )
        print(
            f"回放结果: {replay_result.get('result')}, "
            f"elapsed={float(replay_result.get('elapsed_sec', 0.0)):.2f}s, "
            f"failed_ids={replay_result.get('failed_ids', [])}"
        )


def main():
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent
    defaults_path = (
        Path(args.defaults_path).expanduser()
        if args.defaults_path
        else (script_dir / "muscle_adjustment_defaults.py")
    )
    snapshot_path = (
        Path(args.snapshot_path).expanduser()
        if args.snapshot_path
        else (script_dir / f"motor_position_snapshots_{args.hand}.json")
    )
    snapshot_history = mpt._load_snapshot_history(snapshot_path)

    defaults = mpt._load_defaults_from_py(defaults_path)
    defaults_hand = defaults[args.hand]
    tor_safety_violations = mpt._validate_tor_pair_safety(args.hand, defaults_hand["tor"])
    runtime_defaults_hand = defaults_hand
    if tor_safety_violations:
        runtime_defaults_hand = copy.deepcopy(defaults_hand)
        runtime_defaults_hand["tor"] = [0.0] * mpt.MOTOR_COUNT
        runtime_defaults_hand["tor_active"] = [0.0] * mpt.MOTOR_COUNT

    cmd_topic, feedback_topic = mpt._topics_for_hand(args.hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic
    if args.feedback_topic:
        feedback_topic = args.feedback_topic

    rclpy.init()
    node = mpt.MotorPositionTestNode(
        hand=args.hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        defaults_hand=runtime_defaults_hand,
        assist_mode=mpt.ASSIST_MODE_OFF,
    )

    if tor_safety_violations:
        node.set_all_zero_output()
    else:
        node.set_idle_tension()

    spin_thread = threading.Thread(target=mpt._spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if tor_safety_violations:
            mpt._print_tor_safety_violations(args.hand, tor_safety_violations)
            mpt._hold_zero_output_until_exit(node)
            return

        ok, msg = node.check_topic_safety(timeout_sec=12.0)
        if not ok:
            print(f"错误: {msg}")
            return

        if not node.wait_feedback(timeout_sec=6.0):
            cmd_pub = node.count_publishers(cmd_topic)
            cmd_sub = node.count_subscribers(cmd_topic)
            fb_pub = node.count_publishers(feedback_topic)
            fb_sub = node.count_subscribers(feedback_topic)
            print(
                "错误: 启动后未收到反馈数据，请检查设备与话题连接。"
                f"\n当前话题状态: {cmd_topic} pubs={cmd_pub}, subs={cmd_sub}; "
                f"{feedback_topic} pubs={fb_pub}, subs={fb_sub}"
            )
            return

        print(
            f"进入快照索引回放模式 hand={args.hand}, cmd_topic={cmd_topic}, feedback_topic={feedback_topic}\n"
            f"快照文件: {snapshot_path} (已加载 {len(snapshot_history)} 条)\n"
            "回放策略: 快速模式（仅下发目标，不做到位判定与超时等待）\n"
            f"自动循环参数: mode={args.auto_loop_mode or '<none>'}, "
            f"interval={float(args.auto_loop_interval):.2f}s"
        )
        _run_replay_loop(
            node,
            snapshot_history,
            args.index,
            auto_loop_mode=args.auto_loop_mode,
            auto_loop_interval=args.auto_loop_interval,
        )
    except KeyboardInterrupt:
        print("收到 Ctrl+C，退出。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if mpt.DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from mpt.DEVICES_MSG_IMPORT_ERROR
    main()
