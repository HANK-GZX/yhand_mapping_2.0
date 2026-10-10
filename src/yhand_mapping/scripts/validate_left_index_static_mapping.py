#!/usr/bin/env python3
"""Repeat ten left-index tightening sweeps from zero to -200."""

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy

import data_collection_main as dc
from data_collection_config import (
    ARRIVAL_CONFIRM_SEC,
    MOVE_TIMEOUT_SEC,
    POINT_ARRIVAL_TOL,
    PROCESS_SAMPLE_HZ,
    RESULT_WINDOW_END_SEC,
    RESULT_WINDOW_START_SEC,
)
from workspace_boundary_config import HAND_MOTOR_MAP, get_hand_topics


HAND = "left"
FINGER = "index"
HAND_Q_SLICE = slice(8, 28)
FINGER_Q_SLICE = slice(4, 8)
Q_DIM = 4

INDEX_MOTOR_NAME_TO_ID = {
    name: int(motor_id)
    for name, motor_id in HAND_MOTOR_MAP[HAND][FINGER].items()
}
ACTIVE_MOTOR_ID = int(INDEX_MOTOR_NAME_TO_ID["flex_deep"])
INDEX_MOTOR_IDS = sorted(set(INDEX_MOTOR_NAME_TO_ID.values()))


class CsvStream:
    def __init__(self, path: Path, fieldnames: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def _parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Repeat left-index motor-3 sweeps from zero to -200."
    )
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--glove-topic", default="/glove_joint_data")
    parser.add_argument("--glove-raw-skeleton-topic", default="/glove_joint_data_raw_skeleton")
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "motor_test_logs" / "static_mapping_validation"),
    )
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--first-target", type=float, default=-10.0)
    parser.add_argument("--last-target", type=float, default=-200.0)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--min-result-samples", type=int, default=20)
    parser.add_argument(
        "--include-timeouts-in-summary",
        action="store_true",
        help="Include finite timeout trials in repeatability summaries.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the RUN confirmation.")
    return parser.parse_args()


def _target_positions(first: float, last: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("--positions must be at least 2")
    if first >= 0.0 or last >= 0.0 or last >= first:
        raise ValueError("Expected tightening targets with 0 > first-target > last-target")
    if last < -200.0 - 1e-9:
        raise ValueError("Refusing a target below the requested -200 endpoint")
    return [float(round(v, 9)) for v in np.linspace(first, last, count)]


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values, sample_std: bool = False):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
        }
    ddof = 1 if sample_std and arr.size > 1 else 0
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
    }


def _extract_samples(outcome: dict, trial_meta: dict) -> list[dict]:
    glove_rows = list(outcome.get("result_glove_rows", []))
    feedback_rows = list(outcome.get("result_fb_rows", []))
    rows = []
    for sample_index in range(min(len(glove_rows), len(feedback_rows))):
        glove = np.asarray(glove_rows[sample_index], dtype=np.float64)
        feedback = list(feedback_rows[sample_index])
        if glove.size < 48 or len(feedback) < 1 + 28 * 5:
            continue

        finger_q = glove[HAND_Q_SLICE][FINGER_Q_SLICE]
        row = {
            **trial_meta,
            "sample_index": int(sample_index),
            "t_result_sec": float(feedback[0]),
        }
        for qi in range(Q_DIM):
            row[f"q_{qi}"] = float(finger_q[qi])
        for motor_id in INDEX_MOTOR_IDS:
            base = 1 + (motor_id - 1) * 5
            row[f"m{motor_id}_pos"] = float(feedback[base + 0])
            row[f"m{motor_id}_vel"] = float(feedback[base + 1])
            row[f"m{motor_id}_tor"] = float(feedback[base + 2])
        rows.append(row)
    return rows


def _build_trial_row(
    outcome: dict,
    samples: list[dict],
    trial_meta: dict,
    zero_result: dict,
    min_result_samples: int,
    include_timeouts: bool,
) -> dict:
    row = {
        **trial_meta,
        "reached": int(bool(outcome.get("reached", False))),
        "timeout": int(bool(outcome.get("timeout", False))),
        "move_elapsed_sec": float(outcome.get("move_elapsed_sec", float("nan"))),
        "arrival_max_error": float(outcome.get("max_error", float("nan"))),
        "result_sample_count": len(samples),
        "zero_success": int(bool(zero_result.get("success", False))),
        "zero_attempts": int(zero_result.get("attempts", 0)),
    }

    all_finite = bool(samples)
    for qi in range(Q_DIM):
        values = _stats(sample[f"q_{qi}"] for sample in samples)
        for name in ("mean", "std", "min", "max", "range"):
            row[f"q_{qi}_{name}"] = values[name]
        all_finite = all_finite and values["count"] == len(samples)

    max_abs_vel = 0.0
    for motor_id in INDEX_MOTOR_IDS:
        pos = _stats(sample[f"m{motor_id}_pos"] for sample in samples)
        velocities = [
            abs(float(sample[f"m{motor_id}_vel"]))
            for sample in samples
            if math.isfinite(float(sample[f"m{motor_id}_vel"]))
        ]
        for name in ("mean", "std", "min", "max", "range"):
            row[f"m{motor_id}_pos_{name}"] = pos[name]
        row[f"m{motor_id}_max_abs_vel"] = max(velocities, default=float("nan"))
        all_finite = all_finite and pos["count"] == len(samples)
        all_finite = all_finite and len(velocities) == len(samples)
        if velocities:
            max_abs_vel = max(max_abs_vel, max(velocities))

    row["max_abs_index_motor_vel"] = max_abs_vel if samples else float("nan")
    row["finite_pairs"] = int(all_finite)
    reached_ok = bool(outcome.get("reached", False)) or bool(include_timeouts)
    row["usable_for_summary"] = int(
        bool(zero_result.get("success", False))
        and all_finite
        and len(samples) >= int(min_result_samples)
        and reached_ok
    )
    return row


def _raw_headers() -> list[str]:
    headers = [
        "trial_id",
        "sweep",
        "position_index",
        "target_pos",
        "sample_index",
        "t_result_sec",
    ]
    headers.extend(f"q_{qi}" for qi in range(Q_DIM))
    for motor_id in INDEX_MOTOR_IDS:
        headers.extend(
            [f"m{motor_id}_pos", f"m{motor_id}_vel", f"m{motor_id}_tor"]
        )
    return headers


def _trial_headers() -> list[str]:
    headers = [
        "trial_id",
        "sweep",
        "position_index",
        "target_pos",
        "reached",
        "timeout",
        "move_elapsed_sec",
        "arrival_max_error",
        "result_sample_count",
        "zero_success",
        "zero_attempts",
        "finite_pairs",
        "usable_for_summary",
        "max_abs_index_motor_vel",
    ]
    for qi in range(Q_DIM):
        headers.extend(f"q_{qi}_{name}" for name in ("mean", "std", "min", "max", "range"))
    for motor_id in INDEX_MOTOR_IDS:
        headers.extend(
            f"m{motor_id}_pos_{name}" for name in ("mean", "std", "min", "max", "range")
        )
        headers.append(f"m{motor_id}_max_abs_vel")
    return headers


def _write_summaries(run_root: Path, trials: list[dict], targets: list[float]):
    repeatability = []
    for position_index, target in enumerate(targets, start=1):
        group = [
            row
            for row in trials
            if int(row["position_index"]) == position_index
            and int(row["usable_for_summary"]) == 1
        ]
        output = {
            "position_index": position_index,
            "target_pos": target,
            "usable_sweeps": len(group),
        }
        q_stds = []
        for qi in range(Q_DIM):
            values = _stats((row[f"q_{qi}_mean"] for row in group), sample_std=True)
            output[f"q_{qi}_mean_across_sweeps"] = values["mean"]
            output[f"q_{qi}_std_across_sweeps"] = values["std"]
            output[f"q_{qi}_range_across_sweeps"] = values["range"]
            if math.isfinite(values["std"]):
                q_stds.append(values["std"])
        motor_values = _stats(
            (row[f"m{ACTIVE_MOTOR_ID}_pos_mean"] for row in group),
            sample_std=True,
        )
        output["active_motor_mean_across_sweeps"] = motor_values["mean"]
        output["active_motor_std_across_sweeps"] = motor_values["std"]
        output["max_q_std_across_sweeps"] = max(q_stds, default=float("nan"))
        repeatability.append(output)

    repeat_headers = ["position_index", "target_pos", "usable_sweeps"]
    for qi in range(Q_DIM):
        repeat_headers.extend(
            [
                f"q_{qi}_mean_across_sweeps",
                f"q_{qi}_std_across_sweeps",
                f"q_{qi}_range_across_sweeps",
            ]
        )
    repeat_headers.extend(
        [
            "active_motor_mean_across_sweeps",
            "active_motor_std_across_sweeps",
            "max_q_std_across_sweeps",
        ]
    )
    _write_csv(run_root / "repeatability_summary.csv", repeatability, repeat_headers)

    stability = []
    for trial in trials:
        output = {
            key: trial[key]
            for key in (
                "trial_id",
                "sweep",
                "position_index",
                "target_pos",
                "reached",
                "timeout",
                "result_sample_count",
                "usable_for_summary",
                "max_abs_index_motor_vel",
            )
        }
        q_ranges = []
        for qi in range(Q_DIM):
            value = float(trial[f"q_{qi}_range"])
            output[f"q_{qi}_range"] = value
            if math.isfinite(value):
                q_ranges.append(value)
        motor_ranges = []
        for motor_id in INDEX_MOTOR_IDS:
            value = float(trial[f"m{motor_id}_pos_range"])
            output[f"m{motor_id}_pos_range"] = value
            if math.isfinite(value):
                motor_ranges.append(value)
        output["max_q_range"] = max(q_ranges, default=float("nan"))
        output["max_index_motor_pos_range"] = max(motor_ranges, default=float("nan"))
        stability.append(output)

    stability_headers = [
        "trial_id",
        "sweep",
        "position_index",
        "target_pos",
        "reached",
        "timeout",
        "result_sample_count",
        "usable_for_summary",
        "max_abs_index_motor_vel",
    ]
    stability_headers.extend(f"q_{qi}_range" for qi in range(Q_DIM))
    stability_headers.extend(f"m{mid}_pos_range" for mid in INDEX_MOTOR_IDS)
    stability_headers.extend(["max_q_range", "max_index_motor_pos_range"])
    _write_csv(run_root / "stability_summary.csv", stability, stability_headers)

    repeat_std_values = [
        float(row["max_q_std_across_sweeps"])
        for row in repeatability
        if math.isfinite(float(row["max_q_std_across_sweeps"]))
    ]
    q_range_values = [
        float(row["max_q_range"])
        for row in stability
        if math.isfinite(float(row["max_q_range"]))
    ]
    motor_range_values = [
        float(row["max_index_motor_pos_range"])
        for row in stability
        if math.isfinite(float(row["max_index_motor_pos_range"]))
    ]
    summary = {
        "updated_at": datetime.now().isoformat(),
        "recorded_trials": len(trials),
        "usable_trials": sum(int(row["usable_for_summary"]) for row in trials),
        "reached_trials": sum(int(row["reached"]) for row in trials),
        "timeout_trials": sum(int(row["timeout"]) for row in trials),
        "worst_q_std_across_sweeps": max(repeat_std_values, default=None),
        "worst_result_window_q_range": max(q_range_values, default=None),
        "worst_result_window_motor_pos_range": max(motor_range_values, default=None),
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def _confirm(args, targets: list[float], run_root: Path):
    print("\n[static-validation] left index flex_deep sweep")
    print(f"  active motor: m{ACTIVE_MOTOR_ID}")
    print(f"  targets per sweep: {targets}")
    print(f"  sweeps: {args.sweeps}")
    print(f"  total recorded positions: {len(targets) * args.sweeps}")
    print(f"  output: {run_root}")
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive confirmation unavailable; pass --yes explicitly")
    if input("Type RUN to start motion: ").strip() != "RUN":
        raise RuntimeError("Experiment cancelled")


def main():
    args = _parse_args()
    targets = _target_positions(args.first_target, args.last_target, args.positions)
    if args.sweeps <= 0:
        raise ValueError("--sweeps must be positive")

    output_root = Path(args.output_dir).expanduser().resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"{run_stamp}_left_index_deep_sweeps"
    _confirm(args, targets, run_root)
    run_root.mkdir(parents=True, exist_ok=False)

    cmd_topic, feedback_topic = get_hand_topics(HAND)
    cmd_topic = str(args.cmd_topic or cmd_topic)
    feedback_topic = str(args.feedback_topic or feedback_topic)
    defaults_path = Path(args.defaults_path).expanduser().resolve()
    defaults_hand = dc._load_defaults_from_py(defaults_path, HAND)
    finger_motor_set = dc._finger_motor_sets(HAND)
    combo_stub = SimpleNamespace(combo_id="left_index_deep_sweep")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "status": "starting",
        "hand": HAND,
        "finger": FINGER,
        "active_motor_id": ACTIVE_MOTOR_ID,
        "active_motor_name": "flex_deep",
        "index_motor_name_to_id": INDEX_MOTOR_NAME_TO_ID,
        "targets": targets,
        "sweeps": int(args.sweeps),
        "cmd_topic": cmd_topic,
        "feedback_topic": feedback_topic,
        "glove_topic": str(args.glove_topic),
        "defaults_path": str(defaults_path),
        "move_timeout_sec": float(MOVE_TIMEOUT_SEC),
        "arrival_confirm_sec": float(ARRIVAL_CONFIRM_SEC),
        "point_arrival_tol": float(POINT_ARRIVAL_TOL),
        "result_window_sec": [float(RESULT_WINDOW_START_SEC), float(RESULT_WINDOW_END_SEC)],
        "sample_hz": float(PROCESS_SAMPLE_HZ),
    }
    _write_json(run_root / "metadata.json", metadata)

    raw_stream = CsvStream(run_root / "result_window_samples.csv", _raw_headers())
    trial_stream = CsvStream(run_root / "trials.csv", _trial_headers())
    trials = []
    interrupted = False
    trial_id = 0

    rclpy.init()
    node = dc.DataCollectionNode(
        hand=HAND,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        glove_topic=str(args.glove_topic),
        glove_raw_skeleton_topic=str(args.glove_raw_skeleton_topic),
        defaults_hand=defaults_hand,
    )
    spin_thread = threading.Thread(target=dc._spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        ready_ok, ready_msg = node.wait_ready(
            timeout_sec=8.0,
            require_feedback=True,
            require_glove=True,
            require_glove_raw=False,
        )
        if not ready_ok:
            raise RuntimeError(f"Startup not ready: {ready_msg}")

        metadata["status"] = "running"
        metadata["started_at"] = datetime.now().isoformat()
        _write_json(run_root / "metadata.json", metadata)

        for sweep in range(1, int(args.sweeps) + 1):
            print(f"\n[static-validation] sweep {sweep}/{args.sweeps}: zero -> -200")
            zero_result = dc._run_zero_calibration(node, defaults_hand)
            if not bool(zero_result.get("success", False)):
                raise RuntimeError(f"Zero calibration failed before sweep {sweep}: {zero_result}")

            for position_index, target in enumerate(targets, start=1):
                trial_id += 1
                trial_meta = {
                    "trial_id": trial_id,
                    "sweep": sweep,
                    "position_index": position_index,
                    "target_pos": target,
                }
                print(
                    f"[static-validation] sweep={sweep}/{args.sweeps} "
                    f"position={position_index}/{len(targets)} m{ACTIVE_MOTOR_ID}={target:.3f}"
                )
                outcome = dc._execute_point(
                    node=node,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger=FINGER,
                    combo=combo_stub,
                    target_by_motor={ACTIVE_MOTOR_ID: target},
                    active_motors=[ACTIVE_MOTOR_ID],
                    active_mode="x_only",
                    point_tolerance=float(POINT_ARRIVAL_TOL),
                    point_id=trial_id,
                    nominal_total=len(targets) * int(args.sweeps),
                )
                samples = _extract_samples(outcome, trial_meta)
                for sample in samples:
                    raw_stream.write(sample)
                trial = _build_trial_row(
                    outcome=outcome,
                    samples=samples,
                    trial_meta=trial_meta,
                    zero_result=zero_result,
                    min_result_samples=int(args.min_result_samples),
                    include_timeouts=bool(args.include_timeouts_in_summary),
                )
                trial_stream.write(trial)
                trials.append(trial)
                q_ranges = [round(float(trial[f"q_{qi}_range"]), 6) for qi in range(Q_DIM)]
                print(
                    f"[static-validation] reached={trial['reached']} timeout={trial['timeout']} "
                    f"samples={trial['result_sample_count']} q_range={q_ranges} "
                    f"m{ACTIVE_MOTOR_ID}_range={float(trial[f'm{ACTIVE_MOTOR_ID}_pos_range']):.6f}"
                )

        metadata["status"] = "finished"
    except KeyboardInterrupt:
        interrupted = True
        metadata["status"] = "interrupted"
        print("\n[static-validation] interrupted; preserving completed records")
    finally:
        summary = _write_summaries(run_root, trials, targets)
        metadata["finished_at"] = datetime.now().isoformat()
        metadata["interrupted"] = interrupted
        metadata["summary"] = summary
        _write_json(run_root / "metadata.json", metadata)
        raw_stream.close()
        trial_stream.close()

        try:
            node.set_command_matrix(dc._build_idle_tor_matrix(defaults_hand))
            time.sleep(0.2)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

    print(f"\n[static-validation] output={run_root}")
    print(f"[static-validation] summary={run_root / 'summary.json'}")


if __name__ == "__main__":
    if dc.DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source the ROS2 workspace before running."
        ) from dc.DEVICES_MSG_IMPORT_ERROR
    main()
