#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import rclpy

import data_collection_main as dcm
from workspace_boundary_config import get_hand_topics
from hand_motor_map import remap_motor_dict, remap_motor_ids


MOTOR_COL_PATTERNS = (
    re.compile(r"^motor_(\d+)_pos$"),
    re.compile(r"^m(\d+)_pos$"),
)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")


@dataclass
class CsvFrameCommand:
    csv_row_index: int
    frame_idx: int
    target_by_motor: dict[int, float]


@dataclass
class CsvCollection:
    name: str
    csv_path: Path
    frames: list[CsvFrameCommand]
    empty_rows: list[int]
    incomplete_rows: list[int]
    controlled_motor_ids: list[int]


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _detect_motor_columns(fieldnames) -> list[tuple[str, int]]:
    cols = []
    for raw_name in list(fieldnames or []):
        name = str(raw_name or "").strip()
        if not name:
            continue
        motor_id = None
        for pat in MOTOR_COL_PATTERNS:
            m = pat.match(name)
            if m is not None:
                motor_id = int(m.group(1))
                break
        if motor_id is None:
            continue
        if not (1 <= motor_id <= dcm.MOTOR_COUNT):
            continue
        cols.append((name, int(motor_id)))
    return cols


def _parse_optional_int(raw, fallback: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def _load_csv_frames(
    csv_path: Path,
    start_row: int,
    stride: int,
    max_points: int,
) -> tuple[list[CsvFrameCommand], list[int], list[int], list[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    stride = max(1, int(stride))
    start_row = max(0, int(start_row))
    max_points = int(max_points)

    frames: list[CsvFrameCommand] = []
    empty_rows = []
    incomplete_rows = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        motor_cols = _detect_motor_columns(reader.fieldnames)
        if not motor_cols:
            raise RuntimeError(
                "No motor position columns found in CSV. "
                "Expected columns like motor_<id>_pos (or m<id>_pos)."
            )
        controlled_motor_ids = sorted({int(mid) for _, mid in motor_cols})

        for csv_row_idx, row in enumerate(reader):
            if csv_row_idx < start_row:
                continue
            if (csv_row_idx - start_row) % stride != 0:
                continue

            target_by_motor: dict[int, float] = {}
            for col_name, motor_id in motor_cols:
                raw = str(row.get(col_name, "")).strip()
                if raw == "":
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                if not math.isfinite(val):
                    continue
                target_by_motor[int(motor_id)] = float(val)

            if not target_by_motor:
                empty_rows.append(int(csv_row_idx))
                continue

            # CSV combo mode expects full-hand command per point:
            # all controlled motors from header must be present in each row.
            if len(target_by_motor) < len(controlled_motor_ids):
                incomplete_rows.append(int(csv_row_idx))
                continue

            frame_idx = _parse_optional_int(row.get("frame_idx", ""), csv_row_idx)
            frames.append(
                CsvFrameCommand(
                    csv_row_index=int(csv_row_idx),
                    frame_idx=int(frame_idx),
                    target_by_motor=target_by_motor,
                )
            )

            if max_points > 0 and len(frames) >= max_points:
                break

    if not frames:
        raise RuntimeError("No valid command frames parsed from CSV after filtering.")
    return frames, empty_rows, incomplete_rows, controlled_motor_ids


def _build_args():
    p = argparse.ArgumentParser(
        description=(
            "Data collection driven by five_finger_delta_mapping motor-feedback CSV. "
            "Each CSV row is treated as one full-hand position-control point."
        )
    )
    p.add_argument("--hand", default=dcm.DEFAULT_HAND, choices=list(dcm.HAND_CHOICES))
    p.add_argument("--source-motor-map", choices=["current", "legacy"], default="current")
    p.add_argument("--cmd-topic", default="")
    p.add_argument("--feedback-topic", default="")
    p.add_argument("--glove-topic", default=dcm.DEFAULT_GLOVE_TOPIC)
    p.add_argument("--glove-raw-skeleton-topic", default=dcm.DEFAULT_GLOVE_RAW_SKELETON_TOPIC)
    p.add_argument("--defaults-path", default="")
    p.add_argument("--output-dir", default="")

    p.add_argument(
        "--cmd-csv-path",
        default="",
        help="Path to five_finger_delta_mapping motor feedback CSV (frame_idx,...,motor_<id>_pos...).",
    )
    for finger in FINGER_ORDER:
        p.add_argument(
            f"--{finger}-csv-path",
            default="",
            help=f"Optional CSV path for {finger}. Provide any 1-5 per-finger paths.",
        )
    p.add_argument("--start-row", default="0", help="Start data row index in CSV (0-based, excludes header).")
    p.add_argument("--stride", default="1", help="Use every N-th row from CSV.")
    p.add_argument("--max-points", default="0", help="Max points to execute. 0 means all selected rows.")
    p.add_argument("--collection-name", default="csv_full_hand")
    p.add_argument(
        "--zero-between-csv",
        default="true",
        help="Run zero calibration before each provided CSV after the first (true/false).",
    )

    p.add_argument(
        "--point-arrival-tol",
        default="7.0",
        help="Arrival tolerance in degrees for all active motors.",
    )
    p.add_argument(
        "--arrival-confirm-sec",
        default="1.0",
        help=(
            "Continuous duration (sec) that all controlled motors' feedback errors "
            "must stay <= point-arrival-tol to be marked reached."
        ),
    )
    p.add_argument(
        "--post-reach-pretighten-sec",
        default="2.0",
        help=(
            "After reached, hold pretighten(current-only tighten) for this long before "
            "result window starts recording."
        ),
    )
    p.add_argument(
        "--result-window-sec",
        default="1.0",
        help="Result averaging window length in seconds after post-reach pretighten.",
    )
    p.add_argument(
        "--zero-calibrate-every-points",
        default=str(dcm.ZERO_CALIBRATE_EVERY_POINTS),
        help="Periodic zero calibration interval. <=0 disables periodic calibration.",
    )
    p.add_argument(
        "--zero-calibrate-at-start",
        default="false",
        help="Run one zero calibration before first CSV point (true/false).",
    )
    p.add_argument(
        "--use-antagonist-release",
        default="false",
        help=(
            "Enable legacy antagonist release/rewind timeline during point move "
            "(true/false). Default false for pure full-hand position-control."
        ),
    )
    p.add_argument("--disable-glove-required", action="store_true")
    p.add_argument(
        "--disable-glove-raw-required",
        action="store_true",
        help="Only disable raw-skeleton readiness requirement; glove48 is still required.",
    )
    return p.parse_args()


def _resolve_csv_path(raw_path: str) -> Path:
    csv_path = Path(raw_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    return csv_path


def _load_collections(args) -> list[CsvCollection]:
    finger_paths = {
        finger: str(getattr(args, f"{finger}_csv_path", "")).strip()
        for finger in FINGER_ORDER
    }
    provided_fingers = [finger for finger, raw in finger_paths.items() if raw]
    single_csv_raw = str(args.cmd_csv_path).strip()

    if provided_fingers:
        if single_csv_raw:
            raise ValueError(
                "--cmd-csv-path cannot be combined with per-finger CSV path arguments."
            )
        collection_specs = [
            (finger, _resolve_csv_path(finger_paths[finger]))
            for finger in provided_fingers
        ]
    else:
        if not single_csv_raw:
            raise ValueError(
                "Provide --cmd-csv-path for legacy single-CSV mode, or provide 1-5 "
                "--<finger>-csv-path arguments."
            )
        collection_name = str(args.collection_name).strip() or "csv_full_hand"
        collection_specs = [(collection_name, _resolve_csv_path(single_csv_raw))]

    collections = []
    for collection_name, csv_path in collection_specs:
        frames, empty_rows, incomplete_rows, controlled_motor_ids = _load_csv_frames(
            csv_path=csv_path,
            start_row=int(args.start_row),
            stride=int(args.stride),
            max_points=int(args.max_points),
        )
        if args.source_motor_map == "legacy":
            for frame in frames:
                frame.target_by_motor = remap_motor_dict(args.hand, frame.target_by_motor)
            controlled_motor_ids = sorted(remap_motor_ids(args.hand, controlled_motor_ids))
        collections.append(
            CsvCollection(
                name=collection_name,
                csv_path=csv_path,
                frames=frames,
                empty_rows=empty_rows,
                incomplete_rows=incomplete_rows,
                controlled_motor_ids=controlled_motor_ids,
            )
        )
    return collections


def main():
    args = _build_args()

    hand = str(args.hand).strip().lower()
    if hand not in dcm.HAND_CHOICES:
        raise ValueError(f"Unsupported --hand={hand!r}")

    script_dir = Path(__file__).resolve().parent
    defaults_path = Path(args.defaults_path).expanduser() if str(args.defaults_path).strip() else (script_dir / "muscle_adjustment_defaults.py")
    output_root = Path(args.output_dir).expanduser() if str(args.output_dir).strip() else dcm.default_output_root(script_dir)

    cmd_topic, fb_topic = get_hand_topics(hand)
    cmd_topic = str(args.cmd_topic or cmd_topic)
    fb_topic = str(args.feedback_topic or fb_topic)
    glove_topic = str(args.glove_topic or dcm.DEFAULT_GLOVE_TOPIC)
    glove_raw_topic = str(args.glove_raw_skeleton_topic or dcm.DEFAULT_GLOVE_RAW_SKELETON_TOPIC)

    collections = _load_collections(args)
    batch_mode = len(collections) > 1
    all_motor_ids = sorted(
        {
            int(motor_id)
            for collection in collections
            for motor_id in collection.controlled_motor_ids
        }
    )
    point_arrival_tol = float(args.point_arrival_tol)
    arrival_confirm_sec = max(0.0, float(args.arrival_confirm_sec))
    post_reach_pretighten_sec = max(0.0, float(args.post_reach_pretighten_sec))
    result_window_sec = max(0.05, float(args.result_window_sec))
    zero_every = int(args.zero_calibrate_every_points)
    zero_at_start = _parse_bool(args.zero_calibrate_at_start)
    zero_between_csv = _parse_bool(args.zero_between_csv)
    use_antagonist_release = _parse_bool(args.use_antagonist_release)

    defaults_hand = dcm._load_defaults_from_py(defaults_path, hand)

    collection_names = [collection.name for collection in collections]
    run_root, hand_root, dir_map = dcm._prepare_output_dirs(output_root, hand, collection_names)
    zero_events_csv = hand_root / "zero_calibration_events.csv"

    print(f"[csv-data-collection] hand={hand}")
    print(
        f"[csv-data-collection] mode={'multi_csv' if batch_mode else 'single_csv'}, "
        f"collections={collection_names}"
    )
    for collection in collections:
        print(
            f"[csv-data-collection] collection={collection.name}, csv={collection.csv_path}, "
            f"points={len(collection.frames)}, motor_count={len(collection.controlled_motor_ids)}, "
            f"motors={collection.controlled_motor_ids}"
        )
        if collection.empty_rows:
            print(
                f"[csv-data-collection] collection={collection.name}, "
                f"skipped_empty_rows={len(collection.empty_rows)}"
            )
        if collection.incomplete_rows:
            print(
                f"[csv-data-collection] collection={collection.name}, "
                f"skipped_incomplete_rows={len(collection.incomplete_rows)} "
                "(require all controlled motors per row)"
            )
    print(f"[csv-data-collection] output={run_root}")
    if batch_mode:
        print(f"[csv-data-collection] zero_between_csv={'on' if zero_between_csv else 'off'}")
    print(
        "[csv-data-collection] point_move_antagonist_release="
        f"{'on' if use_antagonist_release else 'off'}"
    )
    print(
        f"[csv-data-collection] arrival_criterion: controlled_motors_per_csv, "
        f"error<=tol for continuous {arrival_confirm_sec:.3f}s"
    )
    print(
        "[csv-data-collection] post_reach_recording: pretighten_wait="
        f"{post_reach_pretighten_sec:.3f}s, result_window={result_window_sec:.3f}s"
    )

    rclpy.init()
    node = dcm.DataCollectionNode(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=fb_topic,
        glove_topic=glove_topic,
        glove_raw_skeleton_topic=glove_raw_topic,
        defaults_hand=defaults_hand,
    )
    spin_thread = threading.Thread(target=dcm._spin_node, args=(node,), daemon=True)
    spin_thread.start()

    finger_motor_set = {
        collection.name: list(collection.controlled_motor_ids)
        for collection in collections
    }

    global_point = 0
    global_timeout = 0
    stop_requested = False
    interrupted_by_runtime = False
    zero_calib_state = "ok"
    zero_calib_fail_streak = 0
    zero_calib_total_events = 0
    zero_calib_failed_events = 0
    unreached_by_collection = {collection.name: [] for collection in collections}
    executed_by_collection = {collection.name: 0 for collection in collections}
    timeouts_by_collection = {collection.name: 0 for collection in collections}
    startup_first_frame_files = None
    last_zero_global_point = None
    global_start = time.monotonic()
    original_antagonist_release = bool(dcm.ANTAGONIST_RELEASE_ENABLE)
    original_arrival_confirm_sec = float(dcm.ARRIVAL_CONFIRM_SEC)
    original_result_window_start_sec = float(dcm.RESULT_WINDOW_START_SEC)
    original_result_window_end_sec = float(dcm.RESULT_WINDOW_END_SEC)
    original_stable_hold_sec = float(dcm.STABLE_HOLD_SEC)

    def _run_zero_calibration_and_log(collection_name: str, reason: str):
        nonlocal zero_calib_state
        nonlocal zero_calib_fail_streak
        nonlocal zero_calib_total_events
        nonlocal zero_calib_failed_events
        nonlocal last_zero_global_point

        print(
            f"[csv-data-collection] zero calibration: reason={reason}, "
            f"collection={collection_name}, global_point={global_point}"
        )
        calib_result = dcm._run_zero_calibration(node, defaults_hand)
        success = bool(calib_result.get("success", False))
        attempts = int(calib_result.get("attempts", 0))
        last_max_abs_pos = calib_result.get("last_max_abs_pos", None)
        if success:
            zero_calib_state = "ok"
            zero_calib_fail_streak = 0
        else:
            zero_calib_fail_streak += 1
            zero_calib_state = "degraded"
            zero_calib_failed_events += 1
        zero_calib_total_events += 1
        last_zero_global_point = int(global_point)
        dcm._append_csv_row(
            zero_events_csv,
            dcm._zero_calibration_event_headers(),
            [
                global_point,
                collection_name,
                int(success),
                attempts,
                int(dcm.ZERO_CALIBRATE_MAX_ATTEMPTS),
                last_max_abs_pos,
                zero_calib_state,
                zero_calib_fail_streak,
            ],
        )

    try:
        dcm.ANTAGONIST_RELEASE_ENABLE = bool(use_antagonist_release)
        dcm.ARRIVAL_CONFIRM_SEC = float(arrival_confirm_sec)
        dcm.RESULT_WINDOW_START_SEC = float(post_reach_pretighten_sec)
        dcm.RESULT_WINDOW_END_SEC = float(post_reach_pretighten_sec + result_window_sec)
        dcm.STABLE_HOLD_SEC = max(float(original_stable_hold_sec), float(dcm.RESULT_WINDOW_END_SEC))
        require_glove = not bool(args.disable_glove_required)
        require_glove_raw = require_glove and (not bool(args.disable_glove_raw_required))
        ready_ok, ready_msg = node.wait_ready(
            timeout_sec=dcm.READY_WAIT_TIMEOUT_SEC,
            require_feedback=True,
            require_glove=require_glove,
            require_glove_raw=require_glove_raw,
        )
        if not ready_ok:
            if args.disable_glove_required:
                print(f"[WARN] startup not fully ready: {ready_msg}; continue due to --disable-glove-required")
            else:
                raise RuntimeError(f"Startup not ready: {ready_msg}")

        startup_first_frame_files = dcm._save_startup_first_frame(node=node, hand_root=hand_root)
        if startup_first_frame_files is not None:
            print(
                "[csv-data-collection] startup first frame saved: "
                f"{startup_first_frame_files.get('json', '')}"
            )

        if zero_at_start:
            _run_zero_calibration_and_log(collections[0].name, "startup")

        for collection_index, collection in enumerate(collections):
            collection_name = collection.name
            frames = collection.frames
            motor_union = list(collection.controlled_motor_ids)
            coll_dirs = dir_map[collection_name]
            source_map_csv = coll_dirs["finger_root"] / "point_source_map.csv"
            point_log_csv = coll_dirs["finger_root"] / "point_log.csv"

            if collection_index > 0 and zero_between_csv:
                _run_zero_calibration_and_log(collection_name, "between_csv")

            nominal_total = len(frames)
            print(
                f"[csv-data-collection] collection start: {collection_name} "
                f"({collection_index + 1}/{len(collections)}), points={nominal_total}"
            )
            for point_id, frame in enumerate(frames, start=1):
                if stop_requested:
                    break

                if (
                    zero_every > 0
                    and global_point > 0
                    and (global_point % zero_every == 0)
                    and last_zero_global_point != global_point
                ):
                    _run_zero_calibration_and_log(collection_name, "periodic")

                combo_id = f"csv_frame_{int(frame.frame_idx)}"
                target = {int(mid): float(v) for mid, v in frame.target_by_motor.items()}
                active_motors = list(motor_union)

                outcome = dcm._execute_point(
                    node=node,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger=collection_name,
                    combo=None,
                    target_by_motor=target,
                    active_motors=active_motors,
                    active_mode="csv_24pos",
                    point_tolerance=point_arrival_tol,
                    point_id=point_id,
                    nominal_total=nominal_total,
                    process_include_post_stable=True,
                )

                if bool(outcome.get("interrupted", False)):
                    interrupted_by_runtime = True
                    print(
                        f"[csv-data-collection] point interrupted: collection={collection_name}, "
                        f"id={point_id}, reason={outcome.get('interrupt_reason', '')}"
                    )
                    break

                global_point += 1
                executed_by_collection[collection_name] += 1
                timeout = bool(outcome.get("timeout", False))
                if timeout:
                    global_timeout += 1
                    timeouts_by_collection[collection_name] += 1
                    unreached_by_collection[collection_name].append(point_id)

                point_meta = {
                    "point_id": int(point_id),
                    "global_point_id": int(global_point),
                    "finger": collection_name,
                    "combo_id": combo_id,
                    "csv_row_index": int(frame.csv_row_index),
                    "csv_frame_idx": int(frame.frame_idx),
                    "active_motor_count": len(active_motors),
                    "active_motors": active_motors,
                    "zero_calib_state": zero_calib_state,
                    "zero_calib_fail_streak": int(zero_calib_fail_streak),
                }

                dcm._save_point_files(
                    finger_dirs=coll_dirs,
                    point_id=point_id,
                    point_meta=point_meta,
                    process_glove_rows=outcome["process_glove_rows"],
                    process_glove_skeleton_rows=outcome["process_glove_skeleton_rows"],
                    process_fb_rows=outcome["process_fb_rows"],
                    target_cmd_matrix=outcome["target_cmd_matrix"],
                    result_glove_avg=outcome["result_glove_avg"],
                    result_glove_skeleton_avg=outcome["result_glove_skeleton_avg"],
                    result_fb_avg=outcome["result_fb_avg"],
                    result_window_label=str(outcome["result_window_label"]),
                    result_sample_count=int(outcome["result_sample_count"]),
                )

                dcm._append_csv_row(
                    point_log_csv,
                    dcm._point_log_headers(),
                    [
                        point_id,
                        global_point,
                        collection_name,
                        combo_id,
                        int(bool(outcome["reached"])),
                        int(timeout),
                        float(outcome["move_elapsed_sec"]),
                        float(outcome["max_error"]),
                        str(outcome.get("active_mode", "csv_24pos")),
                        json.dumps([int(mid) for mid in outcome.get("active_motors", active_motors)], ensure_ascii=False),
                        zero_calib_state,
                        zero_calib_fail_streak,
                    ],
                )

                dcm._append_csv_row(
                    source_map_csv,
                    [
                        "point_id",
                        "global_point_id",
                        "csv_row_index",
                        "csv_frame_idx",
                        "active_motor_count",
                        "active_motors",
                    ],
                    [
                        int(point_id),
                        int(global_point),
                        int(frame.csv_row_index),
                        int(frame.frame_idx),
                        int(len(active_motors)),
                        json.dumps(active_motors, ensure_ascii=False),
                    ],
                )

                if (point_id % max(1, int(dcm.STATUS_PRINT_EVERY_POINTS))) == 0:
                    ratio = float(global_timeout) / max(1.0, float(global_point))
                    print(
                        f"[csv-data-collection] collection={collection_name}, "
                        f"progress={point_id}/{nominal_total}, "
                        f"timeouts={global_timeout}, timeout_ratio={ratio:.4f}"
                    )

            print(f"[csv-data-collection] collection finished: {collection_name}")
            if interrupted_by_runtime or stop_requested:
                break

    except KeyboardInterrupt:
        stop_requested = True
        print("\n[csv-data-collection] KeyboardInterrupt received, safe exit...")
    finally:
        total_elapsed = float(time.monotonic() - global_start)
        collection_summaries = []
        for collection in collections:
            coll_dirs = dir_map[collection.name]
            collection_summaries.append(
                {
                    "collection_name": collection.name,
                    "cmd_csv_path": str(collection.csv_path),
                    "csv_total_rows_used": len(collection.frames),
                    "csv_empty_target_rows": [int(v) for v in collection.empty_rows[:2000]],
                    "csv_incomplete_target_rows": [
                        int(v) for v in collection.incomplete_rows[:2000]
                    ],
                    "executed_points": int(executed_by_collection[collection.name]),
                    "timeouts": int(timeouts_by_collection[collection.name]),
                    "motor_union": [int(v) for v in collection.controlled_motor_ids],
                    "unreached_points": [
                        int(v) for v in unreached_by_collection[collection.name]
                    ],
                    "point_source_map_csv": str(
                        coll_dirs["finger_root"] / "point_source_map.csv"
                    ),
                    "point_log_csv": str(coll_dirs["finger_root"] / "point_log.csv"),
                }
            )
        first_collection = collections[0]
        first_dirs = dir_map[first_collection.name]
        summary = {
            "hand": hand,
            "created_at": datetime.now().isoformat(),
            "run_root": str(run_root),
            "batch_mode": bool(batch_mode),
            "collection_name": (
                first_collection.name if not batch_mode else "multi_finger_csv_batch"
            ),
            "collection_names": collection_names,
            "collections": collection_summaries,
            "cmd_csv_path": str(first_collection.csv_path) if not batch_mode else "",
            "csv_total_rows_used": sum(len(c.frames) for c in collections),
            "csv_empty_target_rows": (
                [int(v) for v in first_collection.empty_rows[:2000]]
                if not batch_mode
                else []
            ),
            "csv_incomplete_target_rows": (
                [int(v) for v in first_collection.incomplete_rows[:2000]]
                if not batch_mode
                else []
            ),
            "executed_points": int(global_point),
            "timeouts": int(global_timeout),
            "timeout_ratio": float(global_timeout) / max(1.0, float(global_point)),
            "elapsed_sec": total_elapsed,
            "avg_point_per_sec": float(global_point) / max(1e-6, total_elapsed),
            "motor_union": [int(v) for v in all_motor_ids],
            "point_arrival_tol": float(point_arrival_tol),
            "arrival_confirm_sec": float(arrival_confirm_sec),
            "post_reach_pretighten_sec": float(post_reach_pretighten_sec),
            "result_window_sec": float(result_window_sec),
            "result_window_label": f"{float(dcm.RESULT_WINDOW_START_SEC):.3f}-{float(dcm.RESULT_WINDOW_END_SEC):.3f}",
            "zero_calibrate_every_points": int(zero_every),
            "zero_calibrate_at_start": bool(zero_at_start),
            "zero_between_csv": bool(zero_between_csv),
            "use_antagonist_release": bool(use_antagonist_release),
            "zero_calibration_state": str(zero_calib_state),
            "zero_calibration_fail_streak": int(zero_calib_fail_streak),
            "zero_calibration_events": int(zero_calib_total_events),
            "zero_calibration_failed_events": int(zero_calib_failed_events),
            "zero_calibration_events_file": str(zero_events_csv),
            "unreached_count": int(
                sum(len(points) for points in unreached_by_collection.values())
            ),
            "unreached_by_collection": {
                name: [int(v) for v in points]
                for name, points in unreached_by_collection.items()
            },
            "interrupted": bool(stop_requested or interrupted_by_runtime),
            "startup_first_frame_files": startup_first_frame_files,
            "point_source_map_csv": (
                str(first_dirs["finger_root"] / "point_source_map.csv")
                if not batch_mode
                else ""
            ),
            "point_log_csv": (
                str(first_dirs["finger_root"] / "point_log.csv")
                if not batch_mode
                else ""
            ),
        }

        (hand_root / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[csv-data-collection] finished. summary={hand_root / 'run_summary.json'}")

        try:
            node.set_command_matrix(dcm._build_idle_tor_matrix(defaults_hand))
            time.sleep(0.2)
        except Exception:
            pass
        dcm.ANTAGONIST_RELEASE_ENABLE = bool(original_antagonist_release)
        dcm.ARRIVAL_CONFIRM_SEC = float(original_arrival_confirm_sec)
        dcm.RESULT_WINDOW_START_SEC = float(original_result_window_start_sec)
        dcm.RESULT_WINDOW_END_SEC = float(original_result_window_end_sec)
        dcm.STABLE_HOLD_SEC = float(original_stable_hold_sec)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if dcm.DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from dcm.DEVICES_MSG_IMPORT_ERROR
    main()
