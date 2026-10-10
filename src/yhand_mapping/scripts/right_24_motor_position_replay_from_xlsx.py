#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

from workspace_boundary_config import get_hand_topics


from hand_motor_map import remap_motor_dict

from hand_motor_map import (
    WRIST_MOTOR_IDS_BY_HAND,
)

MOTOR_COUNT = 28
NS_MAIN = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Wrist motors are excluded for 24-motor combined position control.
RIGHT_WRIST_MOTORS = set(WRIST_MOTOR_IDS_BY_HAND['right'])
LEFT_WRIST_MOTORS = set(WRIST_MOTOR_IDS_BY_HAND['left'])

# Same 24-motor position limits used by five_finger_mapping defaults.
MOTOR_FEEDBACK_LIMITS_RIGHT = {20: (-50.0, 500.0),
 18: (-400.0, 50.0),
 13: (-200.0, 400.0),
 21: (-50.0, 500.0),
 22: (-100.0, 400.0),
 28: (-400.0, 50.0),
 17: (-500.0, 50.0),
 11: (-50.0, 500.0),
 14: (-100.0, 100.0),
 15: (-100.0, 100.0),
 1: (-500.0, 50.0),
 2: (-500.0, 50.0),
 9: (-50.0, 500.0),
 16: (-500.0, 50.0),
 10: (-100.0, 100.0),
 23: (-100.0, 100.0),
 3: (-500.0, 50.0),
 7: (-50.0, 50.0),
 24: (-50.0, 500.0),
 8: (-50.0, 50.0),
 27: (-500.0, 50.0),
 25: (-50.0, 500.0),
 6: (-50.0, 50.0),
 4: (-50.0, 50.0)}

MOTOR_FEEDBACK_LIMITS_LEFT = {6: (-50.0, 500.0),
 4: (-600.0, 50.0),
 27: (-30.0, 50.0),
 7: (-50.0, 500.0),
 8: (-30.0, 50.0),
 14: (-600.0, 50.0),
 3: (-600.0, 50.0),
 25: (-50.0, 500.0),
 28: (-30.0, 50.0),
 1: (-30.0, 50.0),
 15: (-600.0, 50.0),
 16: (-600.0, 50.0),
 23: (-50.0, 500.0),
 2: (-600.0, 50.0),
 24: (-30.0, 50.0),
 9: (-30.0, 50.0),
 17: (-600.0, 50.0),
 21: (-30.0, 50.0),
 10: (-50.0, 500.0),
 22: (-30.0, 50.0),
 13: (-600.0, 50.0),
 11: (-50.0, 500.0),
 20: (-30.0, 50.0),
 18: (-30.0, 50.0)}


def _parse_csv_ints(raw: str):
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return [int(v) for v in vals]


def _load_defaults_from_py(defaults_path: Path, hand: str):
    spec = importlib.util.spec_from_file_location("replay_defaults_dynamic", str(defaults_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load defaults file: {defaults_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if defaults is None or hand not in defaults:
        raise RuntimeError(f"DEFAULTS or hand '{hand}' not found in {defaults_path}")

    hand_defaults = defaults[hand]
    out = {}
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.03)):
        values = list(hand_defaults.get(key, []))
        if len(values) < MOTOR_COUNT:
            values += [fallback] * (MOTOR_COUNT - len(values))
        elif len(values) > MOTOR_COUNT:
            values = values[:MOTOR_COUNT]
        out[key] = [float(v) for v in values]

    tor_active_values = list(hand_defaults.get("tor_active", out["tor"]))
    if len(tor_active_values) < MOTOR_COUNT:
        tor_active_values += [out["tor"][i] for i in range(len(tor_active_values), MOTOR_COUNT)]
    elif len(tor_active_values) > MOTOR_COUNT:
        tor_active_values = tor_active_values[:MOTOR_COUNT]
    out["tor_active"] = [float(v) for v in tor_active_values]
    return out


def _read_shared_strings(zf: ZipFile):
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    out = []
    for si in root.findall("a:si", NS_MAIN):
        out.append("".join(t.text or "" for t in si.findall(".//a:t", NS_MAIN)))
    return out


def _parse_cell_value(cell, shared_strings):
    ctype = cell.attrib.get("t", "n")
    if ctype == "inlineStr":
        t = cell.find("a:is/a:t", NS_MAIN)
        return "" if t is None else (t.text or "")

    v = cell.find("a:v", NS_MAIN)
    raw = "" if v is None or v.text is None else v.text
    if ctype == "s":
        return shared_strings[int(raw)] if raw else ""
    return raw


def _read_motor_pos_from_xlsx(path: Path):
    """
    Expected format per file:
    - row1 ignored
    - row(2..29) -> motor_id = row_index - 1
    - col A is position value
    """
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = root.findall(".//a:sheetData/a:row", NS_MAIN)

    row_to_pos = {}
    for row in rows:
        rid = int(row.attrib.get("r", "0"))
        # File format contract: only row2..row29 are motor position rows.
        if rid < 2 or rid > MOTOR_COUNT + 1:
            continue
        a_cell = None
        for c in row.findall("a:c", NS_MAIN):
            ref = c.attrib.get("r", "")
            if ref.startswith("A"):
                a_cell = c
                break
        if a_cell is None:
            continue
        raw = _parse_cell_value(a_cell, shared_strings)
        if raw in ("", None):
            continue
        row_to_pos[rid] = float(raw)

    pos_by_motor_id = {}
    for rid in range(2, MOTOR_COUNT + 2):
        if rid not in row_to_pos:
            raise ValueError(f"Missing row A{rid} in {path}")
        motor_id = rid - 1
        pos_by_motor_id[motor_id] = float(row_to_pos[rid])
    return pos_by_motor_id


def _safe_float(raw):
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _is_finite(v: float) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _select_hand_raw_skeleton_from_msg(raw_data, hand_side_code: float):
    if not raw_data:
        return None
    try:
        arr = [float(v) for v in raw_data]
    except Exception:
        return None
    if len(arr) < 176 or (len(arr) % 176) != 0:
        return None
    chunk_count = len(arr) // 176
    chunks = []
    for i in range(chunk_count):
        start = i * 176
        chunk = arr[start : start + 176]
        chunks.append(chunk)
        if not chunk:
            continue
        side_code = float(chunk[0])
        if abs(side_code - float(hand_side_code)) <= 1e-6:
            return chunk
    # Fallback-1: keep unknown side-code chunks (typically serial not in local glove SN map).
    for chunk in chunks:
        if chunk and abs(float(chunk[0]) - (-1.0)) <= 1e-6:
            return chunk
    # Fallback-2: use first chunk to avoid blocking forever when side-tag mapping is unavailable.
    if chunks:
        return chunks[0]
    return None


def _read_first_row_from_xlsx(path: Path):
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = root.findall(".//a:sheetData/a:row", NS_MAIN)
    if not rows:
        return []
    first_row = rows[0]
    out = []
    for c in first_row.findall("a:c", NS_MAIN):
        raw = _parse_cell_value(c, shared_strings)
        out.append(_safe_float(raw))
    return out


def _read_vector_row_from_xlsx(path: Path, expected_len: int):
    vals = list(_read_first_row_from_xlsx(path))
    if len(vals) < expected_len:
        vals += [float("nan")] * (expected_len - len(vals))
    return vals[:expected_len]


def _read_vector_row_from_csv(path: Path, expected_len: int):
    best = []
    best_finite = -1
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            vals = [_safe_float(v) for v in row]
            finite_count = sum(1 for v in vals if _is_finite(v))
            if finite_count > best_finite:
                best = vals
                best_finite = finite_count
            if finite_count >= expected_len:
                best = vals
                break
    if len(best) < expected_len:
        best += [float("nan")] * (expected_len - len(best))
    return best[:expected_len]


def _read_vector_row_file(path: Path, expected_len: int):
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _read_vector_row_from_xlsx(path, expected_len=expected_len)
    if suffix == ".csv":
        return _read_vector_row_from_csv(path, expected_len=expected_len)
    raise ValueError(f"Unsupported file suffix for vector row: {path}")


def _read_motor_pos_from_csv(path: Path):
    vals = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            v = _safe_float(row[0])
            if _is_finite(v):
                vals.append(float(v))
    if len(vals) < MOTOR_COUNT:
        raise ValueError(f"CSV first column has only {len(vals)} numeric rows (< {MOTOR_COUNT}): {path}")
    out = {}
    for motor_id in range(1, MOTOR_COUNT + 1):
        out[motor_id] = float(vals[motor_id - 1])
    return out


def _read_motor_pos_from_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _read_motor_pos_from_xlsx(path)
    if suffix == ".csv":
        return _read_motor_pos_from_csv(path)
    raise ValueError(f"Unsupported file suffix for motor input: {path}")


def _sorted_data_files(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".xlsx", ".csv")]
    return sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def _find_matching_stem_file(folder: Path, stem: str):
    for suffix in (".xlsx", ".csv"):
        p = folder / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def _average_vectors(rows, expected_len: int):
    if not rows:
        return [float("nan")] * int(expected_len)
    out = []
    for i in range(int(expected_len)):
        finite_vals = []
        for r in rows:
            if i < len(r):
                v = float(r[i])
                if _is_finite(v):
                    finite_vals.append(v)
        if not finite_vals:
            out.append(float("nan"))
        else:
            out.append(float(sum(finite_vals) / len(finite_vals)))
    return out


def _compute_error_stats(expected, measured):
    n = min(len(expected), len(measured))
    max_err = float("nan")
    max_idx_1b = -1
    max_exp = float("nan")
    max_meas = float("nan")
    sum_abs = 0.0
    sum_sq = 0.0
    valid = 0
    for i in range(n):
        e = float(expected[i])
        m = float(measured[i])
        if (not _is_finite(e)) or (not _is_finite(m)):
            continue
        err = abs(m - e)
        if (not _is_finite(max_err)) or (err > max_err):
            max_err = float(err)
            max_idx_1b = i + 1
            max_exp = e
            max_meas = m
        sum_abs += err
        sum_sq += (err * err)
        valid += 1
    mean_abs = (sum_abs / valid) if valid > 0 else float("nan")
    rmse = math.sqrt(sum_sq / valid) if valid > 0 else float("nan")
    return {
        "valid_dims": int(valid),
        "max_abs_err": float(max_err),
        "max_err_index_1based": int(max_idx_1b),
        "expected_at_max": float(max_exp),
        "measured_at_max": float(max_meas),
        "mean_abs_err": float(mean_abs),
        "rmse": float(rmse),
    }


def _write_single_row_csv(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _write_single_row_xlsx(path: Path, row):
    if Workbook is None:
        raise RuntimeError("openpyxl is required to write xlsx files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.append([float(v) for v in row])
    wb.save(path)


def _load_frames(folder: Path):
    files = sorted(
        folder.glob("*.xlsx"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in folder: {folder}")

    frames = []
    for f in files:
        pos_by_motor = _read_motor_pos_from_xlsx(f)
        frames.append((f.name, pos_by_motor))
    return frames


def _load_single_frame(folder: Path, single_file: str):
    p = Path(single_file).expanduser()
    if not p.is_absolute():
        p = (folder / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"single file not found: {p}")
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"single file must be .xlsx: {p}")
    pos_by_motor = _read_motor_pos_from_xlsx(p)
    return [(p.name, pos_by_motor)]


def _build_idle_matrix(defaults_hand):
    matrix = []
    for i in range(MOTOR_COUNT):
        matrix.append([0.0, 0.0, float(defaults_hand["tor"][i]), 0.0, 0.0])
    return matrix


def _matrix_to_msg(matrix):
    msg = YHandMsg()
    msg.motors = [
        Motor(
            pos=float(r[0]),
            vel=float(r[1]),
            tor=float(r[2]),
            kp=float(r[3]),
            kd=float(r[4]),
        )
        for r in matrix
    ]
    msg.pretightenforce.flag = False
    msg.pretightenforce.torque = 0.0
    msg.pretightenforce.ms = 0
    return msg


class Right24ReplayNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        defaults_hand,
        frames,
        controlled_motor_ids,
        limits_map,
        control_period_sec: float,
        status_period_sec: float,
        verbose: bool,
        reach_threshold: float,
        arrival_confirm_frames: int,
        arrival_timeout_sec: float,
        feedback_stale_sec: float,
        unreached_max_log: int,
        log_publish_full: bool,
    ):
        super().__init__(f"{hand}_24_motor_position_replay")
        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.defaults_hand = defaults_hand
        self.frames = list(frames)
        self.controlled_motor_ids = list(controlled_motor_ids)
        self.limits_map = dict(limits_map)
        self.control_period_sec = float(control_period_sec)
        self.status_period_sec = float(status_period_sec)
        self.verbose = bool(verbose)
        self.reach_threshold = float(reach_threshold)
        self.arrival_confirm_frames = max(1, int(arrival_confirm_frames))
        self.arrival_timeout_sec = float(arrival_timeout_sec)
        self.feedback_stale_sec = float(feedback_stale_sec)
        self.unreached_max_log = max(1, int(unreached_max_log))
        self.log_publish_full = bool(log_publish_full)

        if not self.frames:
            raise RuntimeError("No frame loaded.")
        if len(self.frames) > 1:
            self.get_logger().warning(
                f"[24_motor_replay] detected {len(self.frames)} frames, "
                "current mode will only hold the first frame target."
            )

        self.target_file_name, target_pos_by_motor_id = self.frames[0]
        self.target_matrix, self.clip_count = self._build_frame_matrix(target_pos_by_motor_id)
        self.target_pos_by_motor = {mid: float(self.target_matrix[mid - 1][0]) for mid in self.controlled_motor_ids}

        self.start_ts = time.monotonic()
        self.ok_frames = 0
        self.reached_logged = False
        self.timeout_logged = False
        self.last_feedback_pos_by_motor = {}
        self.last_feedback_stamp = None
        self.last_no_feedback_warn_stamp = 0.0
        self.last_stale_warn_stamp = 0.0

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.feedback_subscriber = self.create_subscription(
            YHandMsg, self.feedback_topic, self._on_feedback, 10
        )
        self.publish_timer = self.create_timer(self.control_period_sec, self._publish_tick)
        self.status_timer = self.create_timer(self.status_period_sec, self._status_tick)

        self.get_logger().info(
            "[24_motor_replay] started "
            f"hand={self.hand}, cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, "
            f"target_file={self.target_file_name}, controlled_motor_ids={self.controlled_motor_ids}, "
            f"control_period_sec={self.control_period_sec}, status_period_sec={self.status_period_sec}, "
            f"reach_threshold={self.reach_threshold}, arrival_confirm_frames={self.arrival_confirm_frames}, "
            f"arrival_timeout_sec={self.arrival_timeout_sec}, feedback_stale_sec={self.feedback_stale_sec}, "
            f"unreached_max_log={self.unreached_max_log}, clip_count={self.clip_count}"
        )
        publish_log = (
            {mid: round(self.target_pos_by_motor[mid], 3) for mid in self.controlled_motor_ids}
            if self.log_publish_full
            else {mid: round(self.target_pos_by_motor[mid], 3) for mid in self.controlled_motor_ids[:8]}
        )
        self.get_logger().info(
            f"[24_motor_replay] holding target from file={self.target_file_name}, "
            f"publish_pos={json.dumps(publish_log, ensure_ascii=False)}"
        )

    def _on_feedback(self, msg):
        pos_by_motor = {}
        for idx, motor in enumerate(msg.motors):
            motor_id = idx + 1
            pos_by_motor[motor_id] = float(motor.pos)
        self.last_feedback_pos_by_motor = pos_by_motor
        self.last_feedback_stamp = time.monotonic()

    def _build_frame_matrix(self, pos_by_motor_id):
        matrix = _build_idle_matrix(self.defaults_hand)
        clip_count = 0
        for motor_id in self.controlled_motor_ids:
            idx = int(motor_id) - 1
            target_pos = float(pos_by_motor_id[motor_id])
            limit = self.limits_map.get(motor_id)
            if limit is not None:
                clipped = float(max(limit[0], min(limit[1], target_pos)))
                if abs(clipped - target_pos) > 1e-6:
                    clip_count += 1
                target_pos = clipped

            matrix[idx] = [
                target_pos,
                float(self.defaults_hand["vel"][idx]),
                float(self.defaults_hand["tor_active"][idx]),
                float(self.defaults_hand["kp"][idx]),
                float(self.defaults_hand["kd"][idx]),
            ]
        return matrix, clip_count

    def _publish_tick(self):
        self.publisher.publish(_matrix_to_msg(self.target_matrix))

    def _evaluate_reached(self):
        missing_ids = []
        failed_ids = []
        max_err = 0.0
        unreached = []

        for motor_id in self.controlled_motor_ids:
            target = float(self.target_pos_by_motor[motor_id])
            cur = self.last_feedback_pos_by_motor.get(motor_id)
            if cur is None:
                missing_ids.append(motor_id)
                failed_ids.append(motor_id)
                continue
            err = abs(float(cur) - target)
            if err > max_err:
                max_err = err
            if err > self.reach_threshold:
                failed_ids.append(motor_id)
                unreached.append((motor_id, target, float(cur), float(err)))

        reached = (not missing_ids) and (not failed_ids)
        return reached, failed_ids, missing_ids, max_err, unreached

    def _all_feedback_pos_snapshot(self):
        out = {}
        for motor_id in range(1, MOTOR_COUNT + 1):
            pos = self.last_feedback_pos_by_motor.get(motor_id)
            out[str(motor_id)] = None if pos is None else round(float(pos), 3)
        return out

    def _status_tick(self):
        now = time.monotonic()
        elapsed = now - self.start_ts

        if self.last_feedback_stamp is None:
            if now - self.last_no_feedback_warn_stamp >= 2.0:
                self.last_no_feedback_warn_stamp = now
                self.get_logger().warning(
                    "[24_motor_replay] feedback not received yet, cannot check reached state."
                )
            return

        fb_age = now - self.last_feedback_stamp
        if fb_age > self.feedback_stale_sec:
            if now - self.last_stale_warn_stamp >= 1.0:
                self.last_stale_warn_stamp = now
                self.get_logger().warning(
                    f"[24_motor_replay] feedback stale: fb_age={fb_age:.3f}s > {self.feedback_stale_sec:.3f}s"
                )
            if self.verbose:
                self.get_logger().info(
                    f"[24_motor_replay] feedback_all_pos={json.dumps(self._all_feedback_pos_snapshot(), ensure_ascii=False)}"
                )
            return

        if self.verbose:
            self.get_logger().info(
                f"[24_motor_replay] feedback_all_pos={json.dumps(self._all_feedback_pos_snapshot(), ensure_ascii=False)}"
            )

        reached, failed_ids, missing_ids, max_err, unreached = self._evaluate_reached()
        if reached:
            self.ok_frames += 1
        else:
            self.ok_frames = 0

        if (not self.reached_logged) and self.ok_frames >= self.arrival_confirm_frames:
            self.reached_logged = True
            self.get_logger().info(
                f"[24_motor_replay] reached target and now holding. "
                f"elapsed={elapsed:.3f}s, confirm_frames={self.arrival_confirm_frames}, max_err={max_err:.3f}"
            )

        if (not self.reached_logged) and (not self.timeout_logged) and elapsed >= self.arrival_timeout_sec:
            self.timeout_logged = True
            self.get_logger().warning(
                f"[24_motor_replay] arrival timeout. elapsed={elapsed:.3f}s, max_err={max_err:.3f}, "
                f"failed_count={len(failed_ids)}, missing_count={len(missing_ids)}"
            )

        if self.verbose:
            if unreached:
                unreached.sort(key=lambda x: x[3], reverse=True)
                top = unreached[: self.unreached_max_log]
                detail = {
                    str(mid): {
                        "target": round(target, 3),
                        "feedback": round(feedback, 3),
                        "abs_err": round(abs_err, 3),
                    }
                    for mid, target, feedback, abs_err in top
                }
                self.get_logger().warning(
                    f"[24_motor_replay] holding_status: reached={self.reached_logged}, "
                    f"ok_frames={self.ok_frames}/{self.arrival_confirm_frames}, fb_age={fb_age:.3f}s, "
                    f"max_err={max_err:.3f}, unreached={json.dumps(detail, ensure_ascii=False)}"
                )
            else:
                self.get_logger().info(
                    f"[24_motor_replay] holding_status: reached={self.reached_logged}, "
                    f"ok_frames={self.ok_frames}/{self.arrival_confirm_frames}, fb_age={fb_age:.3f}s, "
                    f"max_err={max_err:.3f}"
                )


class ReplayVerifyNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        glove_raw_skeleton_topic: str,
        defaults_hand,
        control_period_sec: float,
    ):
        super().__init__(f"{hand}_replay_verify")
        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.glove_raw_skeleton_topic = glove_raw_skeleton_topic
        self.defaults_hand = defaults_hand
        self.control_period_sec = float(control_period_sec)
        self.hand_side_code = 0.0 if hand == "left" else 1.0

        self._lock = threading.Lock()
        self.command_matrix = _build_idle_matrix(self.defaults_hand)
        self.feedback_received = False
        self.last_feedback_pos_by_motor = {}
        self.last_feedback_stamp = 0.0
        self.glove_raw_received = False
        self.last_glove_raw_stamp = 0.0
        self.last_glove_skeleton176 = [float("nan")] * 176
        self.glove_raw_seq = 0
        self.glove_raw_side_code = float("nan")
        self._warned_glove_side_fallback = False
        self.target_pos_by_motor = {}
        self.controlled_motor_ids = []

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.feedback_subscriber = self.create_subscription(
            YHandMsg, self.feedback_topic, self._on_feedback, 10
        )
        self.glove_raw_subscriber = self.create_subscription(
            Float64MultiArray, self.glove_raw_skeleton_topic, self._on_glove_raw, 10
        )
        self.publish_timer = self.create_timer(self.control_period_sec, self._publish_tick)

        self.get_logger().info(
            "[replay_verify] started "
            f"hand={self.hand}, cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, "
            f"glove_raw_skeleton_topic={self.glove_raw_skeleton_topic}, "
            f"control_period_sec={self.control_period_sec}"
        )

    def _on_feedback(self, msg):
        pos_by_motor = {}
        for idx, motor in enumerate(msg.motors):
            motor_id = idx + 1
            pos_by_motor[motor_id] = float(motor.pos)
        with self._lock:
            self.last_feedback_pos_by_motor = pos_by_motor
            self.feedback_received = True
            self.last_feedback_stamp = time.monotonic()

    def _on_glove_raw(self, msg):
        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        selected = _select_hand_raw_skeleton_from_msg(data, self.hand_side_code)
        if selected is None:
            return
        side_code = float(selected[0]) if selected else float("nan")
        with self._lock:
            self.last_glove_skeleton176 = [float(v) for v in selected[:176]]
            self.glove_raw_received = True
            self.last_glove_raw_stamp = time.monotonic()
            self.glove_raw_seq += 1
            self.glove_raw_side_code = float(side_code)
        if (
            _is_finite(side_code)
            and abs(float(side_code) - float(self.hand_side_code)) > 1e-6
            and (not self._warned_glove_side_fallback)
        ):
            self._warned_glove_side_fallback = True
            self.get_logger().warning(
                "[replay_verify] glove_raw side code does not match requested hand; "
                f"requested={self.hand_side_code:.1f}, received={side_code:.1f}. "
                "Proceeding with fallback chunk selection."
            )

    def _publish_tick(self):
        with self._lock:
            matrix = [row[:] for row in self.command_matrix]
        self.publisher.publish(_matrix_to_msg(matrix))

    def set_target_from_pos_map(self, pos_by_motor_id, controlled_motor_ids=None):
        matrix = _build_idle_matrix(self.defaults_hand)
        target = {}
        if controlled_motor_ids is None:
            controlled_motor_ids = sorted(int(mid) for mid in pos_by_motor_id.keys())
        for motor_id in controlled_motor_ids:
            mid = int(motor_id)
            if not (1 <= mid <= MOTOR_COUNT):
                continue
            if mid not in pos_by_motor_id:
                continue
            idx = mid - 1
            target_pos = float(pos_by_motor_id[mid])
            matrix[idx] = [
                target_pos,
                float(self.defaults_hand["vel"][idx]),
                float(self.defaults_hand["tor_active"][idx]),
                float(self.defaults_hand["kp"][idx]),
                float(self.defaults_hand["kd"][idx]),
            ]
            target[mid] = target_pos
        with self._lock:
            self.command_matrix = [row[:] for row in matrix]
            self.target_pos_by_motor = dict(target)
            self.controlled_motor_ids = list(sorted(target.keys()))

    def wait_ready(
        self,
        timeout_sec: float,
        require_feedback: bool = True,
        require_glove_raw: bool = True,
        require_cmd_subscriber: bool = False,
    ):
        start = time.monotonic()
        while (time.monotonic() - start) < float(timeout_sec):
            with self._lock:
                fb_ok = (not bool(require_feedback)) or bool(self.feedback_received)
                raw_ok = (not bool(require_glove_raw)) or bool(self.glove_raw_received)
            cmd_ok = (not bool(require_cmd_subscriber)) or (self.count_subscribers(self.cmd_topic) > 0)
            if fb_ok and raw_ok and cmd_ok:
                return True
            time.sleep(0.05)
        return False

    def ready_status(self):
        with self._lock:
            fb_received = bool(self.feedback_received)
            raw_received = bool(self.glove_raw_received)
            fb_age = (time.monotonic() - self.last_feedback_stamp) if fb_received else float("inf")
            raw_age = (time.monotonic() - self.last_glove_raw_stamp) if raw_received else float("inf")
            raw_side = float(self.glove_raw_side_code)
        return (
            f"feedback_received={fb_received}, glove_raw_received={raw_received}, "
            f"feedback_age={fb_age:.3f}, glove_raw_age={raw_age:.3f}, "
            f"glove_raw_side_code={raw_side}, "
            f"fb_pub_count={self.count_publishers(self.feedback_topic)}, "
            f"raw_pub_count={self.count_publishers(self.glove_raw_skeleton_topic)}, "
            f"cmd_sub_count={self.count_subscribers(self.cmd_topic)}"
        )

    def _evaluate_reached(self, reach_threshold: float):
        with self._lock:
            target = dict(self.target_pos_by_motor)
            feedback = dict(self.last_feedback_pos_by_motor)
            controlled = list(self.controlled_motor_ids)
        missing_ids = []
        failed_ids = []
        max_err = 0.0
        for motor_id in controlled:
            tgt = float(target[motor_id])
            cur = feedback.get(motor_id)
            if cur is None:
                missing_ids.append(motor_id)
                failed_ids.append(motor_id)
                continue
            err = abs(float(cur) - tgt)
            if err > max_err:
                max_err = float(err)
            if err > float(reach_threshold):
                failed_ids.append(motor_id)
        reached = (not missing_ids) and (not failed_ids)
        return reached, failed_ids, missing_ids, float(max_err)

    def wait_until_reached(
        self,
        timeout_sec: float,
        reach_threshold: float,
        arrival_confirm_frames: int,
        poll_sec: float = 0.01,
    ):
        confirm_need = max(1, int(arrival_confirm_frames))
        start = time.monotonic()
        ok_frames = 0
        last_max_err = float("inf")
        while (time.monotonic() - start) < float(timeout_sec):
            with self._lock:
                fb_ok = bool(self.feedback_received)
            if not fb_ok:
                time.sleep(float(poll_sec))
                continue
            reached, _, _, max_err = self._evaluate_reached(reach_threshold=reach_threshold)
            last_max_err = float(max_err)
            if reached:
                ok_frames += 1
                if ok_frames >= confirm_need:
                    return True, float(time.monotonic() - start), float(last_max_err)
            else:
                ok_frames = 0
            time.sleep(float(poll_sec))
        return False, float(time.monotonic() - start), float(last_max_err)

    def collect_glove_samples(self, sample_count: int, timeout_sec: float):
        need = max(1, int(sample_count))
        samples = []
        with self._lock:
            last_seq = int(self.glove_raw_seq)
            had_glove = bool(self.glove_raw_received)
            latest = list(self.last_glove_skeleton176)
        if need <= 1 and had_glove:
            return [latest]
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while len(samples) < need and time.monotonic() < deadline:
            with self._lock:
                cur_seq = int(self.glove_raw_seq)
                cur = list(self.last_glove_skeleton176)
                cur_has = bool(self.glove_raw_received)
            if cur_has and cur_seq > last_seq:
                samples.append(cur)
                last_seq = cur_seq
            time.sleep(0.005)
        if not samples and had_glove:
            samples.append(latest)
        return samples


def _resolve_verify_dirs(args):
    motor_dir_raw = str(args.verify_motor_feedback_dir).strip()
    glove_dir_raw = str(args.verify_glove_skeleton_dir).strip()
    folder = Path(args.folder).expanduser().resolve()

    if motor_dir_raw:
        motor_dir = Path(motor_dir_raw).expanduser().resolve()
    else:
        maybe = folder / "motor_feedback"
        motor_dir = maybe if maybe.is_dir() else folder

    if glove_dir_raw:
        glove_dir = Path(glove_dir_raw).expanduser().resolve()
    else:
        sibling = motor_dir.parent / "Glove Skeleton Data"
        if sibling.is_dir():
            glove_dir = sibling
        else:
            maybe = folder / "Glove Skeleton Data"
            glove_dir = maybe

    return motor_dir, glove_dir


def _run_replay_verify(args, hand: str, cmd_topic: str, feedback_topic: str, defaults_hand):
    motor_dir, glove_dir = _resolve_verify_dirs(args)
    if not motor_dir.exists():
        raise FileNotFoundError(f"verify motor_feedback dir not found: {motor_dir}")

    motor_files = _sorted_data_files(motor_dir)
    if not motor_files:
        raise FileNotFoundError(f"No csv/xlsx files found in motor_feedback dir: {motor_dir}")

    # Name alignment only:
    # if glove skeleton dir exists, keep only files whose stem exists in that folder.
    # no numerical comparison is performed in this mode.
    selected_motor_files = list(motor_files)
    if glove_dir.exists():
        aligned = []
        for mf in motor_files:
            if _find_matching_stem_file(glove_dir, mf.stem) is not None:
                aligned.append(mf)
        selected_motor_files = aligned
        skipped = len(motor_files) - len(selected_motor_files)
        if skipped > 0:
            print(
                f"[replay-record] skipped_unmatched_files={skipped} "
                f"(motor_feedback={len(motor_files)}, matched={len(selected_motor_files)})"
            )
    if not selected_motor_files:
        raise RuntimeError(
            "No motor files selected for replay record. "
            f"motor_feedback={motor_dir}, glove_skeleton={glove_dir if glove_dir.exists() else '<not found>'}"
        )

    output_dir_raw = str(args.verify_output_dir).strip()
    if output_dir_raw:
        output_root = Path(output_dir_raw).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (motor_dir.parent / f"replay_record_{stamp}").resolve()
    capture_dir = output_root / "recorded_glove_skeleton"
    output_root.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    node = ReplayVerifyNode(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        glove_raw_skeleton_topic=str(args.glove_raw_skeleton_topic),
        defaults_hand=defaults_hand,
        control_period_sec=float(args.control_period_sec),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    summary_rows = []
    try:
        ready = node.wait_ready(
            timeout_sec=float(args.ready_timeout_sec),
            require_feedback=False,
            require_glove_raw=True,
            require_cmd_subscriber=True,
        )
        if not ready:
            raise RuntimeError(
                "Timeout waiting for glove_raw + cmd subscriber.\n"
                f"feedback_topic={feedback_topic}, glove_raw_topic={args.glove_raw_skeleton_topic}\n"
                f"{node.ready_status()}"
            )

        total = len(selected_motor_files)
        for idx, motor_file in enumerate(selected_motor_files, start=1):
            cmd_pos_by_motor = _read_motor_pos_from_file(motor_file)
            if args.source_motor_map == "legacy":
                cmd_pos_by_motor = remap_motor_dict(hand, cmd_pos_by_motor)
            controlled_motor_ids = sorted(int(mid) for mid in cmd_pos_by_motor.keys())
            node.set_target_from_pos_map(cmd_pos_by_motor, controlled_motor_ids=controlled_motor_ids)
            # Hold at least 30 publish ticks to ensure command is actually consumed by downstream driver.
            hold_sec = max(
                float(args.control_period_sec) * 30.0,
                float(args.verify_post_reach_wait_sec),
            )
            node._publish_tick()
            if hold_sec > 0.0:
                time.sleep(hold_sec)

            samples = node.collect_glove_samples(
                sample_count=int(args.verify_glove_sample_count),
                timeout_sec=float(args.verify_glove_sample_timeout_sec),
            )
            measured_skeleton176 = _average_vectors(samples, expected_len=176)

            capture_path = capture_dir / f"{motor_file.stem}.xlsx"
            _write_single_row_xlsx(capture_path, measured_skeleton176)

            row = {
                "seq": int(idx),
                "file": motor_file.name,
                "motor_file": str(motor_file),
                "sample_count": int(len(samples)),
                "recorded_glove_xlsx": str(capture_path),
            }
            summary_rows.append(row)

            print(
                f"[replay-record] {idx}/{total} file={motor_file.name} "
                f"samples={len(samples)} saved={capture_path.name}"
            )

        summary_csv = output_root / "record_manifest.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "seq",
                    "file",
                    "motor_file",
                    "sample_count",
                    "recorded_glove_xlsx",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        global_summary = {
            "hand": hand,
            "motor_feedback_dir": str(motor_dir),
            "glove_skeleton_dir_for_name_alignment": str(glove_dir) if glove_dir.exists() else "",
            "output_root": str(output_root),
            "selected_files": int(len(selected_motor_files)),
            "finished_files": int(len(summary_rows)),
            "mode": "record_only",
        }
        summary_json = output_root / "record_summary.json"
        summary_json.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[replay-record] manifest_csv={summary_csv}")
        print(f"[replay-record] summary_json={summary_json}")
    finally:
        try:
            if bool(args.publish_idle_on_exit):
                idle = _build_idle_matrix(defaults_hand)
                node.publisher.publish(_matrix_to_msg(idle))
                time.sleep(0.02)
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            if spin_thread.is_alive():
                spin_thread.join(timeout=2.0)
        except Exception:
            pass
        node.destroy_node()


def _parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replay right-hand 24-motor position commands from xlsx files (drop wrist motors)."
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--source-motor-map", choices=["current", "legacy"], default="current")
    parser.add_argument(
        "--folder",
        default=str(script_dir / "2026.5.18"),
        help="Folder containing xlsx frames.",
    )
    parser.add_argument(
        "--single-file",
        default="",
        help="Replay only one xlsx file. Supports absolute path or name under --folder (e.g. 04978.xlsx).",
    )
    parser.add_argument("--cmd-topic", default="", help="Override command topic")
    parser.add_argument("--feedback-topic", default="", help="Override feedback topic")
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default="/glove_joint_data_raw_skeleton",
        help="Raw glove skeleton topic (Float64MultiArray, 176/352 payload).",
    )
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
        help="Path to muscle_adjustment_defaults.py",
    )
    parser.add_argument(
        "--exclude-motor-ids",
        default="",
        help="Comma separated motor IDs to exclude. Empty uses hand default wrist set.",
    )
    parser.add_argument(
        "--frame-interval-sec",
        type=float,
        default=0.2,
        help="Deprecated in single-file hold mode.",
    )
    parser.add_argument(
        "--control-period-sec",
        type=float,
        default=0.02,
        help="Command publish period while holding target (same role as motor_position_test CONTROL_PERIOD_SEC).",
    )
    parser.add_argument(
        "--status-period-sec",
        type=float,
        default=0.5,
        help="Status log/update period for reached checking.",
    )
    parser.add_argument("--loop", action="store_true", help="Deprecated in single-file hold mode.")
    parser.add_argument("--verbose", action="store_true", help="Print holding status details.")
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=7.0,
        help="Absolute position error threshold to judge motor reached (align with motor_position_test ARRIVAL_POS_TOL).",
    )
    parser.add_argument(
        "--arrival-confirm-frames",
        type=int,
        default=30,
        help="Consecutive reached frames required (align with motor_position_test ARRIVAL_CONFIRM_FRAMES).",
    )
    parser.add_argument(
        "--arrival-timeout-sec",
        type=float,
        default=6.0,
        help="Arrival timeout in seconds (align with motor_position_test replay timeout).",
    )
    parser.add_argument(
        "--feedback-stale-sec",
        type=float,
        default=1.0,
        help="Feedback stale threshold in seconds (align with motor_position_test FEEDBACK_STALE_SEC).",
    )
    parser.add_argument(
        "--unreached-max-log",
        type=int,
        default=24,
        help="Maximum unreached motors to print in one log line.",
    )
    parser.add_argument(
        "--log-publish-full",
        action="store_true",
        help="Print all controlled published target positions instead of only first 8.",
    )
    parser.add_argument(
        "--publish-idle-on-exit",
        action="store_true",
        help="Publish idle(0-pos) command before exit. Default is disabled to keep last command.",
    )
    parser.add_argument(
        "--verify-mode",
        action="store_true",
        help=(
            "Replay record mode: use motor_feedback files as cmd input, subscribe glove raw "
            "skeleton, and save captured glove skeleton xlsx files with aligned names."
        ),
    )
    parser.add_argument(
        "--verify-motor-feedback-dir",
        default="",
        help=(
            "Directory containing motor_feedback xlsx/csv files. "
            "Empty: use <folder>/motor_feedback if exists, else use --folder."
        ),
    )
    parser.add_argument(
        "--verify-glove-skeleton-dir",
        default="",
        help=(
            "Directory containing reference glove skeleton xlsx/csv files. "
            "Empty: use sibling 'Glove Skeleton Data' of motor dir."
        ),
    )
    parser.add_argument(
        "--verify-output-dir",
        default="",
        help="Output directory for record manifest/summary and captured glove xlsx data.",
    )
    parser.add_argument(
        "--ready-timeout-sec",
        type=float,
        default=20.0,
        help="Startup timeout waiting for glove_raw + cmd subscriber.",
    )
    parser.add_argument(
        "--verify-glove-sample-count",
        type=int,
        default=5,
        help="Number of glove_raw samples to average per replayed file.",
    )
    parser.add_argument(
        "--verify-glove-sample-timeout-sec",
        type=float,
        default=1.5,
        help="Timeout waiting glove_raw samples for one replayed file.",
    )
    parser.add_argument(
        "--verify-post-reach-wait-sec",
        type=float,
        default=0.2,
        help="Extra wait time after reached and before glove sampling.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if YHandMsg is None or Motor is None:
        raise RuntimeError(f"Cannot import devices_pkg.msg: {DEVICES_MSG_IMPORT_ERROR}")

    hand = str(args.hand).strip().lower()
    folder = Path(args.folder).expanduser().resolve()
    defaults_path = Path(args.defaults_path).expanduser().resolve()

    cmd_topic, feedback_topic = get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = str(args.cmd_topic).strip()
    if args.feedback_topic:
        feedback_topic = str(args.feedback_topic).strip()

    defaults_hand = _load_defaults_from_py(defaults_path, hand=hand)
    if bool(args.verify_mode):
        rclpy.init()
        try:
            _run_replay_verify(
                args=args,
                hand=hand,
                cmd_topic=cmd_topic,
                feedback_topic=feedback_topic,
                defaults_hand=defaults_hand,
            )
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        return

    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    if str(args.single_file).strip():
        frames = _load_single_frame(folder, str(args.single_file).strip())
    else:
        frames = _load_frames(folder)
    if args.source_motor_map == "legacy":
        frames = [(name, remap_motor_dict(hand, positions)) for name, positions in frames]
    if len(frames) != 1:
        print(
            f"提示: 当前读取到 {len(frames)} 个位置文件，脚本将仅使用第一个并持续保持目标位姿。"
        )
    if bool(args.loop):
        print("提示: --loop 在当前单文件保持模式中不生效。")
    if float(args.frame_interval_sec) != 0.2:
        print("提示: --frame-interval-sec 在当前单文件保持模式中不生效。")

    if args.exclude_motor_ids.strip():
        exclude_motor_ids = set(_parse_csv_ints(args.exclude_motor_ids))
    else:
        exclude_motor_ids = set(RIGHT_WRIST_MOTORS if hand == "right" else LEFT_WRIST_MOTORS)
    controlled_motor_ids = [mid for mid in range(1, MOTOR_COUNT + 1) if mid not in exclude_motor_ids]
    if len(controlled_motor_ids) != 24:
        print(
            f"警告: 当前受控电机数量为 {len(controlled_motor_ids)} (非24)。"
            f"exclude={sorted(exclude_motor_ids)} controlled={controlled_motor_ids}"
        )

    limits_map = MOTOR_FEEDBACK_LIMITS_RIGHT if hand == "right" else MOTOR_FEEDBACK_LIMITS_LEFT
    if set(controlled_motor_ids) != set(limits_map.keys()):
        print(
            "提示: 受控电机ID集合与 five_finger 默认24电机集合不完全一致，"
            "将按当前 controlled_motor_ids 发布。"
        )

    rclpy.init()
    node = Right24ReplayNode(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        defaults_hand=defaults_hand,
        frames=frames,
        controlled_motor_ids=controlled_motor_ids,
        limits_map=limits_map,
        control_period_sec=float(args.control_period_sec),
        status_period_sec=float(args.status_period_sec),
        verbose=bool(args.verbose),
        reach_threshold=float(args.reach_threshold),
        arrival_confirm_frames=int(args.arrival_confirm_frames),
        arrival_timeout_sec=float(args.arrival_timeout_sec),
        feedback_stale_sec=float(args.feedback_stale_sec),
        unreached_max_log=int(args.unreached_max_log),
        log_publish_full=bool(args.log_publish_full),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if bool(args.publish_idle_on_exit):
                idle = _build_idle_matrix(defaults_hand)
                node.publisher.publish(_matrix_to_msg(idle))
                time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
