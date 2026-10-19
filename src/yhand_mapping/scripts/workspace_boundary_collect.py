#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

from workspace_boundary_config import (
    DEFAULT_HAND,
    DEFAULT_MAX_OUTWARD_EXPAND,
    DEFAULT_SAMPLE_HZ,
    FINGER_LABELS,
    FINGER_ORDER,
    MOTOR_COUNT,
    SIMULTANEOUS_EXTENSOR_CURRENT,
    SIMULTANEOUS_FLEXOR_CURRENT,
    SIMULTANEOUS_SIDE_CURRENT,
    build_model_specs,
    get_hand_topics,
    get_simultaneous_tension_currents,
)
from workspace_boundary_fit import (
    fit_boundary_2d,
    fit_boundary_3d,
    projected_points_stats,
    to_jsonable_float_array,
    to_jsonable_float_matrix,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
else:  # pragma: no cover
    # Try common CJK-capable fonts first; fallback still works for ASCII labels.
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


class PassiveBoundaryNode(Node):
    def __init__(self, hand: str, cmd_topic: str, feedback_topic: str, tension_currents):
        super().__init__(f"workspace_boundary_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError("Cannot import devices_pkg.msg. Please source workspace first.")

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.tension_currents = np.asarray(tension_currents, dtype=np.float64)
        if self.tension_currents.shape[0] != MOTOR_COUNT:
            raise ValueError(
                f"tension_currents length must be {MOTOR_COUNT}, got {self.tension_currents.shape[0]}"
            )

        self._lock = threading.Lock()
        self.latest_pos = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_vel = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_tor = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.feedback_seq = 0
        self._sample_hook = None

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.subscription = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.publish_timer = self.create_timer(0.02, self._publish_tension_cb)

        self.get_logger().info(
            "[workspace_boundary] started. hand=%s, cmd_topic=%s, feedback_topic=%s, "
            "tension_current[min=%.4fA max=%.4fA]"
            % (
                self.hand,
                self.cmd_topic,
                self.feedback_topic,
                float(np.min(self.tension_currents)),
                float(np.max(self.tension_currents)),
            )
        )

    def _feedback_cb(self, msg):
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        now_mono = time.monotonic()
        now_wall = time.time()
        hook = None
        hook_sample = None
        with self._lock:
            for i in range(MOTOR_COUNT):
                if i < len(motors):
                    m = motors[i]
                    self.latest_pos[i] = float(m.pos)
                    self.latest_vel[i] = float(m.vel)
                    self.latest_tor[i] = float(m.tor)
            self.feedback_received = True
            self.last_feedback_ts = now_mono
            self.feedback_seq += 1
            if self._sample_hook is not None:
                hook = self._sample_hook
                hook_sample = {
                    "wall": float(now_wall),
                    "feedback_mono": float(now_mono),
                    "feedback_seq": int(self.feedback_seq),
                    "pos": self.latest_pos.copy(),
                    "vel": self.latest_vel.copy(),
                    "tor": self.latest_tor.copy(),
                }
        if hook is not None and hook_sample is not None:
            try:
                hook(hook_sample)
            except Exception:
                # Sampling hook should never break feedback processing.
                pass

    def _publish_tension_cb(self):
        with self._lock:
            self._publish_currents_locked()

    def set_tension_currents(self, currents):
        values = np.asarray(currents, dtype=np.float64)
        if values.shape != (MOTOR_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {MOTOR_COUNT} finite motor currents.")
        with self._lock:
            self.tension_currents = values.copy()
            self._publish_currents_locked()

    def _publish_currents_locked(self):
        msg = YHandMsg()
        motor_msgs = []
        for i in range(MOTOR_COUNT):
            motor_msgs.append(
                Motor(
                    pos=0.0,
                    vel=0.0,
                    tor=float(self.tension_currents[i]),
                    kp=0.0,
                    kd=0.0,
                )
            )
        msg.motors = motor_msgs
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.publisher.publish(msg)

    def wait_feedback(self, timeout_sec: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            with self._lock:
                if self.feedback_received:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self):
        with self._lock:
            if not self.feedback_received:
                return None
            return {
                "mono": time.monotonic(),
                "feedback_age": time.monotonic() - self.last_feedback_ts,
                "feedback_seq": int(self.feedback_seq),
                "feedback_mono": float(self.last_feedback_ts),
                "wall": time.time(),
                "pos": self.latest_pos.copy(),
                "vel": self.latest_vel.copy(),
                "tor": self.latest_tor.copy(),
            }

    def set_sample_hook(self, hook):
        with self._lock:
            self._sample_hook = hook


class RealtimePanel:
    def __init__(self, model_specs, enabled: bool = True):
        self.enabled = bool(enabled) and (plt is not None)
        self.model_specs = model_specs
        self.model_by_finger = {}
        for spec in self.model_specs:
            self.model_by_finger.setdefault(spec.finger, []).append(spec)

        self.fig = None
        self.axes = {}
        self.current_finger = None
        self.combo_views = []
        if not self.enabled:
            return

        plt.ion()
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle("Workspace Boundary Panel", fontsize=12)

    @staticmethod
    def _layout_cols_for_finger(finger: str, n_plots: int) -> int:
        if finger in ("index", "middle"):
            return 3
        if finger in ("ring", "little"):
            return 2
        if finger == "thumb":
            return 4
        return max(2, min(4, n_plots))

    def _build_combo_views_for_finger(self, finger: str):
        views = []
        specs = self.model_by_finger.get(finger, [])
        for spec in specs:
            if spec.dims == 2:
                for direction in ("in", "out"):
                    views.append(
                        {
                            "spec": spec,
                            "direction": direction,
                            "opp_mode": None,
                        }
                    )
            else:
                for direction in ("in", "out"):
                    for opp_mode in ("off", "on"):
                        views.append(
                            {
                                "spec": spec,
                                "direction": direction,
                                "opp_mode": opp_mode,
                            }
                        )
        return views

    @staticmethod
    def _combo_key(combo_view):
        spec = combo_view["spec"]
        return (spec.model_id, combo_view["direction"], combo_view["opp_mode"])

    @staticmethod
    def _combo_id(combo_view):
        spec = combo_view["spec"]
        direction = combo_view["direction"]
        if int(spec.dims) == 2:
            return f"{spec.model_id}_{direction}"
        return f"{spec.model_id}_{direction}_opp_{combo_view['opp_mode']}"

    def _build_layout_for_finger(self, finger: str):
        self.combo_views = self._build_combo_views_for_finger(finger)
        self.axes = {}
        self.fig.clf()
        self.current_finger = finger
        n_plots = max(1, len(self.combo_views))
        n_cols = self._layout_cols_for_finger(finger, n_plots)
        n_rows = int(np.ceil(n_plots / float(n_cols)))
        self.fig.set_size_inches(max(10.0, 5.0 * n_cols), max(6.0, 3.6 * n_rows), forward=True)
        for idx, combo_view in enumerate(self.combo_views, start=1):
            spec = combo_view["spec"]
            if spec.dims == 3:
                ax = self.fig.add_subplot(n_rows, n_cols, idx, projection="3d")
            else:
                ax = self.fig.add_subplot(n_rows, n_cols, idx)
            self.axes[self._combo_key(combo_view)] = ax
        self.fig.suptitle(f"Workspace Boundary Panel - {finger}", fontsize=13)
        try:
            self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            pass

    @staticmethod
    def _cap_points(pts: np.ndarray, max_points_each: int):
        if pts.ndim == 2 and pts.shape[0] > max_points_each:
            step = int(np.ceil(pts.shape[0] / float(max_points_each)))
            return pts[::step]
        return pts

    @staticmethod
    def _normalize_points(pts: np.ndarray, dims: int):
        arr = np.asarray(pts, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, dims), dtype=np.float64)
        if arr.ndim == 1:
            if arr.size % dims != 0:
                return np.empty((0, dims), dtype=np.float64)
            arr = arr.reshape((-1, dims))
        if arr.ndim != 2 or arr.shape[1] != dims:
            return np.empty((0, dims), dtype=np.float64)
        return arr

    @staticmethod
    def _draw_2d_boundaries_swapped(ax, fit_info, color: str, label_prefix: str):
        if fit_info is None:
            return
        if fit_info.get("status") != "ok":
            return
        y_flex = np.asarray(fit_info.get("x_grid", []), dtype=np.float64)
        x_side_min = np.asarray(fit_info.get("y_min", []), dtype=np.float64)
        x_side_max = np.asarray(fit_info.get("y_max", []), dtype=np.float64)
        if y_flex.size == 0 or x_side_min.size == 0 or x_side_max.size == 0:
            return
        ax.plot(x_side_min, y_flex, color=color, linewidth=1.1, linestyle="--", label=f"{label_prefix} side_min")
        ax.plot(x_side_max, y_flex, color=color, linewidth=1.1, linestyle="-", label=f"{label_prefix} side_max")

    @staticmethod
    def _draw_3d_boundaries_thumb(ax, fit_info, color: str):
        if fit_info is None or fit_info.get("status") != "ok":
            return
        y_flex = np.asarray(fit_info.get("x_grid", []), dtype=np.float64)
        z = np.asarray(fit_info.get("z_grid", []), dtype=np.float64)
        side_min = np.asarray(fit_info.get("y_min", []), dtype=np.float64)
        side_max = np.asarray(fit_info.get("y_max", []), dtype=np.float64)
        if y_flex.size == 0 or z.size == 0 or side_min.size == 0 or side_max.size == 0:
            return
        if side_min.shape != (y_flex.size, z.size) or side_max.shape != (y_flex.size, z.size):
            return

        yy, zz = np.meshgrid(y_flex, z, indexing="ij")
        lo = side_min.copy()
        hi = side_max.copy()

        # Axis convention for thumb 3D combo view:
        # x-axis: side, y-axis: flex/ext, z-axis: opposition.
        ax.plot_wireframe(lo, yy, zz, color=color, rstride=2, cstride=2, linewidth=0.6, alpha=0.6)
        ax.plot_wireframe(hi, yy, zz, color=color, rstride=2, cstride=2, linewidth=0.9, alpha=0.85)

    def _draw_combo(self, ax, combo_view, projected_by_model, fitted_models, max_points_each):
        spec = combo_view["spec"]
        direction = combo_view["direction"]
        opp_mode = combo_view["opp_mode"]
        combo_id = self._combo_id(combo_view)
        fit_combo = None if fitted_models is None else fitted_models.get(combo_id)
        side_motor = spec.y_in_motor if direction == "in" else spec.y_out_motor
        color = "tab:red" if direction == "in" else "tab:blue"

        model_data = projected_by_model.get(spec.model_id, {})
        points = self._normalize_points(model_data.get(direction, []), 2 if spec.dims == 2 else 3)
        points = self._cap_points(points, max_points_each)

        if spec.dims == 2:
            if points.size > 0:
                ax.scatter(points[:, 1], points[:, 0], s=5, c=color, alpha=0.45, label=f"{direction} points")
            self._draw_2d_boundaries_swapped(ax, fit_combo, color=color, label_prefix=direction)
            ax.set_title(f"{spec.model_id} | {direction}", fontsize=10)
            ax.set_xlabel(f"side(m{side_motor})")
            ax.set_ylabel(f"flex/ext(m{spec.x_motor})")
            ax.grid(alpha=0.2)
            if points.size > 0 or (fit_combo is not None and fit_combo.get("status") == "ok"):
                ax.legend(loc="best", fontsize=7)
            return

        # For thumb, opp_on/off are control-mode labels (active or passive opposition),
        # not geometric split by z-position. Both modes share the same sampled workspace points.
        points_use = points
        if points_use.size > 0:
            ax.scatter(
                points_use[:, 1],
                points_use[:, 0],
                points_use[:, 2],
                s=5,
                c=color,
                alpha=0.45,
                label=f"{direction} opp_{opp_mode}",
            )
        else:
            points_use = np.empty((0, 3), dtype=np.float64)

        self._draw_3d_boundaries_thumb(ax, fit_combo, color=color)

        ax.set_title(f"{spec.model_id} | {direction} | opp_{opp_mode}", fontsize=10)
        ax.set_xlabel(f"side(m{side_motor})")
        ax.set_ylabel(f"flex/ext(m{spec.x_motor})")
        ax.set_zlabel(f"opp(m{spec.z_motor})")
        ax.grid(alpha=0.2)
        ax.view_init(elev=18, azim=-58)
        if points_use.size > 0 or (fit_combo is not None and fit_combo.get("status") == "ok"):
            ax.legend(loc="best", fontsize=7)

    def refresh(self, projected_by_model, selected_finger=None, fitted_models=None, max_points_each=2500):
        if not self.enabled or self.fig is None:
            return

        finger = selected_finger if selected_finger in self.model_by_finger else None
        if finger is None:
            if self.model_specs:
                finger = self.model_specs[0].finger
            else:
                return

        if finger != self.current_finger:
            self._build_layout_for_finger(finger)

        for combo_view in self.combo_views:
            key = self._combo_key(combo_view)
            ax = self.axes.get(key)
            if ax is None:
                continue
            ax.clear()
            self._draw_combo(
                ax=ax,
                combo_view=combo_view,
                projected_by_model=projected_by_model,
                fitted_models=fitted_models,
                max_points_each=max_points_each,
            )

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(0.001)

    def save_snapshot(self, path: Path):
        if not self.enabled or self.fig is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150)

    def save_all_finger_snapshots(
        self,
        output_dir: Path,
        projected_by_model,
        selected_finger=None,
        fitted_models=None,
        max_points_each=2500,
        stamp: str | None = None,
    ):
        if not self.enabled or self.fig is None:
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        save_stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        prev_finger = self.current_finger
        ordered_fingers = [f for f in FINGER_ORDER if f in self.model_by_finger]
        saved_paths = []

        try:
            for finger in ordered_fingers:
                self.refresh(
                    projected_by_model,
                    selected_finger=finger,
                    fitted_models=fitted_models,
                    max_points_each=max_points_each,
                )
                path = output_dir / f"panel_{finger}_{save_stamp}.png"
                self.fig.savefig(path, dpi=150)
                saved_paths.append(path)
        finally:
            restore_finger = None
            if selected_finger in self.model_by_finger:
                restore_finger = selected_finger
            elif prev_finger in self.model_by_finger:
                restore_finger = prev_finger
            elif ordered_fingers:
                restore_finger = ordered_fingers[0]
            if restore_finger is not None:
                self.refresh(
                    projected_by_model,
                    selected_finger=restore_finger,
                    fitted_models=fitted_models,
                    max_points_each=max_points_each,
                )

        return saved_paths


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _project_samples(samples_by_finger, model_specs):
    projected = {spec.model_id: {"in": [], "out": []} for spec in model_specs}
    for spec in model_specs:
        for sample in samples_by_finger.get(spec.finger, []):
            pos = sample["pos"]
            x = float(pos[spec.x_motor - 1])
            for direction, y_motor in (("in", spec.y_in_motor), ("out", spec.y_out_motor)):
                point = [x, float(pos[y_motor - 1])]
                if spec.dims == 3:
                    point.append(float(pos[spec.z_motor - 1]))
                projected[spec.model_id][direction].append(point)
    return projected


def _to_numpy_projected(projected):
    out = {}
    for model_id, v in projected.items():
        out[model_id] = {
            "in": np.asarray(v.get("in", []), dtype=np.float64),
            "out": np.asarray(v.get("out", []), dtype=np.float64),
        }
    return out


def _combo_id(spec, direction: str, opp_mode: str | None = None) -> str:
    if int(spec.dims) == 2:
        return f"{spec.model_id}_{direction}"
    return f"{spec.model_id}_{direction}_opp_{opp_mode}"


def _fit_all_models(projected_np, model_specs, max_outward_expand, tension_currents):
    fitted = {}
    for spec in model_specs:
        for direction in ("in", "out"):
            pts = projected_np[spec.model_id][direction]
            y_motor = int(spec.y_in_motor if direction == "in" else spec.y_out_motor)
            if spec.dims == 2:
                combo_id = _combo_id(spec, direction, None)
                stats = projected_points_stats(pts)
                base = {
                    "label": spec.label,
                    "finger": spec.finger,
                    "model_id": spec.model_id,
                    "dims": 2,
                    "direction": direction,
                    "opposition_mode": "none",
                    "collection": {
                        "mode": "per_finger",
                        "sample_scope": spec.finger,
                        "tension_current_by_motor": [float(v) for v in tension_currents],
                        "kp": 0.0,
                        "kd": 0.0,
                    },
                    "axes": {
                        "x_motor": int(spec.x_motor),
                        "y_motor": y_motor,
                        "z_motor": None,
                    },
                }
                if pts.shape[0] < 20:
                    fitted[combo_id] = {
                        **base,
                        "status": "insufficient_points",
                        "stats": stats,
                    }
                    continue
                fit = fit_boundary_2d(pts, max_outward_expand=max_outward_expand, grid_size=90)
                fitted[combo_id] = {
                    **base,
                    "status": "ok",
                    "fit_type": "smooth_envelope_2d",
                    "point_count": int(fit.point_count),
                    "x_grid": to_jsonable_float_array(fit.x_grid),
                    "y_min": to_jsonable_float_array(fit.y_min),
                    "y_max": to_jsonable_float_array(fit.y_max),
                    "stats": stats,
                }
                continue

            stats = projected_points_stats(pts)
            if pts.shape[0] < 45:
                for opp_mode in ("off", "on"):
                    combo_id = _combo_id(spec, direction, opp_mode)
                    base = {
                        "label": spec.label,
                        "finger": spec.finger,
                        "model_id": spec.model_id,
                        "dims": 3,
                        "direction": direction,
                        "opposition_mode": opp_mode,
                        "opp_mode_source": "control_mode",
                        "axes": {
                            "x_motor": int(spec.x_motor),
                            "y_motor": y_motor,
                            "z_motor": int(spec.z_motor) if spec.z_motor is not None else None,
                        },
                    }
                    fitted[combo_id] = {
                        **base,
                        "status": "insufficient_points",
                        "stats": stats,
                    }
                continue

            fit = fit_boundary_3d(
                pts,
                max_outward_expand=max_outward_expand,
                x_grid_size=34,
                z_grid_size=26,
            )
            fit_payload = {
                "status": "ok",
                "fit_type": "smooth_surface_envelope_3d",
                "point_count": int(fit.point_count),
                "x_grid": to_jsonable_float_array(fit.x_grid),
                "z_grid": to_jsonable_float_array(fit.z_grid),
                "y_min": to_jsonable_float_matrix(fit.y_min),
                "y_max": to_jsonable_float_matrix(fit.y_max),
                "stats": stats,
            }

            for opp_mode in ("off", "on"):
                combo_id = _combo_id(spec, direction, opp_mode)
                base = {
                    "label": spec.label,
                    "finger": spec.finger,
                    "model_id": spec.model_id,
                    "dims": 3,
                    "direction": direction,
                    "opposition_mode": opp_mode,
                    "opp_mode_source": "control_mode",
                    "axes": {
                        "x_motor": int(spec.x_motor),
                        "y_motor": y_motor,
                        "z_motor": int(spec.z_motor) if spec.z_motor is not None else None,
                    },
                }
                fitted[combo_id] = {**base, **fit_payload}
    return fitted


def _save_session_csv(path: Path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["wall_time_sec"]
    headers.extend([f"pos_{i}" for i in range(1, MOTOR_COUNT + 1)])
    headers.extend([f"vel_{i}" for i in range(1, MOTOR_COUNT + 1)])
    headers.extend([f"tor_{i}" for i in range(1, MOTOR_COUNT + 1)])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in samples:
            row = [f"{float(s['wall']):.6f}"]
            row.extend([f"{float(v):.9f}" for v in s["pos"]])
            row.extend([f"{float(v):.9f}" for v in s["vel"]])
            row.extend([f"{float(v):.9f}" for v in s["tor"]])
            writer.writerow(row)


def _read_saved_workspace_json(path: Path, hand: str):
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("hand") != hand:
        raise ValueError(f"Invalid saved workspace or hand mismatch: {path}")
    return obj


def _write_workspace_json(path: Path, obj, backup_existing=False):
    content = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.stem}_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        if backup_existing and path.exists():
            backup_dir = path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            shutil.copy2(path, backup)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_boundary_models(path: Path, out_obj, updated_fingers):
    """Replace explicitly changed fingers while preserving the others on disk."""
    previous = _read_saved_workspace_json(path, out_obj["hand"])
    previous_combos = previous.get("combos", {})
    if not isinstance(previous_combos, dict):
        raise ValueError(f"Invalid saved workspace combos: {path}")
    updated = set(updated_fingers)
    combos = dict(out_obj["combos"])
    for combo_id, model in previous_combos.items():
        if model["finger"] not in updated:
            combos[combo_id] = model
    merged = {**previous, **out_obj, "updated_fingers": sorted(updated), "combos": combos}
    _write_workspace_json(path, merged, backup_existing=True)
    return merged


def _save_run_index(path: Path, hand: str, samples_by_finger, model_specs, updated_fingers=None):
    previous = _read_saved_workspace_json(path, hand)
    updated = (set(updated_fingers) if updated_fingers is not None else
               {finger for finger, samples in samples_by_finger.items() if samples})
    finger_counts = {**dict.fromkeys(FINGER_ORDER, 0), **previous.get("finger_counts", {})}
    for finger in updated:
        finger_counts[finger] = len(samples_by_finger[finger])
    combos = [(_combo_id(spec, direction), spec, direction)
              for spec in model_specs if spec.dims == 2 for direction in ("in", "out")]
    obj = {
        "hand": hand,
        "updated_at": datetime.now().isoformat(),
        "collection_mode": "per_finger",
        "finger_counts": finger_counts,
        "combo_counts": {key: finger_counts[spec.finger] for key, spec, _ in combos},
        "combo_axis_motors": {
            key: [int(spec.x_motor), int(spec.y_in_motor if direction == "in" else spec.y_out_motor)]
            for key, spec, direction in combos
        },
    }
    _write_workspace_json(path, obj)


def _clear_finger_raw_data(hand_dir: Path, finger: str):
    finger_dir = hand_dir / "raw" / finger
    if not finger_dir.exists():
        return
    for path in finger_dir.glob("session_*.csv"):
        if path.is_file():
            path.unlink()


def _record_one_segment(
    node: PassiveBoundaryNode,
    sample_hz: float,
    preview_refresh_cb=None,
    preview_interval_sec: float = 0.5,
    live_sample_cb=None,
):
    stop_evt = threading.Event()
    aborted_evt = threading.Event()
    samples = []
    samples_lock = threading.Lock()

    def _wait_stop():
        try:
            input("录制中... 按回车停止当前段采集。")
        except KeyboardInterrupt:
            aborted_evt.set()
        finally:
            stop_evt.set()

    stop_thread = threading.Thread(target=_wait_stop, daemon=True)
    stop_thread.start()

    # Capture in feedback callback thread, so plotting cost does not reduce sampling throughput.
    period = 1.0 / max(1.0, float(sample_hz))
    last_accept_mono = [None]

    def _on_feedback(sample):
        mono = float(sample["feedback_mono"])
        if last_accept_mono[0] is not None and (mono - last_accept_mono[0]) < (period * 0.95):
            return
        last_accept_mono[0] = mono
        with samples_lock:
            samples.append(sample)
        if live_sample_cb is not None:
            live_sample_cb(sample)

    node.set_sample_hook(_on_feedback)
    last_preview_ts = 0.0

    try:
        while not stop_evt.is_set():
            now = time.monotonic()
            if preview_refresh_cb is not None and (now - last_preview_ts) >= preview_interval_sec:
                preview_refresh_cb()
                last_preview_ts = now
            time.sleep(0.01)
    finally:
        node.set_sample_hook(None)

    stop_thread.join(timeout=0.1)
    with samples_lock:
        out_samples = list(samples)
    return out_samples, bool(aborted_evt.is_set())


def _prompt_finger():
    print("\n选择采集手指:")
    for idx, finger in enumerate(FINGER_ORDER, start=1):
        print(f"  {idx}. {FINGER_LABELS[finger]} ({finger})")
    print("  f. 拟合并保存边界")
    print("  p. 刷新图")
    print("  x. 退出")
    raw = input("输入选项: ").strip().lower()
    if raw in ("x", "exit", "quit"):
        return "exit", None
    if raw in ("f", "fit"):
        return "fit", None
    if raw in ("p", "plot"):
        return "plot", None
    try:
        idx = int(raw)
    except ValueError:
        return "invalid", None
    if 1 <= idx <= len(FINGER_ORDER):
        return "finger", FINGER_ORDER[idx - 1]
    return "invalid", None


def _prompt_finger_action(finger: str):
    print(f"\n当前手指: {FINGER_LABELS[finger]} ({finger})")
    print("  r. 开始一段采集（按回车停止）")
    print("  o. 重采覆盖当前手指（清空该手指历史）")
    print("  f. 拟合并保存边界")
    print("  b. 返回手指选择")
    print("  x. 退出")
    raw = input("输入选项: ").strip().lower()
    if raw in ("r", "record", ""):
        return "record"
    if raw in ("o", "overwrite"):
        return "overwrite"
    if raw in ("f", "fit"):
        return "fit"
    if raw in ("b", "back"):
        return "back"
    if raw in ("x", "exit", "quit"):
        return "exit"
    return "invalid"


def _build_default_output_dir(script_dir: Path):
    return script_dir / "motor_test_logs" / "workspace_boundaries"


def _parse_args():
    parser = argparse.ArgumentParser(description="Passive workspace boundary collector for Y-hand.")
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["left", "right"])
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument("--max-outward-expand", type=float, default=DEFAULT_MAX_OUTWARD_EXPAND)
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--disable-plot", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    hand = args.hand.strip().lower()
    cmd_topic, feedback_topic = get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic
    if args.feedback_topic:
        feedback_topic = args.feedback_topic

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_dir) if args.output_dir else _build_default_output_dir(script_dir)
    hand_dir = output_root / hand
    hand_dir.mkdir(parents=True, exist_ok=True)
    model_specs = build_model_specs(hand)
    samples_by_finger = {finger: [] for finger in FINGER_ORDER}
    updated_fingers = set()
    selected_finger = FINGER_ORDER[0]
    latest_fitted_models = None
    tension_currents = get_simultaneous_tension_currents(hand)

    rclpy.init()
    node = PassiveBoundaryNode(hand=hand, cmd_topic=cmd_topic,
                               feedback_topic=feedback_topic, tension_currents=tension_currents)
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()
    panel = RealtimePanel(model_specs, enabled=(not args.disable_plot))

    def _refresh_panel():
        projected_np = _to_numpy_projected(_project_samples(samples_by_finger, model_specs))
        panel.refresh(projected_np, selected_finger=selected_finger, fitted_models=latest_fitted_models)

    def _fit_and_save():
        nonlocal latest_fitted_models
        if not updated_fingers:
            print("没有新增或清空的手指样本，保留已有边界模型。")
            return
        projected_np = _to_numpy_projected(_project_samples(samples_by_finger, model_specs))
        fitted = _fit_all_models(projected_np, model_specs, float(args.max_outward_expand), tension_currents)
        out_obj = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "hand": hand,
            "sample_hz": float(args.sample_hz),
            "tension_current_by_motor": [float(v) for v in tension_currents],
            "tension_current_scope": "all_collection",
            "collection_mode": "per_finger_simultaneous",
            "four_finger_current_control": {
                "mode": "per_finger",
                "flexor_current_a": SIMULTANEOUS_FLEXOR_CURRENT,
                "extensor_current_a": SIMULTANEOUS_EXTENSOR_CURRENT,
                "side_current_a": SIMULTANEOUS_SIDE_CURRENT,
                "kp": 0.0,
                "kd": 0.0,
            },
            "max_outward_expand": float(args.max_outward_expand),
            "topic": {"cmd_topic": cmd_topic, "feedback_topic": feedback_topic},
            "combos": fitted,
        }
        model_path = hand_dir / "boundary_models.json"
        saved = _save_boundary_models(model_path, out_obj, updated_fingers)
        latest_fitted_models = saved["combos"]
        _save_run_index(hand_dir / "run_index.json", hand, samples_by_finger, model_specs,
                        updated_fingers=updated_fingers)
        print("本次更新手指: " + ", ".join(sorted(updated_fingers)) + "；其余手指模型保留。")
        updated_fingers.clear()
        saved_plots = panel.save_all_finger_snapshots(
            output_dir=hand_dir / "plots", projected_by_model=projected_np,
            selected_finger=selected_finger, fitted_models=latest_fitted_models,
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        _refresh_panel()
        print(f"已拟合并保存边界模型: {model_path}")
        if saved_plots:
            print("已保存边界图: " + ", ".join(str(path.name) for path in saved_plots))

    try:
        if not node.wait_feedback(timeout_sec=8.0):
            print("错误: 启动后未收到反馈数据。"
                  f"\n当前话题状态: {cmd_topic} pubs={node.count_publishers(cmd_topic)}, "
                  f"subs={node.count_subscribers(cmd_topic)}; "
                  f"{feedback_topic} pubs={node.count_publishers(feedback_topic)}, "
                  f"subs={node.count_subscribers(feedback_topic)}")
            return
        print(
            "\n进入整指同时采集模式：选择手指 -> r记录 -> 手动掰动 -> 回车停止。"
            "\n  食指/中指每段同时投影6个组合，无名指/小指每段同时投影4个组合。"
            f"\n  四指屈肌 {SIMULTANEOUS_FLEXOR_CURRENT:+.2f} A，伸肌 {SIMULTANEOUS_EXTENSOR_CURRENT:+.2f} A，"
            f"侧摆 {SIMULTANEOUS_SIDE_CURRENT:+.2f} A；持续纯电流控制，kp/kd=0。"
            "\n  拇指和腕部保持默认张紧电流；f拟合保存。"
            f"\n  输出目录：{hand_dir}")
        _refresh_panel()
        while True:
            action, finger = _prompt_finger()
            if action == "exit":
                break
            if action == "plot":
                _refresh_panel()
                continue
            if action == "fit":
                _fit_and_save()
                continue
            if action == "invalid":
                print("输入无效，请重试。")
                continue
            selected_finger = finger
            _refresh_panel()
            while True:
                sub = _prompt_finger_action(finger)
                if sub == "exit":
                    return
                if sub == "back":
                    break
                if sub == "invalid":
                    print("输入无效，请重试。")
                    continue
                if sub == "fit":
                    _fit_and_save()
                    continue
                if sub == "overwrite":
                    samples_by_finger[finger] = []
                    updated_fingers.add(finger)
                    _clear_finger_raw_data(hand_dir, finger)
                    _save_run_index(hand_dir / "run_index.json", hand, samples_by_finger, model_specs,
                                    updated_fingers={finger})
                    latest_fitted_models = None
                    print(f"已清空 {FINGER_LABELS[finger]} 的历史采样，可重新采集。")
                    _refresh_panel()
                    continue
                if sub == "record":
                    print(f"准备采集 {FINGER_LABELS[finger]}（该手指所有组合共用本段样本）。")
                    latest_fitted_models = None
                    seg, aborted = _record_one_segment(
                        node=node, sample_hz=float(args.sample_hz), preview_refresh_cb=_refresh_panel,
                        live_sample_cb=samples_by_finger[finger].append)
                    if aborted:
                        print("采集中收到中断，当前段已停止。")
                    if not seg:
                        print("当前段未记录到有效数据。")
                        continue
                    updated_fingers.add(finger)
                    raw_csv = hand_dir / "raw" / finger / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
                    _save_session_csv(raw_csv, seg)
                    _save_run_index(hand_dir / "run_index.json", hand, samples_by_finger, model_specs,
                                    updated_fingers={finger})
                    duration = float(seg[-1]["wall"]) - float(seg[0]["wall"]) if len(seg) >= 2 else 0.0
                    actual_hz = (len(seg) - 1) / duration if duration > 1e-9 else 0.0
                    print(f"已保存采样段: {raw_csv}\n本段点数={len(seg)}, 当前手指总点数={len(samples_by_finger[finger])}\n"
                          f"本段时长={duration:.2f}s, 实际采样率={actual_hz:.2f}Hz, 目标采样率={float(args.sample_hz):.2f}Hz")
                    _refresh_panel()
    except KeyboardInterrupt:
        print("收到 Ctrl+C，退出程序。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from DEVICES_MSG_IMPORT_ERROR
    main()
