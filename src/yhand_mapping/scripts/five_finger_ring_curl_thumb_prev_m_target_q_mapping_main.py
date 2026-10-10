#!/usr/bin/env python3
import argparse
import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import curl_m0_m2_mlp_mapping_main as curl_mod
import thumb_prev_m_target_q_mlp_mapping_main as thumb_mod

# Backward-compatible alias for callers that still import the non-thumb helpers as `ring_mod`.
ring_mod = curl_mod


DEFAULT_GLOVE_TOPIC = thumb_mod.DEFAULT_GLOVE_TOPIC
CONTROL_PERIOD_SEC = thumb_mod.CONTROL_PERIOD_SEC
LEFT_HAND_GLOVE_SLICE = thumb_mod.LEFT_HAND_GLOVE_SLICE
RIGHT_HAND_GLOVE_SLICE = thumb_mod.RIGHT_HAND_GLOVE_SLICE
MOTOR_COUNT = thumb_mod.MOTOR_COUNT
RAW_SKELETON_ONE_HAND_DIM = thumb_mod.RAW_SKELETON_ONE_HAND_DIM
RAW_SKELETON_TWO_HAND_DIM = thumb_mod.RAW_SKELETON_TWO_HAND_DIM
RAW_Q_STATUS_PERIOD_SEC = curl_mod.RAW_Q_STATUS_PERIOD_SEC

THUMB_FINGER = "thumb"
NON_THUMB_FINGERS = curl_mod.DEFAULT_FINGER_ORDER
FIVE_FINGER_Q_GROUPS = curl_mod.FIVE_FINGER_Q_GROUPS

MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT = dict(thumb_mod.MOTOR_FEEDBACK_LIMITS_LEFT)
MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT = dict(thumb_mod.MOTOR_FEEDBACK_LIMITS_RIGHT)
Q_OFFSET_MAP_DEFAULT = {
    "left": {i: 0.0 for i in range(20)},
    "right": {i: 0.0 for i in range(20)},
}
MOTOR_CMD_OFFSET_MAP_DEFAULT = {
    "left": {},
    "right": {},
}
RING_EXT_CMD_OFFSET_BY_HAND = {'left': {10: 20.0}, 'right': {24: 20.0}}
RING_SIDE_SWING_CMD_OFFSET_BY_HAND = {'left': {21: 20.0, 22: 20.0}, 'right': {7: 20.0, 8: 20.0}}


def _non_thumb_enable_default(raw):
    if raw is None:
        return "true"
    text = str(raw).strip()
    return text if text else "true"


def _build_thumb_spec(args, hand: str, script_dir: Path):
    build_args = args
    if hand == "left":
        right_defaults = thumb_mod.FINGER_DEFAULTS_BY_HAND["right"][THUMB_FINGER]
        thumb_enable_raw = str(getattr(args, "thumb_enable", "")).strip() or str(getattr(args, "enable_model", "")).strip()
        thumb_ckpt_raw = str(getattr(args, "thumb_model_ckpt", "")).strip() or str(getattr(args, "model_ckpt", "")).strip()
        thumb_motor_ids_raw = str(getattr(args, "thumb_model_motor_ids", "")).strip() or str(getattr(args, "model_motor_ids", "")).strip()
        thumb_q_indices_raw = str(getattr(args, "thumb_q_indices", "")).strip() or str(getattr(args, "q_indices", "")).strip()
        thumb_mirror_raw = (
            str(getattr(args, "thumb_mirror_right_from_left", "")).strip()
            or str(getattr(args, "mirror_right_from_left", "")).strip()
        )

        build_args = SimpleNamespace(**vars(args))
        if not thumb_enable_raw:
            build_args.thumb_enable = "true"
        if not thumb_ckpt_raw:
            build_args.thumb_model_ckpt = str(
                thumb_mod._default_model_ckpt_from_parts(script_dir, right_defaults["model_ckpt_rel_parts"])
            )
        if not thumb_motor_ids_raw:
            build_args.thumb_model_motor_ids = str(right_defaults["motor_ids_csv"])
        if not thumb_q_indices_raw:
            build_args.thumb_q_indices = str(right_defaults["q_indices_csv"])
        if not thumb_mirror_raw:
            build_args.thumb_mirror_right_from_left = "true"

    specs = thumb_mod._build_finger_specs(args=build_args, hand=hand, script_dir=script_dir)
    return specs[THUMB_FINGER]


def _build_non_thumb_specs(args, script_dir: Path):
    build_args = SimpleNamespace(
        finger="ring",
        model_ckpt="",
        curl_meta="",
        model_motor_ids="",
        q_indices="",
        mirror_left_from_right="true",
    )
    for finger in NON_THUMB_FINGERS:
        setattr(build_args, f"enable_{finger}", _non_thumb_enable_default(getattr(args, f"enable_{finger}", None)))
        setattr(build_args, f"{finger}_model_ckpt", getattr(args, f"{finger}_model_ckpt", None))
        setattr(build_args, f"{finger}_curl_meta", getattr(args, f"{finger}_curl_meta", None))
        setattr(build_args, f"{finger}_model_motor_ids", getattr(args, f"{finger}_model_motor_ids", None))
        setattr(build_args, f"{finger}_q_indices", getattr(args, f"{finger}_q_indices", None))
        setattr(
            build_args,
            f"{finger}_mirror_left_from_right",
            getattr(args, f"{finger}_mirror_left_from_right", None),
        )
    return curl_mod._build_finger_specs(args=build_args, script_dir=script_dir)


class FiveFingerHybridMappingNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        glove_topic: str,
        glove_raw_skeleton_topic: str,
        non_thumb_q_from_raw_skeleton: bool,
        defaults_hand,
        thumb_spec,
        non_thumb_specs,
        device: str,
        thumb_motor_feedback_limits_left,
        thumb_motor_feedback_limits_right,
        q_offset_map,
        control_period_sec: float,
        status_period_sec: float,
        glove_max_hz: float,
        glove_sub_queue_size: int,
        glove_pending_maxlen: int,
    ):
        super().__init__(f"five_finger_hybrid_mapping_{hand}")
        if thumb_mod.YHandMsg is None or thumb_mod.Motor is None:
            raise RuntimeError(f"Cannot import devices_pkg.msg: {thumb_mod.DEVICES_MSG_IMPORT_ERROR}")
        if thumb_mod.torch is None:
            raise RuntimeError(
                "Cannot import torch. Please install torch in the same Python used by launch.\n"
                f"Import error: {thumb_mod.TORCH_IMPORT_ERROR}"
            )

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.glove_topic = glove_topic
        self.glove_raw_skeleton_topic = glove_raw_skeleton_topic
        self.use_non_thumb_raw_skeleton_q = bool(non_thumb_q_from_raw_skeleton)
        if self.use_non_thumb_raw_skeleton_q and (not callable(curl_mod.compute_non_thumb_q16)):
            raise RuntimeError(
                "Cannot import replace_right_non_thumb_q_from_skeleton.py while "
                "--right-non-thumb-q-from-raw-skeleton is enabled.\n"
                f"Import error: {curl_mod.RIGHT_NON_THUMB_Q_IMPORT_ERROR}"
            )

        self.defaults_hand = defaults_hand
        self.device = curl_mod._resolve_device(device)
        self.hand_glove_slice = LEFT_HAND_GLOVE_SLICE if hand == "left" else RIGHT_HAND_GLOVE_SLICE
        self.q_offsets_this_hand = dict(q_offset_map.get(hand, {}))
        self.thumb_motor_pos_limits, self.thumb_motor_pos_limit_mode = thumb_mod._build_motor_pos_limit_map(
            hand=self.hand,
            limits_left=thumb_motor_feedback_limits_left,
            limits_right=thumb_motor_feedback_limits_right,
        )
        self.non_thumb_motor_pos_limits = (
            curl_mod.MOTOR_FEEDBACK_LIMITS_LEFT if self.hand == "left" else curl_mod.MOTOR_FEEDBACK_LIMITS_RIGHT
        )

        self.control_period_sec = float(control_period_sec)
        self.status_period_sec = float(status_period_sec)
        self.glove_max_hz = float(glove_max_hz)
        self.glove_min_interval_sec = (1.0 / self.glove_max_hz) if self.glove_max_hz > 0.0 else 0.0
        self.glove_sub_queue_size = max(1, int(glove_sub_queue_size))
        self.glove_pending_maxlen = max(1, int(glove_pending_maxlen))

        self.thumb_model = None
        if thumb_spec.get("enabled", False):
            loader = SimpleNamespace(hand=self.hand, device=self.device)
            self.thumb_model = thumb_mod.PrevMTargetQMLPMappingNode._load_finger_model(
                loader,
                name=THUMB_FINGER,
                spec=thumb_spec,
            )

        self.non_thumb_models = {}
        for finger in NON_THUMB_FINGERS:
            spec = non_thumb_specs.get(finger, {"enabled": False})
            if not spec.get("enabled", False):
                continue
            loader = SimpleNamespace(hand=self.hand, device=self.device)
            self.non_thumb_models[finger] = curl_mod.MultiFingerCurlMultiheadMLPMappingNode._load_finger_model(
                loader,
                finger=finger,
                spec=spec,
            )

        if self.thumb_model is None and not self.non_thumb_models:
            raise RuntimeError("No finger model enabled. Enable thumb and/or at least one non-thumb finger.")

        used_motors = {}
        if self.thumb_model is not None:
            for motor_id in self.thumb_model["motor_ids"]:
                used_motors[int(motor_id)] = THUMB_FINGER
        for finger, bundle in self.non_thumb_models.items():
            for motor_id in bundle["controlled_motor_ids"]:
                prev = used_motors.get(int(motor_id))
                if prev is not None:
                    self.get_logger().warn(
                        f"Motor overlap detected: motor {motor_id} is used by both '{prev}' and '{finger}'. "
                        "Later finger in order will overwrite previous command."
                    )
                used_motors[int(motor_id)] = finger
        self.controlled_motor_ids = sorted(used_motors.keys())

        self.publisher = self.create_publisher(thumb_mod.YHandMsg, self.cmd_topic, 10)
        self.feedback_sub = self.create_subscription(thumb_mod.YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.glove_sub = self.create_subscription(
            Float64MultiArray,
            self.glove_topic,
            self._glove_cb,
            self.glove_sub_queue_size,
        )
        self.glove_raw_sub = None
        if self.use_non_thumb_raw_skeleton_q and self.non_thumb_models:
            self.glove_raw_sub = self.create_subscription(
                Float64MultiArray,
                self.glove_raw_skeleton_topic,
                self._glove_raw_cb,
                self.glove_sub_queue_size,
            )

        self.publish_timer = self.create_timer(self.control_period_sec, self._publish_cb)
        self.status_timer = self.create_timer(self.status_period_sec, self._status_cb)
        self.raw_q_status_timer = None
        if self.use_non_thumb_raw_skeleton_q and self.non_thumb_models:
            self.raw_q_status_timer = self.create_timer(float(RAW_Q_STATUS_PERIOD_SEC), self._raw_q_status_cb)

        self.latest_glove20 = None
        self.latest_raw_q_by_finger = {}
        self.last_glove_ts = 0.0
        self.last_glove_accept_ts = 0.0
        self.last_glove_raw_ts = 0.0
        self.last_glove_raw_accept_ts = 0.0
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.latest_feedback_matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        self.tor_idle_latched = np.zeros((MOTOR_COUNT,), dtype=np.bool_)
        self.glove_rx_total = 0
        self.glove_accept_total = 0
        self.glove_drop_rate_limited = 0
        self.glove_raw_rx_total = 0
        self.glove_raw_accept_total = 0
        self.glove_raw_drop_rate_limited = 0
        self.glove_pending = deque(maxlen=self.glove_pending_maxlen)

        self.last_target_matrix = self.build_idle_tor_matrix()
        self.last_cmd_matrix = self._copy_matrix(self.last_target_matrix)
        self.clip_total = 0
        self.clip_last = 0
        self.infer_last_ms = 0.0
        self.infer_ema_ms = 0.0
        self._raw_side_mismatch_warned = False

        self.get_logger().info(
            "[five_finger_hybrid_mapping] started "
            f"hand={self.hand}, cmd={self.cmd_topic}, feedback={self.feedback_topic}, glove={self.glove_topic}, "
            f"raw_topic={self.glove_raw_skeleton_topic if (self.use_non_thumb_raw_skeleton_q and self.non_thumb_models) else 'disabled'}, "
            f"device={self.device}, thumb_enabled={self.thumb_model is not None}, "
            f"non_thumb_enabled={list(self.non_thumb_models.keys())}, controlled_motor_ids={self.controlled_motor_ids}"
        )
        if self.thumb_model is not None:
            self.get_logger().info(
                "[five_finger_hybrid_mapping] "
                f"thumb_ckpt={self.thumb_model['ckpt']}, thumb_motor_ids={self.thumb_model['motor_ids']}, "
                f"thumb_q_indices={self.thumb_model['q_indices']}, thumb_input_cols={self.thumb_model['input_cols']}, "
                f"thumb_target_m_cols={self.thumb_model['target_m_cols']}"
            )
        for finger in NON_THUMB_FINGERS:
            if finger not in self.non_thumb_models:
                continue
            bundle = self.non_thumb_models[finger]
            self.get_logger().info(
                "[five_finger_hybrid_mapping] "
                f"finger={finger}, ckpt={bundle['ckpt_path']}, curl_meta={bundle['curl_meta_path']}, "
                f"q_indices={bundle['q_indices']}, model_motor_ids={bundle['model_motor_ids']}, "
                f"controlled_motor_ids={bundle['controlled_motor_ids']}, "
                f"mirror_left_from_right={bundle['mirror_left_from_right']}"
            )

    def build_idle_tor_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    0.0,
                    float(self.defaults_hand["tor"][i]),
                    0.0,
                    0.0,
                ]
            )
        return matrix

    @staticmethod
    def _copy_matrix(matrix):
        return [[float(v) for v in row] for row in matrix]

    def _feedback_cb(self, msg):
        self.feedback_received = True
        self.last_feedback_ts = time.monotonic()
        matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        limit = min(len(motors), MOTOR_COUNT)
        for i in range(limit):
            motor = motors[i]
            matrix[i, 0] = float(motor.pos)
            matrix[i, 1] = float(motor.vel)
            matrix[i, 2] = float(motor.tor)
            matrix[i, 3] = float(motor.kp)
            matrix[i, 4] = float(motor.kd)
        self.latest_feedback_matrix = matrix

    def _glove_cb(self, msg):
        self.glove_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_accept_ts) < self.glove_min_interval_sec:
            self.glove_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        enabled_count = int(self.thumb_model is not None) + len(self.non_thumb_models)
        if len(data) == 4 and enabled_count == 1:
            glove20 = np.zeros((20,), dtype=np.float32)
            if self.thumb_model is not None:
                q_indices = self.thumb_model["q_indices"]
            else:
                q_indices = next(iter(self.non_thumb_models.values()))["q_indices"]
            for value, q_index in zip(data, q_indices):
                glove20[int(q_index)] = float(value)
        elif len(data) == 20:
            glove20 = np.asarray(data, dtype=np.float32)
        elif len(data) >= 48:
            glove20 = np.asarray(data[self.hand_glove_slice], dtype=np.float32)
        else:
            return

        if glove20.shape[0] != 20:
            return

        self.latest_glove20 = glove20
        self.glove_pending.append(glove20)
        self.last_glove_ts = now
        self.last_glove_accept_ts = now
        self.glove_accept_total += 1

    def _extract_raw_skeleton_chunk(self, data):
        target_code = 1.0 if self.hand == "right" else 0.0
        if len(data) == RAW_SKELETON_ONE_HAND_DIM:
            try:
                side_code = float(data[0])
            except Exception:
                return None
            if abs(side_code - target_code) < 0.5:
                return data
            return None
        if len(data) < RAW_SKELETON_TWO_HAND_DIM:
            return None

        chunks = [
            data[0:RAW_SKELETON_ONE_HAND_DIM],
            data[RAW_SKELETON_ONE_HAND_DIM:RAW_SKELETON_TWO_HAND_DIM],
        ]
        for chunk in chunks:
            try:
                side_code = float(chunk[0])
            except Exception:
                continue
            if abs(side_code - target_code) < 0.5:
                return chunk
        return None

    def _glove_raw_cb(self, msg):
        self.glove_raw_rx_total += 1
        now = time.monotonic()
        if self.glove_min_interval_sec > 0.0 and (now - self.last_glove_raw_accept_ts) < self.glove_min_interval_sec:
            self.glove_raw_drop_rate_limited += 1
            return

        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        chunk = self._extract_raw_skeleton_chunk(data)
        if chunk is None:
            expected_side = "right" if self.hand == "right" else "left"
            if (not self._raw_side_mismatch_warned) and len(data) in (
                RAW_SKELETON_ONE_HAND_DIM,
                RAW_SKELETON_TWO_HAND_DIM,
            ):
                self.get_logger().warn(
                    f"[five_finger_hybrid_mapping] ignore raw skeleton packet because it does not contain the expected {expected_side}-hand chunk."
                )
                self._raw_side_mismatch_warned = True
            return
        self._raw_side_mismatch_warned = False

        try:
            q16_map = curl_mod.compute_non_thumb_q16(chunk)
            raw_by_finger = {}
            for finger in self.non_thumb_models.keys():
                q_vals = []
                for i in range(4):
                    key = f"{finger}_q_{i}"
                    if key not in q16_map:
                        raise KeyError(f"Missing key from compute_non_thumb_q16: {key}")
                    q_vals.append(float(q16_map[key]))
                raw_by_finger[finger] = np.asarray(q_vals, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(
                f"[five_finger_hybrid_mapping] failed to compute {self.hand} non-thumb q from raw skeleton: {exc}"
            )
            return

        self.latest_raw_q_by_finger = raw_by_finger
        self.last_glove_raw_ts = now
        self.last_glove_raw_accept_ts = now
        self.glove_raw_accept_total += 1

    def _try_get_raw_q(self, finger: str):
        raw_q = self.latest_raw_q_by_finger.get(finger)
        if raw_q is None:
            return None
        return np.asarray(raw_q, dtype=np.float32)

    def _build_thumb_cmd(self, matrix, glove20: np.ndarray):
        if self.thumb_model is None:
            return 0

        clip_last = 0
        bundle = self.thumb_model
        history_m = thumb_mod.PrevMTargetQMLPMappingNode._build_history_m_from_feedback(self, bundle["motor_ids"])
        target_q = thumb_mod.PrevMTargetQMLPMappingNode._build_target_q_from_glove20(self, glove20, bundle["q_indices"])
        x = thumb_mod.PrevMTargetQMLPMappingNode._build_model_input(self, bundle, history_m, target_q)
        x_norm = bundle["x_scaler"].transform(x)

        with thumb_mod.torch.no_grad():
            xb = thumb_mod.torch.tensor(x_norm, dtype=thumb_mod.torch.float32, device=self.device)
            pred_m_norm = bundle["model"](xb).cpu().numpy()
        pred_m = bundle["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)

        cmd_pos_by_model_motor_idx = np.zeros((bundle["m_dim"],), dtype=np.float32)
        for out_col_idx, (target_idx, _target_col) in enumerate(bundle["target_plan"]):
            motor_id = int(bundle["motor_ids"][int(target_idx)])
            idx = motor_id - 1
            pred_pos = float(pred_m[out_col_idx])
            lim = self.thumb_motor_pos_limits.get(motor_id)
            if lim is not None:
                pred_pos_clipped = float(np.clip(pred_pos, lim[0], lim[1]))
                if abs(pred_pos_clipped - pred_pos) > 1e-6:
                    clip_last += 1
                pred_pos = pred_pos_clipped
            matrix[idx] = [
                pred_pos,
                float(self.defaults_hand["vel"][idx]),
                float(self.defaults_hand["tor_active"][idx]),
                float(self.defaults_hand["kp"][idx]),
                float(self.defaults_hand["kd"][idx]),
            ]
            cmd_pos_by_model_motor_idx[int(target_idx)] = float(pred_pos)

        bundle["last_history_m"] = history_m.astype(np.float32)
        bundle["last_target_q"] = target_q.astype(np.float32)
        bundle["last_pred_m"] = pred_m.astype(np.float32)
        bundle["last_cmd_pos"] = cmd_pos_by_model_motor_idx.astype(np.float32)
        return int(clip_last)

    def _build_non_thumb_cmd(self, matrix, glove20: np.ndarray):
        clip_last = 0
        for finger in NON_THUMB_FINGERS:
            if finger not in self.non_thumb_models:
                continue
            bundle = self.non_thumb_models[finger]
            if self.use_non_thumb_raw_skeleton_q:
                raw_q = self._try_get_raw_q(finger)
                if raw_q is None:
                    if not bundle["raw_not_ready_warned"]:
                        self.get_logger().warn(
                            f"[five_finger_hybrid_mapping] {self.hand} {finger} raw skeleton q is not ready yet; "
                            "hold previous command for this finger this frame."
                        )
                        bundle["raw_not_ready_warned"] = True
                    self._hold_last_target_for_controlled_motors(matrix, bundle["controlled_motor_ids"])
                    continue
                bundle["raw_not_ready_warned"] = False
            else:
                raw_q = curl_mod.MultiFingerCurlMultiheadMLPMappingNode._build_raw_q_from_glove20(
                    self,
                    glove20,
                    bundle["q_indices"],
                )

            q0 = float(raw_q[0])
            if self.hand == "left" and bundle["mirror_left_from_right"]:
                q0 = -q0
            curl, q_cal = curl_mod.compute_curl(raw_q[1], raw_q[2], raw_q[3], bundle["curl_meta"])
            model_bundle = bundle["model_bundle"]
            if model_bundle["model_kind"] == "q0_curl_multihead":
                x_curl = model_bundle["scalers"]["x_curl"].transform(np.asarray([[curl]], dtype=np.float32))
                x_q0curl = model_bundle["scalers"]["x_q0curl"].transform(
                    np.asarray([[q0, curl]], dtype=np.float32)
                )
                with thumb_mod.torch.no_grad():
                    xb_curl = thumb_mod.torch.tensor(x_curl, dtype=thumb_mod.torch.float32, device=self.device)
                    xb_q0curl = thumb_mod.torch.tensor(x_q0curl, dtype=thumb_mod.torch.float32, device=self.device)
                    pred_curl_norm, pred_q0curl_norm = model_bundle["model"](xb_curl, xb_q0curl)
                pred_curl = model_bundle["scalers"][model_bundle["curl_output_scaler_name"]].inverse(
                    pred_curl_norm.cpu().numpy()
                )[0]
                pred_q0curl = model_bundle["scalers"][model_bundle["q0curl_output_scaler_name"]].inverse(
                    pred_q0curl_norm.cpu().numpy()
                )[0]
                pred_by_col = {}
                for col, value in zip(model_bundle["curl_head_target_m_cols"], pred_curl):
                    pred_by_col[str(col)] = float(value)
                for col, value in zip(model_bundle["q0curl_head_target_m_cols"], pred_q0curl):
                    pred_by_col[str(col)] = float(value)
                pred_m = np.asarray(
                    [float(pred_by_col[col]) for col in model_bundle["target_m_cols"]],
                    dtype=np.float32,
                )
            else:
                x = np.asarray([[curl]], dtype=np.float32)
                x_norm = model_bundle["x_scaler"].transform(x)
                with thumb_mod.torch.no_grad():
                    xb = thumb_mod.torch.tensor(x_norm, dtype=thumb_mod.torch.float32, device=self.device)
                    pred_m_norm = model_bundle["model"](xb).cpu().numpy()
                pred_m = model_bundle["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)

            cmd_pos = []
            for out_col_idx, (target_idx, _target_col) in enumerate(model_bundle["target_plan"]):
                motor_id = int(bundle["model_motor_ids"][int(target_idx)])
                idx = motor_id - 1
                pred_pos = float(pred_m[out_col_idx])
                if finger == "ring":
                    pred_pos += float(RING_EXT_CMD_OFFSET_BY_HAND.get(self.hand, {}).get(motor_id, 0.0))
                    pred_pos += float(RING_SIDE_SWING_CMD_OFFSET_BY_HAND.get(self.hand, {}).get(motor_id, 0.0))
                prev_cmd_pos = float(self.last_target_matrix[idx][0])
                if abs(pred_pos - prev_cmd_pos) <= curl_mod.CMD_POSITION_HOLD_DEADBAND_DEG:
                    pred_pos = prev_cmd_pos
                lim = self.non_thumb_motor_pos_limits.get(motor_id)
                if lim is not None:
                    pred_pos_clipped = float(np.clip(pred_pos, lim[0], lim[1]))
                    if abs(pred_pos_clipped - pred_pos) > 1e-6:
                        clip_last += 1
                    pred_pos = pred_pos_clipped

                matrix[idx] = [
                    pred_pos,
                    float(self.defaults_hand["vel"][idx]),
                    float(self.defaults_hand["tor_active"][idx]),
                    float(self.defaults_hand["kp"][idx]),
                    float(self.defaults_hand["kd"][idx]),
                ]
                cmd_pos.append(float(pred_pos))

            bundle["last_raw_q"] = raw_q.astype(np.float32)
            bundle["last_q_cal"] = np.asarray(q_cal, dtype=np.float32)
            bundle["last_curl"] = float(curl)
            bundle["last_pred_m"] = pred_m.astype(np.float32)
            bundle["last_cmd_pos"] = np.asarray(cmd_pos, dtype=np.float32)
        return int(clip_last)

    def _build_cmd_from_glove20(self, glove20: np.ndarray):
        matrix = self.build_idle_tor_matrix()
        infer_t0 = time.monotonic()
        clip_last = 0
        clip_last += self._build_thumb_cmd(matrix, glove20)
        clip_last += self._build_non_thumb_cmd(matrix, glove20)
        infer_ms = (time.monotonic() - infer_t0) * 1000.0
        self.infer_last_ms = infer_ms
        self.infer_ema_ms = infer_ms if self.infer_ema_ms <= 0.0 else (0.9 * self.infer_ema_ms + 0.1 * infer_ms)
        self.clip_last = int(clip_last)
        self.clip_total += int(clip_last)
        return matrix

    def _publish_matrix(self, matrix):
        return thumb_mod.PrevMTargetQMLPMappingNode._publish_matrix(self, matrix)

    def publish_idle_once(self):
        return thumb_mod.PrevMTargetQMLPMappingNode.publish_idle_once(self)

    def _hold_last_target_for_controlled_motors(self, matrix, controlled_motor_ids):
        for mid in controlled_motor_ids:
            idx = int(mid) - 1
            prev = self.last_target_matrix[idx]
            matrix[idx] = [
                float(prev[0]),
                float(prev[1]),
                float(prev[2]),
                float(prev[3]),
                float(prev[4]),
            ]

    def _apply_directional_tor_active(self, matrix):
        for idx, row in enumerate(matrix):
            kp = float(row[3])
            kd = float(row[4])
            if abs(kp) <= 1e-9 and abs(kd) <= 1e-9:
                self.tor_idle_latched[idx] = False
                continue

            motor_id = idx + 1
            if curl_mod._is_side_swing_motor_id(motor_id):
                enter_deadband_deg = float(curl_mod.SIDE_SWING_TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(curl_mod.SIDE_SWING_TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)
            else:
                enter_deadband_deg = float(curl_mod.TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(curl_mod.TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)

            current_pos = float(self.latest_feedback_matrix[idx, 0])
            if not np.isfinite(current_pos):
                continue

            position_error = float(row[0]) - current_pos
            abs_error = abs(position_error)
            if bool(self.tor_idle_latched[idx]):
                if abs_error > exit_deadband_deg:
                    self.tor_idle_latched[idx] = False
                else:
                    if curl_mod._is_side_swing_motor_id(motor_id):
                        row[1] = 0.0
                        row[2] = float(self.defaults_hand["tor"][idx])
                    else:
                        row[:] = [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    continue

            if abs_error <= enter_deadband_deg:
                self.tor_idle_latched[idx] = True
                if curl_mod._is_side_swing_motor_id(motor_id):
                    row[1] = 0.0
                    row[2] = float(self.defaults_hand["tor"][idx])
                else:
                    row[:] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                continue

            row[2] = curl_mod._directional_tor_active(
                self.defaults_hand["tor_active"][idx],
                self.defaults_hand["tor"][idx],
                row[0],
                current_pos,
            )
        return matrix

    def _publish_cb(self):
        if not self.feedback_received:
            target_matrix = self.build_idle_tor_matrix()
        else:
            if len(self.glove_pending) > 0:
                glove20_last = self.glove_pending[-1]
                self.glove_pending.clear()
                target_matrix = self._build_cmd_from_glove20(glove20_last)
            elif self.latest_glove20 is not None:
                target_matrix = self._copy_matrix(self.last_target_matrix)
            else:
                target_matrix = self.build_idle_tor_matrix()
        self.last_target_matrix = self._copy_matrix(target_matrix)
        matrix = self._copy_matrix(target_matrix)
        matrix = self._apply_directional_tor_active(matrix)
        self.last_cmd_matrix = self._copy_matrix(matrix)
        self._publish_matrix(matrix)

    def _raw_q_ready_all(self):
        if not self.non_thumb_models:
            return self.latest_glove20 is not None
        if not self.use_non_thumb_raw_skeleton_q:
            return self.latest_glove20 is not None
        for finger in self.non_thumb_models.keys():
            if finger not in self.latest_raw_q_by_finger:
                return False
        return True

    def wait_ready(self, timeout_sec: float):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            glove_ready = self._raw_q_ready_all()
            if self.feedback_received and glove_ready:
                return True
        return False

    def _format_five_finger_angles(self):
        if self.latest_glove20 is None:
            return "None"
        out = {}
        for finger, idxs in FIVE_FINGER_Q_GROUPS.items():
            out[finger] = [round(float(self.latest_glove20[i]), 3) for i in idxs]
        return out

    def _format_active_motor_cmd(self):
        out = {}
        for motor_id in self.controlled_motor_ids:
            out[int(motor_id)] = round(float(self.last_cmd_matrix[int(motor_id) - 1][0]), 3)
        return out

    @staticmethod
    def _round_or_none(value):
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return round(value, 3)

    def _format_active_motor_diag(self):
        out = {}
        for motor_id in self.controlled_motor_ids:
            idx = int(motor_id) - 1
            target_pos = self._round_or_none(self.last_target_matrix[idx][0])
            published_pos = self._round_or_none(self.last_cmd_matrix[idx][0])
            feedback_pos = self._round_or_none(self.latest_feedback_matrix[idx, 0])
            torque_fb = self._round_or_none(self.latest_feedback_matrix[idx, 2])
            pure_tension = (
                abs(float(self.last_cmd_matrix[idx][3])) <= 1e-9
                and abs(float(self.last_cmd_matrix[idx][4])) <= 1e-9
            )
            out[int(motor_id)] = {
                "target_pos": target_pos,
                "published_pos": published_pos,
                "feedback_pos": feedback_pos,
                "torque": torque_fb,
                "pure_tension": bool(pure_tension),
            }
        return out

    def _raw_q_status_cb(self):
        if not (self.use_non_thumb_raw_skeleton_q and self.non_thumb_models):
            return
        now = time.monotonic()
        glove_raw_age = (now - self.last_glove_raw_ts) if self.last_glove_raw_ts > 0.0 else float("inf")
        chunks = []
        for finger in NON_THUMB_FINGERS:
            if finger not in self.non_thumb_models:
                continue
            vals = self.latest_raw_q_by_finger.get(finger)
            if vals is None:
                chunks.append(f"{finger}=not_ready")
                continue
            chunks.append(f"{finger}={[round(float(v), 3) for v in vals.tolist()]}")
        self.get_logger().info(
            "[five_finger_hybrid_mapping] "
            f"non_thumb_raw_q={'; '.join(chunks)}, glove_raw_age={glove_raw_age:.3f}s"
        )

    def _status_cb(self):
        now = time.monotonic()
        glove_age = (now - self.last_glove_ts) if self.last_glove_ts > 0.0 else float("inf")
        fb_age = (now - self.last_feedback_ts) if self.last_feedback_ts > 0.0 else float("inf")
        infer_load = (
            self.infer_ema_ms / (self.control_period_sec * 1000.0)
            if self.control_period_sec > 0.0
            else float("nan")
        )
        self.get_logger().info(
            "[five_finger_hybrid_mapping] "
            f"feedback_ready={self.feedback_received}, glove_ready={self.latest_glove20 is not None}, "
            f"raw_q_ready={self._raw_q_ready_all()}, fb_age={fb_age:.3f}s, glove_age={glove_age:.3f}s, "
            f"glove_rx={self.glove_rx_total}, glove_accept={self.glove_accept_total}, "
            f"glove_drop_rl={self.glove_drop_rate_limited}, clip_last={self.clip_last}, clip_total={self.clip_total}, "
            f"infer_last_ms={self.infer_last_ms:.3f}, infer_ema_ms={self.infer_ema_ms:.3f}, infer_load={infer_load:.2f}, "
            f"angles={self._format_five_finger_angles()}, active_diag={self._format_active_motor_diag()}"
        )


def _add_thumb_args(parser: argparse.ArgumentParser):
    parser.add_argument("--enable-model", default="", help="Legacy alias for --enable-thumb")
    parser.add_argument("--model-ckpt", default="", help="Legacy alias for --thumb-model-ckpt")
    parser.add_argument("--model-motor-ids", default="", help="Legacy alias for --thumb-model-motor-ids")
    parser.add_argument("--q-indices", default="", help="Legacy alias for --thumb-q-indices")
    parser.add_argument(
        "--mirror-right-from-left",
        default="",
        help="Legacy alias for --thumb-mirror-right-from-left",
    )
    parser.add_argument("--enable-thumb", dest="thumb_enable", default="")
    parser.add_argument("--thumb-model-ckpt", dest="thumb_model_ckpt", default="")
    parser.add_argument("--thumb-model-motor-ids", dest="thumb_model_motor_ids", default="")
    parser.add_argument("--thumb-q-indices", dest="thumb_q_indices", default="")
    parser.add_argument("--thumb-mirror-right-from-left", dest="thumb_mirror_right_from_left", default="")


def _add_non_thumb_args(parser: argparse.ArgumentParser):
    for finger in NON_THUMB_FINGERS:
        curl_mod._add_finger_cli_args(parser, finger)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Five-finger hybrid mapping: thumb uses prev_m_target_q MLP, non-thumb fingers use q0+curl MLP."
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--cmd-topic", default="", help="Override command topic")
    parser.add_argument("--feedback-topic", default="", help="Override feedback topic")
    parser.add_argument("--glove-topic", default=DEFAULT_GLOVE_TOPIC, help="48D glove topic")
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default="/glove_joint_data_raw_skeleton",
        help="176D raw-skeleton topic used by non-thumb q solving",
    )
    raw_non_thumb_group = parser.add_mutually_exclusive_group()
    raw_non_thumb_group.add_argument(
        "--right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_true",
        default=True,
        help="Solve enabled non-thumb q_0..q_3 from the selected hand raw skeleton (default: enabled).",
    )
    raw_non_thumb_group.add_argument(
        "--no-right-non-thumb-q-from-raw-skeleton",
        dest="right_non_thumb_q_from_raw_skeleton",
        action="store_false",
        help="Disable raw-skeleton q replacement and use glove q indices for non-thumb fingers.",
    )
    _add_thumb_args(parser)
    _add_non_thumb_args(parser)
    parser.add_argument(
        "--defaults-path",
        default=str(script_dir / "muscle_adjustment_defaults.py"),
        help="Path to muscle_adjustment_defaults.py",
    )
    parser.add_argument("--control-period-sec", type=float, default=float(CONTROL_PERIOD_SEC))
    parser.add_argument("--status-period-sec", type=float, default=2.0)
    parser.add_argument(
        "--glove-max-hz",
        type=float,
        default=100.0,
        help="Max accepted glove callback rate. <=0 disables rate limiting",
    )
    parser.add_argument("--glove-sub-queue-size", type=int, default=60)
    parser.add_argument("--glove-pending-maxlen", type=int, default=60)
    parser.add_argument(
        "--motor-feedback-limits-left-json",
        default=json.dumps(MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument(
        "--motor-feedback-limits-right-json",
        default=json.dumps(MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument(
        "--q-offset-map-json",
        default=json.dumps(Q_OFFSET_MAP_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--require-ready", action="store_true", help="Exit if startup is not ready before timeout")
    parser.add_argument("--ready-timeout-sec", type=float, default=12.0)
    return parser.parse_args()


def main():
    args = parse_args()
    hand = args.hand.strip().lower()
    cmd_topic, feedback_topic = thumb_mod.get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic.strip()
    if args.feedback_topic:
        feedback_topic = args.feedback_topic.strip()

    if not getattr(args, "thumb_enable", "").strip():
        args.thumb_enable = str(getattr(args, "enable_model", "")).strip()

    script_dir = Path(__file__).resolve().parent
    thumb_spec = _build_thumb_spec(args=args, hand=hand, script_dir=script_dir)
    non_thumb_specs = _build_non_thumb_specs(args=args, script_dir=script_dir)
    defaults_hand = thumb_mod._load_defaults_from_py(Path(args.defaults_path), hand)
    motor_feedback_limits_left = thumb_mod._parse_motor_feedback_limits_json(
        args.motor_feedback_limits_left_json,
        "--motor-feedback-limits-left-json",
    )
    motor_feedback_limits_right = thumb_mod._parse_motor_feedback_limits_json(
        args.motor_feedback_limits_right_json,
        "--motor-feedback-limits-right-json",
    )
    q_offset_map = thumb_mod._parse_q_offset_map_json(args.q_offset_map_json, "--q-offset-map-json")

    rclpy.init()
    node = None
    try:
        node = FiveFingerHybridMappingNode(
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            glove_topic=args.glove_topic.strip(),
            glove_raw_skeleton_topic=args.glove_raw_skeleton_topic.strip(),
            non_thumb_q_from_raw_skeleton=bool(args.right_non_thumb_q_from_raw_skeleton),
            defaults_hand=defaults_hand,
            thumb_spec=thumb_spec,
            non_thumb_specs=non_thumb_specs,
            device=args.device.strip().lower(),
            thumb_motor_feedback_limits_left=motor_feedback_limits_left,
            thumb_motor_feedback_limits_right=motor_feedback_limits_right,
            q_offset_map=q_offset_map,
            control_period_sec=float(args.control_period_sec),
            status_period_sec=float(args.status_period_sec),
            glove_max_hz=float(args.glove_max_hz),
            glove_sub_queue_size=int(args.glove_sub_queue_size),
            glove_pending_maxlen=int(args.glove_pending_maxlen),
        )

        if args.require_ready:
            ok = node.wait_ready(timeout_sec=float(args.ready_timeout_sec))
            if not ok:
                raise RuntimeError(
                    "Startup not ready within timeout: "
                    f"feedback_ready={node.feedback_received}, glove_ready={node.latest_glove20 is not None}, "
                    f"raw_q_ready={node._raw_q_ready_all()}"
                )

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[five_finger_hybrid_mapping] KeyboardInterrupt received, exiting...")
    finally:
        if node is not None:
            try:
                node.publish_idle_once()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
