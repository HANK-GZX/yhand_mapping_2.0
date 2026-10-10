import zmq
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import argparse
import time

'''
This is a minimal Python example that talks to our MANUS C++ SDK and prints the
data out directly to the terminal. To run this, first run our MANUS SDK and
then this script.

Keep in mind this data runs VERY fast, I would not recommend directly setting
robot hands at this rate but instead slow down the data.

-------------------------------------------------------------------------------
Dual-stream collaboration notes
-------------------------------------------------------------------------------
Design goal:
1) Preserve the original ROS topic `/glove_joint_data` for the ergonomics
   stream so existing nodes remain compatible.
2) Add a second ROS topic `/glove_joint_data_raw_skeleton` for the full raw
   skeleton stream.
3) Receive the two data types from two independent ZMQ ports to avoid stream
   collisions when CONFLATE is enabled.

Wire protocol expected from the modified Windows SDK:
- tcp://<sdk_ip>:18001 -> ergonomics packet, 48 floats
- tcp://<sdk_ip>:18002 -> raw skeleton packet
    single hand:  1 serial token + 25*7 pose values = 176 CSV tokens
    double hand:  352 CSV tokens

ROS publication policy:
- `/glove_joint_data`             : keeps legacy ergonomics / joint-level data
- `/glove_joint_data_raw_skeleton`: publishes raw skeleton numeric payload
-------------------------------------------------------------------------------
'''
# ========================= ROS 话题数据格式说明 =========================
#
# 当前系统通过两个 ROS2 话题发布手套数据：
#
# 1. /glove_joint_data
#    -> 发布 ergonomics + root quaternion
#    -> 固定长度：48 维
#
# 2. /glove_joint_data_raw_skeleton
#    -> 发布 raw skeleton
#    -> 单条消息固定长度：176 维
#    -> 每条消息只对应一只手
#
# ======================================================================
# 1) /glove_joint_data 话题格式
# ======================================================================
#
# /glove_joint_data 的 msg.data 总长度固定为 48
#
# 格式如下：
#   [0:4]   -> Left hand root quaternion   = [qx, qy, qz, qw]
#   [4:8]   -> Right hand root quaternion  = [qx, qy, qz, qw]
#   [8:28]  -> Left hand ergonomics        = 20 values
#   [28:48] -> Right hand ergonomics       = 20 values
#
# 即：
#   48 = 4 + 4 + 20 + 20
#
# ------------------------- 详细索引说明 -------------------------
#
# [0]  Left root qx
# [1]  Left root qy
# [2]  Left root qz
# [3]  Left root qw
#
# [4]  Right root qx
# [5]  Right root qy
# [6]  Right root qz
# [7]  Right root qw
#
# ------------------------- Left ergonomics 20维 -------------------------
#
# [8]   Left Thumb_CMC_Spread
# [9]   Left Thumb_CMC_Flex
# [10]  Left Thumb_PIP_Flex
# [11]  Left Thumb_DIP_Flex
#
# [12]  Left Index_MCP_Spread
# [13]  Left Index_MCP_Flex
# [14]  Left Index_PIP_Flex
# [15]  Left Index_DIP_Flex
#
# [16]  Left Middle_MCP_Spread
# [17]  Left Middle_MCP_Flex
# [18]  Left Middle_PIP_Flex
# [19]  Left Middle_DIP_Flex
#
# [20]  Left Ring_MCP_Spread
# [21]  Left Ring_MCP_Flex
# [22]  Left Ring_PIP_Flex
# [23]  Left Ring_DIP_Flex
#
# [24]  Left Pinky_MCP_Spread
# [25]  Left Pinky_MCP_Flex
# [26]  Left Pinky_PIP_Flex
# [27]  Left Pinky_DIP_Flex
#
# ------------------------- Right ergonomics 20维 ------------------------
#
# [28]  Right Thumb_CMC_Spread
# [29]  Right Thumb_CMC_Flex
# [30]  Right Thumb_PIP_Flex
# [31]  Right Thumb_DIP_Flex
#
# [32]  Right Index_MCP_Spread
# [33]  Right Index_MCP_Flex
# [34]  Right Index_PIP_Flex
# [35]  Right Index_DIP_Flex
#
# [36]  Right Middle_MCP_Spread
# [37]  Right Middle_MCP_Flex
# [38]  Right Middle_PIP_Flex
# [39]  Right Middle_DIP_Flex
#
# [40]  Right Ring_MCP_Spread
# [41]  Right Ring_MCP_Flex
# [42]  Right Ring_PIP_Flex
# [43]  Right Ring_DIP_Flex
#
# [44]  Right Pinky_MCP_Spread
# [45]  Right Pinky_MCP_Flex
# [46]  Right Pinky_PIP_Flex
# [47]  Right Pinky_DIP_Flex
#
# 说明：
# - /glove_joint_data 固定包含左右手
# - 即使某一只手没有有效数据，这个话题长度仍然保持 48
#
# ======================================================================
# 2) /glove_joint_data_raw_skeleton 话题格式
# ======================================================================
#
# /glove_joint_data_raw_skeleton 的 msg.data 单条消息固定长度为 176
#
# 注意：
# - 这里是一条消息对应“一只手”
# - 左右手如果都存在，会在同一个 topic 上分别发布两条 176维消息
# - 不是把左右手拼成一条 352维 ROS 消息
#
# 格式如下：
#   [0]     -> side code
#   [1:176] -> raw skeleton pose data = 25 * 7
#
# 即：
#   176 = 1 + 175
#
# ------------------------- side code 定义 -------------------------
#
# [0] side code:
#   0.0  -> Left hand
#   1.0  -> Right hand
#  -1.0  -> Unknown / unmatched serial
#
# ------------------------- pose 数据格式 -------------------------
#
# 从 [1] 开始，每 7 个数表示一个关键点：
#
#   [x, y, z, qx, qy, qz, qw]
#
# 含义：
#   x, y, z         -> 该关键点在3D空间中的位置
#   qx, qy, qz, qw -> 该关键点在3D空间中的四元数旋转
#
# 节点编号计算方式：
#   node i 的起始索引 = 1 + i * 7
#   node i 的结束索引 = 1 + (i + 1) * 7
#
# 即：
#   node i = msg.data[1 + i*7 : 1 + (i+1)*7]
#
# ------------------------- 25个节点定义 -------------------------
#
# node 0  = Wrist
# node 1  = Thumb Metacarpal
# node 2  = Thumb Proximal
# node 3  = Thumb Distal
# node 4  = Thumb Tip
#
# node 5  = Index Metacarpal
# node 6  = Index Proximal
# node 7  = Index Intermediate
# node 8  = Index Distal
# node 9  = Index Tip
#
# node 10 = Middle Metacarpal
# node 11 = Middle Proximal
# node 12 = Middle Intermediate
# node 13 = Middle Distal
# node 14 = Middle Tip
#
# node 15 = Ring Metacarpal
# node 16 = Ring Proximal
# node 17 = Ring Intermediate
# node 18 = Ring Distal
# node 19 = Ring Tip
#
# node 20 = Pinky Metacarpal
# node 21 = Pinky Proximal
# node 22 = Pinky Intermediate
# node 23 = Pinky Distal
# node 24 = Pinky Tip
#
# ------------------------- 176维逐段对应关系 -------------------------
#
# [0]
#   side code
#
# [1:8]
#   node 0  = Wrist                  = [x, y, z, qx, qy, qz, qw]
#
# [8:15]
#   node 1  = Thumb Metacarpal       = [x, y, z, qx, qy, qz, qw]
#
# [15:22]
#   node 2  = Thumb Proximal         = [x, y, z, qx, qy, qz, qw]
#
# [22:29]
#   node 3  = Thumb Distal           = [x, y, z, qx, qy, qz, qw]
#
# [29:36]
#   node 4  = Thumb Tip              = [x, y, z, qx, qy, qz, qw]
#
# [36:43]
#   node 5  = Index Metacarpal       = [x, y, z, qx, qy, qz, qw]
#
# [43:50]
#   node 6  = Index Proximal         = [x, y, z, qx, qy, qz, qw]
#
# [50:57]
#   node 7  = Index Intermediate     = [x, y, z, qx, qy, qz, qw]
#
# [57:64]
#   node 8  = Index Distal           = [x, y, z, qx, qy, qz, qw]
#
# [64:71]
#   node 9  = Index Tip              = [x, y, z, qx, qy, qz, qw]
#
# [71:78]
#   node 10 = Middle Metacarpal      = [x, y, z, qx, qy, qz, qw]
#
# [78:85]
#   node 11 = Middle Proximal        = [x, y, z, qx, qy, qz, qw]
#
# [85:92]
#   node 12 = Middle Intermediate    = [x, y, z, qx, qy, qz, qw]
#
# [92:99]
#   node 13 = Middle Distal          = [x, y, z, qx, qy, qz, qw]
#
# [99:106]
#   node 14 = Middle Tip             = [x, y, z, qx, qy, qz, qw]
#
# [106:113]
#   node 15 = Ring Metacarpal        = [x, y, z, qx, qy, qz, qw]
#
# [113:120]
#   node 16 = Ring Proximal          = [x, y, z, qx, qy, qz, qw]
#
# [120:127]
#   node 17 = Ring Intermediate      = [x, y, z, qx, qy, qz, qw]
#
# [127:134]
#   node 18 = Ring Distal            = [x, y, z, qx, qy, qz, qw]
#
# [134:141]
#   node 19 = Ring Tip               = [x, y, z, qx, qy, qz, qw]
#
# [141:148]
#   node 20 = Pinky Metacarpal       = [x, y, z, qx, qy, qz, qw]
#
# [148:155]
#   node 21 = Pinky Proximal         = [x, y, z, qx, qy, qz, qw]
#
# [155:162]
#   node 22 = Pinky Intermediate     = [x, y, z, qx, qy, qz, qw]
#
# [162:169]
#   node 23 = Pinky Distal           = [x, y, z, qx, qy, qz, qw]
#
# [169:176]
#   node 24 = Pinky Tip              = [x, y, z, qx, qy, qz, qw]
#
# ------------------------- 额外说明 -------------------------
#
# 1. /glove_joint_data_raw_skeleton 发布的是 pose，不是关节角度。
# 2. 如果只看到 175 个数，通常是打印时去掉了第0位 side/serial 以外的数据头。
# 3. 真正 ROS 消息里，这个 topic 的单条消息长度固定是 176。
# ======================================================================

#数采Y-Hand佩戴手套
# left_glove_sn = "77988769"
# right_glove_sn = "1742DBEA"

#新采购手套
left_glove_sn = "A633F1AE"
right_glove_sn = "3644911F"


class GlovePublisher(Node):
    def __init__(self, log_wrist_node0: bool = False):
        super().__init__('glove_publisher')
        self.log_wrist_node0 = bool(log_wrist_node0)
        self.timer_cb_count = 0
        self.ergo_recv_count = 0
        self.raw_recv_count = 0
        self.ergo_pub_count = 0
        self.raw_pub_count = 0
        self.rate_window_start_ts = time.monotonic()


        # ROS2 publishers
        # Keep the original topic name for ergonomics / joint-level data so
        # downstream subscribers do not need any change.
        self.publisher_ = self.create_publisher(Float64MultiArray, '/glove_joint_data', 10)

        # New topic for raw skeleton data. We keep the name similar to the
        # original one and append the required distinguishing suffix.
        self.raw_skeleton_publisher_ = self.create_publisher(
            Float64MultiArray, '/glove_joint_data_raw_skeleton', 10
        )

        # ZMQ connections to Manus SDK
        # Two sockets are used on purpose:
        # - ergonomics socket on 18001
        # - raw skeleton socket on 18002
        # This mirrors the Windows SDK design and avoids mixing packet types on
        # a single conflated socket.
        self.zmq_context = zmq.Context()

        print("Connecting to SDK ergonomics stream on tcp://192.168.110.125:18001")
        self.ergo_socket = self.zmq_context.socket(zmq.PULL)
        self.ergo_socket.setsockopt(zmq.CONFLATE, True)
        self.ergo_socket.connect("tcp://192.168.3.110:18001")

        print("Connecting to SDK raw skeleton stream on tcp://192.168.110.125:18002")
        self.raw_socket = self.zmq_context.socket(zmq.PULL)
        self.raw_socket.setsockopt(zmq.CONFLATE, True)
        self.raw_socket.connect("tcp://192.168.3.110:18002")

        # ROS2 timer for polling data
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    # Convert glove serial into a compact numeric side tag so the raw skeleton
    # topic can stay a Float64MultiArray without introducing a custom ROS msg.
    def _serial_to_side_code(self, serial_str: str) -> float:
        serial_upper = serial_str.upper()
        if serial_upper == left_glove_sn.upper():
            return 0.0
        if serial_upper == right_glove_sn.upper():
            return 1.0
        return -1.0

    # Convert one raw-skeleton CSV chunk into a purely numeric ROS payload.
    # The first field is the glove serial (string), which cannot be published in
    # Float64MultiArray, so we replace it with a side_code.
    def parse_full_skeleton_chunk(self, data_one_hand):
        '''
        data_one_hand format:
        [glove_serial, x, y, z, qx, qy, qz, qw, ...]
        Returns a purely numeric list:
        [side_code, x, y, z, qx, qy, qz, qw, ...]
        side_code:
            0.0 -> left glove
            1.0 -> right glove
           -1.0 -> unknown glove serial
        '''
        serial = str(data_one_hand[0]).upper()
        pose_values = list(map(float, data_one_hand[1:]))
        side_code = self._serial_to_side_code(serial)

        if serial not in (left_glove_sn.upper(), right_glove_sn.upper()):
            self.get_logger().warn(f"Skeleton Data - Serial Number not found: {serial}")

        return [side_code] + pose_values

    # Convert quaternion [qx, qy, qz, qw] to Euler angles [roll, pitch, yaw].
    # Rotation convention:
    # - roll  -> rotation around X axis
    # - pitch -> rotation around Y axis
    # - yaw   -> rotation around Z axis
    def _quat_to_euler_deg(self, qx, qy, qz, qw):
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-12:
            return float("nan"), float("nan"), float("nan")

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if sinp >= 1.0:
            pitch = math.pi / 2.0
        elif sinp <= -1.0:
            pitch = -math.pi / 2.0
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def _side_code_to_label(self, side_code: float) -> str:
        if abs(side_code - 0.0) < 0.5:
            return "left"
        if abs(side_code - 1.0) < 0.5:
            return "right"
        return "unknown"

    # Print node0 (wrist) raw pose + Euler angles in real time.
    def _log_wrist_node0(self, raw_chunk):
        if not self.log_wrist_node0:
            return
        if len(raw_chunk) < 8:
            return

        side_code = float(raw_chunk[0])
        side_label = self._side_code_to_label(side_code)
        x, y, z, qx, qy, qz, qw = [float(v) for v in raw_chunk[1:8]]
        roll_deg, pitch_deg, yaw_deg = self._quat_to_euler_deg(qx, qy, qz, qw)

        self.get_logger().info(
            "[raw_skeleton][node0_wrist] "
            f"side={side_label}({side_code:.1f}) "
            f"raw=[x={x:.6f}, y={y:.6f}, z={z:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}, qw={qw:.6f}] "
            f"euler_deg=[roll_x={roll_deg:.3f}, pitch_y={pitch_deg:.3f}, yaw_z={yaw_deg:.3f}]"
        )

    # Handle the legacy 48-float ergonomics packet.
    # Layout:
    #   left_root_quat(4) + right_root_quat(4) + left_ergo(20) + right_ergo(20)
    # We publish it unchanged to the original topic name.
    def handle_ergonomics_packet(self, data):
        if len(data) == 48:
            left_data = list(map(float, data[0:24]))
            right_data = list(map(float, data[24:48]))

            joint_data = left_data + right_data

            msg = Float64MultiArray()
            msg.data = joint_data
            self.publisher_.publish(msg)
            self.ergo_pub_count += 1
        else:
            self.get_logger().warn(f'Unexpected ergonomics packet length: {len(data)}')

    # Handle the raw skeleton packet from the dedicated 18002 stream.
    # Each node contributes 7 floats: [x, y, z, qx, qy, qz, qw].
    #
    # Current parser assumes 25 nodes / hand, which matches the current team
    # expectation and the sender implementation. If MANUS node count changes in
    # the future, update both the Windows sender and this parser together.
    def handle_raw_skeleton_packet(self, data):
        # One hand: 1 serial + 25*7 pose values = 176 tokens
        if len(data) == 176:
            self.raw_recv_count += 1
            raw_data = self.parse_full_skeleton_chunk(data[0:176])
            self._log_wrist_node0(raw_data)

            msg = Float64MultiArray()
            # Published payload format:
            #   [side_code, x, y, z, qx, qy, qz, qw, ...]
            msg.data = raw_data
            self.raw_skeleton_publisher_.publish(msg)
            self.raw_pub_count += 1

        # Two hands: 2 * 176 = 352 tokens
        elif len(data) == 352:
            self.raw_recv_count += 1
            raw_left_or_first = self.parse_full_skeleton_chunk(data[0:176])
            raw_right_or_second = self.parse_full_skeleton_chunk(data[176:352])
            self._log_wrist_node0(raw_left_or_first)
            self._log_wrist_node0(raw_right_or_second)

            msg = Float64MultiArray()
            # For two hands we concatenate the two serialized chunks into one
            # Float64MultiArray message on the raw skeleton topic.
            msg.data = raw_left_or_first + raw_right_or_second
            self.raw_skeleton_publisher_.publish(msg)
            self.raw_pub_count += 1

        else:
            self.get_logger().warn(f'Unexpected raw skeleton packet length: {len(data)}')

    # Poll both streams independently on every timer tick.
    # Non-blocking polling keeps one stream from stalling the other.
    def timer_callback(self):
        try:
            self.timer_cb_count += 1
            # Poll ergonomics stream
            if self.ergo_socket.poll(timeout=0):
                ergo_message = self.ergo_socket.recv()
                ergo_message = ergo_message.decode('utf-8')
                ergo_data = ergo_message.split(',')
                self.ergo_recv_count += 1

                self.handle_ergonomics_packet(ergo_data)

            # Poll raw skeleton stream
            if self.raw_socket.poll(timeout=0):
                raw_message = self.raw_socket.recv()
                raw_message = raw_message.decode('utf-8')
                raw_data = raw_message.split(',')

                self.handle_raw_skeleton_packet(raw_data)

            self._maybe_log_rates()

        except Exception as e:
            self.get_logger().warn(f'Error receiving glove data: {str(e)}')

    def _maybe_log_rates(self):
        now = time.monotonic()
        dt = now - self.rate_window_start_ts
        if dt < 1.0:
            return
        timer_hz = self.timer_cb_count / max(dt, 1e-6)
        ergo_recv_hz = self.ergo_recv_count / max(dt, 1e-6)
        raw_recv_hz = self.raw_recv_count / max(dt, 1e-6)
        ergo_pub_hz = self.ergo_pub_count / max(dt, 1e-6)
        raw_pub_hz = self.raw_pub_count / max(dt, 1e-6)
        self.get_logger().info(
            "[glove_pub_rate] "
            f"timer_hz={timer_hz:.1f}, "
            f"ergo_recv_hz={ergo_recv_hz:.1f}, ergo_pub_hz={ergo_pub_hz:.1f}, "
            f"raw_recv_hz={raw_recv_hz:.1f}, raw_pub_hz={raw_pub_hz:.1f}, "
            f"log_wrist_node0={self.log_wrist_node0}"
        )
        self.timer_cb_count = 0
        self.ergo_recv_count = 0
        self.raw_recv_count = 0
        self.ergo_pub_count = 0
        self.raw_pub_count = 0
        self.rate_window_start_ts = now


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Publish MANUS ergonomics/raw-skeleton streams to ROS2 topics. "
            "Use --log-wrist-node0 to print node0 wrist pose/euler log."
        )
    )
    parser.add_argument(
        "--log-wrist-node0",
        action="store_true",
        help="Enable per-frame wrist(node0) raw pose + Euler log on /glove_joint_data_raw_skeleton stream.",
    )
    return parser.parse_known_args(argv)


def main(args=None):
    cli_args, ros_args = _parse_cli_args(args)
    rclpy.init(args=ros_args)
    node = GlovePublisher(log_wrist_node0=bool(cli_args.log_wrist_node0))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
