from mocap_main_base import MCPBase
from mocap_api import MCPAvatar
import asyncio
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

TARGET_TF_PUBLISH_HZ = 100.0

# Define the parent-child relationship for each link
links_parent = {
    "Hips": "world",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightTiptoe": "RightFoot",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftTiptoe": "LeftFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",
    "Head1": "Head",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand",
    "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2",
    "RightInHandIndex": "RightHand",
    "RightHandIndex1": "RightInHandIndex",
    "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2",
    "RightInHandMiddle": "RightHand",
    "RightHandMiddle1": "RightInHandMiddle",
    "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2",
    "RightInHandRing": "RightHand",
    "RightHandRing1": "RightInHandRing",
    "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",
    "RightInHandPinky": "RightHand",
    "RightHandPinky1": "RightHandPinky",
    "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand",
    "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2",
    "LeftInHandIndex": "LeftHand",
    "LeftHandIndex1": "LeftInHandIndex",
    "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2",
    "LeftInHandMiddle": "LeftHand",
    "LeftHandMiddle1": "LeftInHandMiddle",
    "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftInHandRing": "LeftHand",
    "LeftHandRing1": "LeftInHandRing",
    "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",
    "LeftInHandPinky": "LeftHand",
    "LeftHandPinky1": "LeftHandPinky",
    "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2"
}

class MCPMainStickmanROS2(MCPBase):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = Node("real_time_transform_publisher")
        self.br = TransformBroadcaster(self.node)
        self.publish_period_sec = 1.0 / max(TARGET_TF_PUBLISH_HZ, 1e-6)
        self.last_publish_ts = 0.0
        self.avatar_cb_count = 0
        self.tf_pub_count = 0
        self.rate_window_start_ts = time.monotonic()

    def handleAvatar(self, avatar_handle):
        try:
            now = time.monotonic()
            self.avatar_cb_count += 1

            # Handle avatar update event
            avatar = MCPAvatar(avatar_handle)  # Get avatar data
            transforms = []

            # Get all joint data
            joints = avatar.get_joints()
            for joint in joints:
                link_name = joint.get_name()
                position = joint.get_local_position()
                rotation = joint.get_local_rotation()

                # Build transforms for the full body frame in one batch to reduce ROS message overhead.
                transform = self.build_transform(link_name, position, rotation)
                if transform is not None:
                    transforms.append(transform)

            # Head1 
            rotation = (0.0, 0.0, 0.0, 1.0) 
            position = (0.000000, 16.450001, 0.000000)
            transform = self.build_transform('Head1', position, rotation)
            if transform is not None:
                transforms.append(transform)

            # LeftTiptoe RightTiptoe
            position = (0.000000, -7.850000, 14.280000)
            transform = self.build_transform('LeftTiptoe', position, rotation)
            if transform is not None:
                transforms.append(transform)
            transform = self.build_transform('RightTiptoe', position, rotation)
            if transform is not None:
                transforms.append(transform)

            # Handle the root joint of the robot
            root_joint = avatar.get_root_joint()
            position = root_joint.get_local_position()
            rotation = root_joint.get_local_rotation()

            # Publish the root joint transform
            transform = self.build_transform('base_link', position, rotation, frame_id='world')
            if transform is not None:
                transforms.append(transform)

            if transforms:
                self.br.sendTransform(transforms)
                self.last_publish_ts = now
                self.tf_pub_count += 1
            self._maybe_log_rates(now)

        except Exception as e:
            print(f"Error handling avatar: {e}")

    def _maybe_log_rates(self, now):
        dt = now - self.rate_window_start_ts
        if dt < 1.0:
            return
        avatar_hz = self.avatar_cb_count / max(dt, 1e-6)
        tf_pub_hz = self.tf_pub_count / max(dt, 1e-6)
        print(
            f"[mocap_tf_pub] avatar_hz={avatar_hz:.1f}, "
            f"tf_pub_hz={tf_pub_hz:.1f}, target_hz={TARGET_TF_PUBLISH_HZ:.1f}, "
            "publish_mode=event_driven"
        )
        self.avatar_cb_count = 0
        self.tf_pub_count = 0
        self.rate_window_start_ts = now

    def build_transform(self, child_frame_id, position, rotation, frame_id=None):
        try:
            if frame_id is None:
                frame_id = links_parent[child_frame_id]

            t = TransformStamped()
            t.header.stamp = self.node.get_clock().now().to_msg()
            t.header.frame_id = frame_id
            t.child_frame_id = child_frame_id

            # Set translation
            t.transform.translation.x = position[2] / 100
            t.transform.translation.y = position[0] / 100
            t.transform.translation.z = position[1] / 100

            # Set rotation
            # mocap_api.get_local_rotation() returns quaternion in (w, x, y, z).
            # ROS Transform expects (x, y, z, w).
            t.transform.rotation.x = rotation[1]
            t.transform.rotation.y = rotation[2]
            t.transform.rotation.z = rotation[3]
            t.transform.rotation.w = rotation[0]

            return t
        except Exception as e:
            print(f"Error sending transform for {child_frame_id}: {e}")
            return None

    def main(self):
        asyncio.run(self.main_async())

if __name__ == '__main__':
    MCPMainStickmanROS2().main()
