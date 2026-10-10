import zmq
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

'''
This is a minimal Python example that talks to our MANUS C++ SDK and prints the data out directly to the terminal.
To run this, first run our MANUS SDK and then this script.

Keep in mind this data runs VERY fast, I would not recommend directly setting robot hands at this rate but instead slow down the data.
'''

left_glove_sn = "77988769"
right_glove_sn = "1742DBEA"

class GlovePublisher(Node):
    def __init__(self):
        super().__init__('glove_publisher')

        # ROS2 publisher
        self.publisher_ = self.create_publisher(Float64MultiArray, '/glove_joint_data', 10)

        # ZMQ connection to Manus SDK
        self.zmq_context = zmq.Context()
        print("Connecting to SDK")
        self.socket = self.zmq_context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect("tcp://192.168.1.17:18001")

        # ROS2 timer for polling data
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def parse_full_skeleton(self, data):
        if data[0] == left_glove_sn:
            print("Left Glove Skeleton Data")
            print(list(map(float,data[1:])))
        elif data[0] == right_glove_sn:
            print("Right Glove Skeleton Data")
            print(list(map(float,data[1:])))
        else:
            print("Serial Number not found: " + str(data[0]))

    def timer_callback(self):
        try:
            if self.socket.poll(timeout=1):  # Non-blocking receive
                message = self.socket.recv()
                message = message.decode('utf-8')
                data = message.split(",")   

                if len(data) == 48:
                    print("Left Glove Joint-level Ergonomics Data:")
                    left_data = list(map(float, data[0:24]))
                    print(left_data)
                    print("Right Glove Joint-level Ergonomics Data:")
                    right_data = list(map(float, data[24:48]))
                    print(right_data)

                    # Combine both hands
                    joint_data = list(map(float, data))  # 保留全部 48 个数

                    # ROS2 publish
                    msg = Float64MultiArray()
                    msg.data = joint_data
                    self.publisher_.publish(msg)

                elif len(data) == 352:
                    self.parse_full_skeleton(data[0:176])
                    self.parse_full_skeleton(data[176:352])
                elif len(data) == 176:
                    self.parse_full_skeleton(data[0:176])
        except Exception as e:
            self.get_logger().warn(f'Error receiving glove data: {str(e)}')

    # def timer_callback(self):
    #     try:
    #         # Simulate fake 24-joint data
    #         joint_data = [0.1 * i for i in range(24)]  # [0.0, 0.1, 0.2, ..., 2.3]

    #         # Print to terminal for verification
    #         print("Simulated glove joint data (24 values):")
    #         print(joint_data)

    #         # Publish to ROS2 topic
    #         msg = Float64MultiArray()
    #         msg.data = joint_data
    #         self.publisher_.publish(msg)

    #     except Exception as e:
    #         self.get_logger().warn(f'Error publishing test data: {str(e)}')



def main(args=None):
    rclpy.init(args=args)
    node = GlovePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
