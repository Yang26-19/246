import rclpy
from rclpy.node import Node
# Import the message type that holds data describing robot joint angle states
from sensor_msgs.msg import JointState

# Import the class that publishes coordinate frame transform information
from tf2_ros import TransformBroadcaster

# Import the message type that expresses a transform from one coordinate frame to another
from geometry_msgs.msg import TransformStamped

import numpy as np
from numpy.typing import NDArray

from transform_helpers.utils import rotmat2q

# Modified DH Params for the Franka FR3 robot arm
# https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
# meters
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]

# radians
alpha_list = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = [
    "fr3_link0",
    "fr3_link1",
    "fr3_link2",
    "fr3_link3",
    "fr3_link4",
    "fr3_link5",
    "fr3_link6",
    "fr3_link7",
    "fr3_link8",
]

def get_transform_n_to_n_minus_one(n: int, theta: float) -> NDArray:
    """
    Calculates the transform from frame n to frame n-1 using modified Denavit-Hartenberg parameters.
    """
    a = DH_PARAMS[n - 1][0]
    d = DH_PARAMS[n - 1][1]
    alpha = DH_PARAMS[n - 1][2]

    transform_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])

    return transform_matrix


class ForwardKinematicCalculator(Node):
    def __init__(self):
        super().__init__("fk_calculator")

        # Create a subscriber to joint states
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.publish_transforms, 10
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Optional prefix for frames
        self.prefix = "my_robot/"

    def publish_transforms(self, msg: JointState):
        """
        Callback to publish transforms for each joint.
        """
        self.get_logger().debug(str(msg))

        # Traverse frames from last to first
        for i in range(len(FRAMES) - 1, -1, -1):
            frame_id = self.prefix + FRAMES[i]
            if i != 0:
                parent_id = self.prefix + FRAMES[i - 1]
            else:
                parent_id = self.prefix + BASE_FRAME

            # Determine the joint angle theta
            theta = 0
            if i != len(FRAMES) - 1 and i != 0:
                # Retrieve joint position from JointState message
                theta = msg.position[i - 1]
            elif i == len(FRAMES) - 1:
                # Static flange transform
                theta = 0
            else:
                theta = 0

            # Create the TransformStamped message
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_id
            t.child_frame_id = frame_id

            # Calculate the transform matrix
            if i != 0:
                transform = get_transform_n_to_n_minus_one(i, theta)
            else:
                transform = np.eye(4)  # Base transform is identity

            # Convert rotation matrix to quaternion
            quat = rotmat2q(transform[:3, :3])

            # Set translation and rotation in the TransformStamped message
            t.transform.translation.x = transform[0, 3]
            t.transform.translation.y = transform[1, 3]
            t.transform.translation.z = transform[2, 3]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    fk_calculator = ForwardKinematicCalculator()

    try:
        rclpy.spin(fk_calculator)  # Spin to keep the node running
    except KeyboardInterrupt:
        pass

    fk_calculator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
