
from geometry_msgs.msg import Quaternion
import numpy as np
from numpy.typing import NDArray
from math import sqrt

def rotmat2q(T: NDArray) -> Quaternion:
    """
    Converts a 3x3 rotation matrix to a ROS geometry_msgs.msg.Quaternion.
    """
    # Check if the input matrix is 3x3
    if T.shape != (3, 3):
        raise ValueError("Input rotation matrix must be 3x3.")

    # Calculate the trace of the rotation matrix
    trace = T[0, 0] + T[1, 1] + T[2, 2]

    if trace > 0:
        s = sqrt(trace + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * s
        qx = (T[2, 1] - T[1, 2]) / s
        qy = (T[0, 2] - T[2, 0]) / s
        qz = (T[1, 0] - T[0, 1]) / s
    elif (T[0, 0] > T[1, 1]) and (T[0, 0] > T[2, 2]):
        s = sqrt(1.0 + T[0, 0] - T[1, 1] - T[2, 2]) * 2  # S = 4 * qx
        qw = (T[2, 1] - T[1, 2]) / s
        qx = 0.25 * s
        qy = (T[0, 1] + T[1, 0]) / s
        qz = (T[0, 2] + T[2, 0]) / s
    elif T[1, 1] > T[2, 2]:
        s = sqrt(1.0 + T[1, 1] - T[0, 0] - T[2, 2]) * 2  # S = 4 * qy
        qw = (T[0, 2] - T[2, 0]) / s
        qx = (T[0, 1] + T[1, 0]) / s
        qy = 0.25 * s
        qz = (T[1, 2] + T[2, 1]) / s
    else:
        s = sqrt(1.0 + T[2, 2] - T[0, 0] - T[1, 1]) * 2  # S = 4 * qz
        qw = (T[1, 0] - T[0, 1]) / s
        qx = (T[0, 2] + T[2, 0]) / s
        qy = (T[1, 2] + T[2, 1]) / s
        qz = 0.25 * s

    # Return the quaternion as a ROS geometry_msgs.msg.Quaternion
    return Quaternion(x=qx, y=qy, z=qz, w=qw)
