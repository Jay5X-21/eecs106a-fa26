"""Student functions for HW6: trajectories and integration in SE(3)."""

import numpy as np

from hw2.kin_func_skeleton import inverse_SE3, renormalize_SE3, se3_to_twist, twist_to_SE3


L1 = 0.57
L2 = 0.25
L3 = 0.20
L4 = 0.10
K = 0.5


def trajectory(t):
    """Return the chair-swing pose defined in the handout.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing g_ab(t) in the ride
        frame.
    """
    phi = K * np.sin(t)

    p = np.array([
        -L2 * np.sin(t) - L3 * np.sin(t) * np.sin(phi),
        L2 * np.cos(t) + L3 * np.cos(t) * np.sin(phi),
        L1 + L4 - L4 * np.cos(t) - L3 * np.cos(phi)
    ])

    Bx = np.array([-np.cos(t), -np.sin(t), 0])
    Bz = np.array([
        np.sin(phi) * np.sin(t),
        -np.sin(phi) * np.cos(t),
        np.cos(phi)
    ])

    By = np.cross(Bz, Bx)
    R = np.column_stack((Bx, By, Bz))

    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p

    return g


def body_velocity(t):
    """Calculate the instantaneous body velocity of the chair swing.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing the body velocity
        matrix in se(3). Analytically differentiate the full trajectory
        matrix, including its rotation, and compute the body velocity.

    inverse_SE3 from kin_func_skeleton inverts the pose.
    """
    phi = K * np.sin(t)
    phi_dot = K * np.cos(t)

    s = np.sin(t)
    c = np.cos(t)
    sp = np.sin(phi)
    cp = np.cos(phi)

    p_dot = np.array([
        -L2 * c - L3 * c * sp - L3 * s * cp * phi_dot,
        -L2 * s - L3 * s * sp + L3 * c * cp * phi_dot,
        L4 * s + L3 * sp * phi_dot
        ])

    R_dot = np.array([
        [s, c * cp - phi_dot * s * sp, c * sp + phi_dot * s * cp],
        [-c, s * cp + phi_dot * c * sp, s * sp - phi_dot * c * cp],
        [0, -phi_dot * cp, -phi_dot * sp]
        ])

    g_dot = np.zeros((4, 4))
    g_dot[:3, :3] = R_dot
    g_dot[:3, 3] = p_dot

    g = trajectory(t)
    V_body_hat = inverse_SE3(g) @ g_dot

    return V_body_hat


def euler_step(pose, velocity_hat, dt):
    """Advance an SE(3) pose with body velocity held constant for one step.

    Args:
        pose: NumPy array with shape (4, 4), the current pose.
        velocity_hat: NumPy array with shape (4, 4), a body velocity in se(3).
        dt: Nonnegative scalar timestep in seconds.

    Returns:
        NumPy array with shape (4, 4), the next SE(3) pose. Apply the
        exponential rigid-motion increment on the right of the current
        pose, as in Problem 4.

    se3_to_twist converts the velocity matrix to a vector. twist_to_SE3
    computes the rigid-motion increment over dt. Apply renormalize_SE3
    to the updated pose to correct floating-point drift in its rotation.
    """
    twist = se3_to_twist(velocity_hat)
    increment = twist_to_SE3(twist, dt)

    next_pose = pose @ increment
    next_pose = renormalize_SE3(next_pose)

    return next_pose