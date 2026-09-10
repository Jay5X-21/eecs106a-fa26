"""HW1: transform the four vehicle corners into the world frame."""

import numpy as np


def get_corners(xy, theta, corner1, corner2, corner3, corner4):
    """Return the four world-frame corner positions in the same order.

    Args:
        xy: Vehicle position as a NumPy array of shape (2, 1).
        theta: Vehicle heading in radians, measured counterclockwise.
        corner1, corner2, corner3, corner4: Vehicle-frame corner positions,
            each with shape (2, 1).

    Returns:
        A tuple of four NumPy arrays, each with shape (2, 1).
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    world_corner1 = R @ corner1 + xy
    world_corner2 = R @ corner2 + xy
    world_corner3 = R @ corner3 + xy
    world_corner4 = R @ corner4 + xy

    return world_corner1, world_corner2, world_corner3, world_corner4
