# HW2: Exponential Coordinates

Complete the documented 3D functions in `kin_func_skeleton.py` and the box and
camera functions in `hw2.py`. Keep their signatures unchanged. The supplied 2D
examples are for reference and are not graded.

Use the course Python environment (Python 3.13.5). From the repository root:

```bash
python -m pip install -r hw2/requirements.txt
python hw2/check_hw2.py
python hw2/kin_func_skeleton.py
```

The starter intentionally fails checks until implemented. `check_hw2.py`
provides a few simple numerical and interface checks; passing it does not
guarantee full credit. The original examples in the kinematics file provide a
few additional numerical checks, not exhaustive coverage.

## Kinematics conventions

Inputs are NumPy arrays: rotation vectors have shape `(3,)`, twists have shape
`(6,)` and order `[vx, vy, vz, wx, wy, wz]`. `axis_angle_to_SO3(omega, theta)` computes
`exp(hat(omega) * theta)`; non-unit axes are allowed, so the rotation angle is
`norm(omega) * theta`. `twist_to_SE3`, `se3_to_SE3`, and `forward_kinematics`
must support pure translation (`omega == 0`). `forward_kinematics` takes twists
as columns of a
`(6, N)` array and displacements as an `(N,)` array; multiply joint exponentials
in column order. It must also support prismatic and mixed chains. Outputs are
finite 3x3 or 4x4 matrices as documented in the function docstrings.

## Visualization

After completing the relevant functions, run the visualization from the
repository root:

```bash
python hw2/visualize_conveyor.py
```

The Swift browser view displays the conveyor, box, camera, coordinate frames,
and camera optical axis. It also compares the poses from your transform
functions with the motion described by your twists. Swift is only needed for
the visualization, not for the checker or grader.

Submit the complete semester repository. The grader reads `hw2/hw2.py` and
`hw2/kin_func_skeleton.py`; direct file uploads are not supported. Visualization
helpers are provided support code, not functions to complete.
