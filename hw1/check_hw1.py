#!/usr/bin/env python3
"""Public HW1 checks shared by students and the Gradescope autograder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


RTOL = 1e-2
ATOL = 1e-8


def public_cases() -> list[dict[str, object]]:
    """Return deterministic cases that are safe to distribute to students."""
    standard = (
        [[3.265], [-0.865]],
        [[3.265], [0.865]],
        [[-0.725], [0.865]],
        [[-0.725], [-0.865]],
    )
    return [
        {
            "name": "Identity rotation",
            "xy": [[0.0], [0.0]],
            "theta": 0.0,
            "corners": standard,
            "expected": standard,
        },
        {
            "name": "Translation and quarter turn",
            "xy": [[2.0], [-1.0]],
            "theta": float(np.pi / 2),
            "corners": standard,
            "expected": (
                [[2.865], [2.265]],
                [[1.135], [2.265]],
                [[1.135], [-1.725]],
                [[2.865], [-1.725]],
            ),
        },
        {
            "name": "Negative rotation",
            "xy": [[-3.5], [4.25]],
            "theta": float(-np.pi / 3),
            "corners": standard,
            "expected": (
                [[-2.6166119743], [0.9899270566]],
                [[-1.1183880257], [1.8549270566]],
                [[-3.1133880257], [5.3103684177]],
                [[-4.6116119743], [4.4453684177]],
            ),
        },
        {
            "name": "General corner vectors",
            "xy": [[1.25], [0.75]],
            "theta": 0.37,
            "corners": (
                [[0.5], [-2.0]],
                [[1.75], [0.25]],
                [[-0.5], [1.25]],
                [[-2.0], [-0.75]],
            ),
            "expected": (
                [[2.4393945367], [-0.9338469752]],
                [[2.7911689968], [1.6159088423]],
                [[0.3318170372], [1.7346014660]],
                [[-0.3434431172], [-0.6724763731]],
            ),
        },
    ]


def evaluate_function(
    get_corners: Callable[..., object],
    cases: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Evaluate a function and return per-case, per-corner correctness."""
    results = []
    for case in cases:
        try:
            xy = np.asarray(case["xy"], dtype=float)
            corners = tuple(np.asarray(corner, dtype=float) for corner in case["corners"])
            actual = get_corners(xy.copy(), float(case["theta"]), *(c.copy() for c in corners))
            if not isinstance(actual, (tuple, list)) or len(actual) != 4:
                raise TypeError("get_corners must return exactly four values")

            corner_results = []
            for observed, expected in zip(actual, case["expected"], strict=True):
                observed_array = np.asarray(observed)
                expected_array = np.asarray(expected)
                correct = (
                    observed_array.shape == (2, 1)
                    and np.issubdtype(observed_array.dtype, np.number)
                    and bool(np.all(np.isfinite(observed_array)))
                    and bool(
                        np.allclose(observed_array, expected_array, rtol=RTOL, atol=ATOL)
                    )
                )
                corner_results.append(correct)
            results.append({"name": case["name"], "corners": corner_results})
        except BaseException as error:
            results.append(
                {
                    "name": case["name"],
                    "corners": [False] * 4,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return results


def load_submission(path: Path):
    """Load a student module from an exact path."""
    spec = importlib.util.spec_from_file_location("hw1_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "get_corners", None)
    if not callable(function):
        raise AttributeError("hw1.py must define a callable get_corners")
    return function


def main() -> int:
    submission = Path(__file__).with_name("hw1.py")
    if not submission.is_file():
        print(f"HW1 self-check: missing {submission.name} next to this checker.")
        return 1

    try:
        function = load_submission(submission)
    except BaseException as error:
        print(f"HW1 self-check: unable to import hw1.py: {type(error).__name__}: {error}")
        return 1

    results = evaluate_function(function, public_cases())
    passed = True
    for result in results:
        correct = sum(result["corners"])
        status = "PASS" if correct == 4 else "FAIL"
        print(f"[{status}] {result['name']}: {correct}/4 corners correct")
        if "error" in result:
            print(f"       {result['error']}")
        passed &= correct == 4

    print("All public HW1 checks passed." if passed else "Some public HW1 checks failed.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
