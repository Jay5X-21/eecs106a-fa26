"""SymPy starter for HW8: OSIRIS-RExploring, parts (b) and (c).

Run python hw8/OSIRIS-RExploring.py after completing the TODOs.
Documentation: https://docs.sympy.org/latest/index.html
"""

import sympy as sp


def main():
    """
    Enter the matrices from part (a) and construct the controllability matrix before running this script.
    """
    p = sp.symbols("p", positive=True)
    omega = sp.symbols("omega", real=True, nonzero=True)

    # TODO: enter A (4, 4) and B (4, 2) from part (a), using sp.Matrix.
    A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [3 * omega**2, 0, 0, 2 * p * omega], [0, 0, -2 * omega / p, 0]])
    B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1 / p]])

    # (b) Each eigenvalue is paired with its algebraic multiplicity.
    print("Eigenvalues and algebraic multiplicities:")
    sp.pprint(A.eigenvals())

    # Each entry contains an eigenvalue, multiplicity, and eigenvector basis.
    print("Eigenvalues, algebraic multiplicities, and eigenvector bases:")
    sp.pprint(A.eigenvects())

    # (c) TODO: construct C = [B, AB, A**2 B, A**3 B], with shape (4, 8).
    # sp.Matrix.hstack joins matrices horizontally; * multiplies SymPy matrices.
    C = sp.Matrix.hstack(B, A * B, A**2 * B, A**3 * B)

    print("Controllability matrix:")
    sp.pprint(C)
    print("Rank:", C.rank())


if __name__ == "__main__":
    main()
