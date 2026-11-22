"""Matrix utilities written from scratch.

Matrices are plain Python lists of lists of floats. numpy is NOT used here;
it is only used in tests/ to validate these functions.
"""

TOL = 1e-12  # pivots smaller than this are treated as zero


class SingularMatrixError(ValueError):
    """Raised when a matrix has no inverse (its determinant is zero)."""


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def shape(A):
    return (len(A), len(A[0]) if A else 0)


def transpose(A):
    rows, cols = shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def mat_mul(A, B):
    """Matrix product A x B. Entry (i, j) = row i of A . column j of B."""
    ra, ca = shape(A)
    rb, cb = shape(B)
    if ca != rb:
        raise ValueError(f"Cannot multiply {ra}x{ca} by {rb}x{cb}")
    return [[sum(A[i][k] * B[k][j] for k in range(ca)) for j in range(cb)]
            for i in range(ra)]


def mat_vec(A, v):
    """Matrix times vector."""
    if shape(A)[1] != len(v):
        raise ValueError("Matrix columns must equal vector length")
    return [sum(a * x for a, x in zip(row, v)) for row in A]


def _check_square(A):
    r, c = shape(A)
    if r != c or r == 0:
        raise ValueError("Matrix must be square and non-empty")
    return r


def determinant(A):
    """Determinant by Gaussian elimination with partial pivoting.

    Row swap flips the sign; adding a multiple of a row changes nothing;
    the determinant is the product of the pivots (times the sign).
    """
    n = _check_square(A)
    M = [[float(x) for x in row] for row in A]
    det = 1.0
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[p][col]) < TOL:
            return 0.0
        if p != col:
            M[col], M[p] = M[p], M[col]
            det = -det
        det *= M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n):
                M[r][c] -= f * M[col][c]
    return det


def determinant_cofactor(A):
    """Determinant by cofactor expansion along the first row (slow, for checks)."""
    n = _check_square(A)
    if n == 1:
        return float(A[0][0])
    if n == 2:
        return float(A[0][0] * A[1][1] - A[0][1] * A[1][0])
    total = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in A[1:]]
        total += ((-1) ** j) * A[0][j] * determinant_cofactor(minor)
    return total


def inverse(A):
    """Inverse by Gauss-Jordan elimination on [A | I] with partial pivoting.

    Raises SingularMatrixError if A has no inverse (det A = 0).
    """
    n = _check_square(A)
    aug = [[float(x) for x in A[i]] + identity(n)[i] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[p][col]) < TOL:
            raise SingularMatrixError(
                "Matrix is singular (determinant = 0): it has no inverse.")
        aug[col], aug[p] = aug[p], aug[col]
        pivot = aug[col][col]
        aug[col] = [x / pivot for x in aug[col]]
        for r in range(n):
            if r != col:
                f = aug[r][col]
                if f != 0.0:
                    aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def is_invertible(A):
    return abs(determinant(A)) >= TOL


def format_matrix(A, digits=3):
    return "\n".join("[ " + "  ".join(f"{x:8.{digits}f}" for x in row) + " ]"
                     for row in A)
