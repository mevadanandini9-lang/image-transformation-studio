import numpy as np
import pytest

from math_engine.matrix import (SingularMatrixError, determinant, determinant_cofactor,
                                identity, inverse, mat_mul, mat_vec, transpose)

RNG = np.random.default_rng(0)


# ---- mat_mul -----------------------------------------------------------
def test_mat_mul_normal():
    assert mat_mul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]

def test_mat_mul_edge_shape_mismatch_and_noncommutative():
    with pytest.raises(ValueError):
        mat_mul([[1, 2, 3]], [[1, 2, 3]])
    A, B = [[1, 2], [3, 4]], [[0, 1], [1, 0]]
    assert mat_mul(A, B) != mat_mul(B, A)

def test_mat_mul_vs_numpy():
    A, B = RNG.normal(size=(4, 3)), RNG.normal(size=(3, 5))
    assert np.allclose(mat_mul(A.tolist(), B.tolist()), A @ B)

def test_mat_vec_and_transpose():
    assert mat_vec([[0, -1], [1, 0]], [1, 0]) == [0, 1]
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


# ---- determinant -------------------------------------------------------
def test_determinant_normal_hand_solved():
    # 3x3 solved by hand: 2(1*2-0*0) - 0 + 1(0*0-1*1) = 3
    assert determinant([[2, 0, 1], [0, 1, 0], [1, 0, 2]]) == pytest.approx(3.0)

def test_determinant_edge_singular_and_row_swap():
    assert determinant([[1, 2], [2, 4]]) == 0.0
    assert determinant([[0, 1], [1, 0]]) == pytest.approx(-1.0)  # needs a row swap

def test_determinant_vs_numpy():
    for n in (2, 3, 5):
        A = RNG.normal(size=(n, n))
        assert determinant(A.tolist()) == pytest.approx(np.linalg.det(A))

def test_determinant_cofactor_matches_elimination():
    A = RNG.normal(size=(4, 4)).tolist()
    assert determinant_cofactor(A) == pytest.approx(determinant(A))


# ---- inverse -----------------------------------------------------------
def test_inverse_normal_2x2_known():
    assert np.allclose(inverse([[4, 7], [2, 6]]), [[0.6, -0.7], [-0.2, 0.4]])

def test_inverse_edge_singular_raises():
    with pytest.raises(SingularMatrixError):
        inverse([[1, 2], [2, 4]])
    with pytest.raises(SingularMatrixError):
        inverse([[1, 0, 0], [0, 0, 0], [0, 0, 1]])

def test_inverse_vs_numpy_and_identity():
    A = RNG.normal(size=(5, 5))
    Ai = inverse(A.tolist())
    assert np.allclose(Ai, np.linalg.inv(A))
    assert np.allclose(mat_mul(A.tolist(), Ai), identity(5))

def test_inverse_needs_pivoting():
    A = [[0, 1], [1, 0]]               # zero in the first pivot position
    assert np.allclose(inverse(A), A)
