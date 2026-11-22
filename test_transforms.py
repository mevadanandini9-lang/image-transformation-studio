import numpy as np
import pytest

from math_engine.matrix import determinant, inverse, mat_mul, mat_vec, transpose
from math_engine.transforms import (about_point, compose, describe_determinant,
                                    flip_horizontal, flip_vertical, rotation,
                                    scaling, shear, translation)


def test_rotation_normal_90_degrees_counter_clockwise_on_screen():
    p = mat_vec(rotation(90), [1, 0, 1])      # a point to the right ...
    assert p == pytest.approx([0, -1, 1])     # ... moves UP (y is down on screen)

def test_rotation_edge_zero_and_full_turn_are_identity():
    assert np.allclose(rotation(0), np.eye(3))
    assert np.allclose(rotation(360), np.eye(3), atol=1e-12)

def test_rotation_vs_numpy_inverse_is_transpose():
    R = np.array(rotation(37))
    assert np.allclose(inverse(R.tolist()), np.linalg.inv(R))
    assert np.allclose(transpose(R.tolist()), np.linalg.inv(R))   # orthogonal matrix
    assert determinant(R.tolist()) == pytest.approx(1.0)


def test_determinants_of_each_edit():
    assert determinant(flip_horizontal()) == -1
    assert determinant(flip_vertical()) == -1
    assert determinant(shear(0.7, 0)) == pytest.approx(1.0)       # shear keeps area
    assert determinant(scaling(2, 3)) == pytest.approx(6.0)
    assert determinant(scaling(-2, 1)) == pytest.approx(-2.0)     # negative scale mirrors

def test_zero_scale_has_zero_determinant_and_is_described_as_collapsed():
    d = describe_determinant(scaling(1, 0))
    assert d["det"] == 0 and d["orientation"] == "collapsed"

def test_describe_determinant_flip_and_enlarge():
    assert describe_determinant(flip_horizontal())["orientation"] == "flipped"
    assert describe_determinant(scaling(2, 2))["area_factor"] == pytest.approx(4.0)


def test_compose_order_matters():
    a, b = rotation(90), scaling(2, 1)
    assert not np.allclose(compose([a, b]), compose([b, a]))
    # compose([A, B]) means A first, then B  ==  B x A
    assert np.allclose(compose([a, b]), mat_mul(b, a))

def test_compose_empty_is_identity_and_vs_numpy():
    assert np.allclose(compose([]), np.eye(3))
    A, B, C = rotation(20), shear(0.3, 0.1), scaling(2, 0.5)
    assert np.allclose(compose([A, B, C]), np.array(C) @ np.array(B) @ np.array(A))

def test_about_point_keeps_centre_fixed():
    M = about_point(compose([rotation(33), scaling(2, 2)]), 10, 20)
    assert mat_vec(M, [10, 20, 1]) == pytest.approx([10, 20, 1])

def test_undo_is_multiplying_by_inverse():
    A, B = rotation(30), shear(0.4, 0.0)
    total = compose([A, B])
    assert np.allclose(mat_mul(inverse(B), total), A)
