import numpy as np
import pytest

from math_engine.matrix import SingularMatrixError
from math_engine.transforms import (about_point, flip_horizontal, flip_vertical,
                                    rotation, scaling, shear, compose)
from math_engine.warp import (forward_vs_inverse, forward_warp, inverse_warp,
                              output_canvas)

RNG = np.random.default_rng(1)
GRAY = RNG.integers(0, 256, size=(20, 30), dtype=np.uint8)
COLOR = RNG.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
I3 = np.eye(3).tolist()


def centred(M, img):
    h, w = img.shape[:2]
    return about_point(M, (w - 1) / 2, (h - 1) / 2)


# ---- inverse_warp: normal, edge, library comparison -------------------------
@pytest.mark.parametrize("interp", ["nearest", "bilinear"])
def test_identity_returns_same_image(interp):
    assert np.array_equal(inverse_warp(GRAY, I3, interp), GRAY)
    assert np.array_equal(inverse_warp(COLOR, I3, interp), COLOR)

def test_flips_match_numpy():
    assert np.array_equal(inverse_warp(GRAY, centred(flip_horizontal(), GRAY), "nearest"),
                          GRAY[:, ::-1])
    assert np.array_equal(inverse_warp(COLOR, centred(flip_vertical(), COLOR), "nearest"),
                          COLOR[::-1])

def test_rotate_90_matches_numpy_rot90():
    out = inverse_warp(GRAY, centred(rotation(90), GRAY), "nearest")
    assert np.array_equal(out, np.rot90(GRAY))          # counter-clockwise
    out = inverse_warp(COLOR, centred(rotation(-90), COLOR), "nearest")
    assert np.array_equal(out, np.rot90(COLOR, -1))

def test_singular_matrix_raises():
    with pytest.raises(SingularMatrixError):
        inverse_warp(GRAY, centred(scaling(1, 0), GRAY))

def test_invalid_interpolation_raises():
    with pytest.raises(ValueError):
        inverse_warp(GRAY, I3, "cubic")

def test_bilinear_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    from math_engine.matrix import inverse
    from math_engine.warp import output_canvas
    M = centred(compose([rotation(25), scaling(1.3, 1.3)]), GRAY)
    out, valid = inverse_warp(GRAY, M, "bilinear", expand=True, return_mask=True)
    h, w = GRAY.shape
    ow, oh, Mf = output_canvas(M, w, h, True)
    Mi = inverse(Mf)
    # scipy works in (row, col) = (y, x)
    ref = ndimage.affine_transform(
        GRAY.astype(float),
        [[Mi[1][1], Mi[1][0]], [Mi[0][1], Mi[0][0]]],
        offset=[Mi[1][2], Mi[0][2]], output_shape=(oh, ow), order=1, cval=0.0)
    inside = valid.copy()
    inside[:2], inside[-2:], inside[:, :2], inside[:, -2:] = False, False, False, False
    assert np.abs(out[inside].astype(float) - ref[inside]).max() <= 1.0


# ---- canvas ----------------------------------------------------------------
def test_canvas_sizes():
    assert output_canvas(I3, 30, 20)[:2] == (30, 20)
    w, h, _ = output_canvas(centred(rotation(90), GRAY), 30, 20)
    assert (w, h) == (20, 30)
    with pytest.raises(ValueError):
        output_canvas(scaling(500, 500), 30, 20)         # too large

def test_no_expand_keeps_size():
    assert inverse_warp(GRAY, centred(rotation(30), GRAY), expand=False).shape == GRAY.shape


# ---- inverse mapping avoids holes -----------------------------------------------
@pytest.mark.parametrize("M", [rotation(30), scaling(2, 2), shear(0.5, 0.2)])
def test_forward_leaves_holes_inverse_does_not(M):
    white = np.full((40, 40), 255, dtype=np.uint8)
    r = forward_vs_inverse(white, centred(M, white))
    assert r["n_holes"] > 0                              # forward mapping has holes
    inv, footprint = inverse_warp(white, centred(M, white), "nearest", return_mask=True)
    assert np.all(inv[footprint] == 255)                 # inverse mapping: every pixel filled

def test_forward_warp_with_zero_determinant_collapses_to_a_line():
    out, written = forward_warp(GRAY, centred(scaling(1, 0), GRAY), expand=False)
    assert len(np.unique(np.nonzero(written)[0])) == 1   # all pixels in one row
