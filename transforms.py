"""2D transformation matrices in homogeneous coordinates (3x3).

A pixel (x, y) becomes the column vector [x, y, 1]. Image coordinates:
x points right, y points DOWN. Every edit is a 3x3 matrix, so a chain of
edits is just a product of matrices.
"""
import math

from matrix import identity, mat_mul, determinant


def _clean(v):
    return 0.0 if abs(v) < 1e-15 else v


def translation(tx, ty):
    return [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]


def scaling(sx, sy):
    return [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]


def rotation(degrees):
    """Rotation that LOOKS counter-clockwise on screen (y axis points down)."""
    t = math.radians(degrees)
    c, s = _clean(math.cos(t)), _clean(math.sin(t))
    return [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]


def flip_horizontal():
    return [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def flip_vertical():
    return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]


def shear(kx=0.0, ky=0.0):
    """x' = x + kx*y ; y' = y + ky*x."""
    return [[1.0, kx, 0.0], [ky, 1.0, 0.0], [0.0, 0.0, 1.0]]


def compose(mats):
    """Combine matrices given in the order they are APPLIED.

    compose([A, B, C]) = C x B x A  (A acts first). Order matters.
    """
    result = identity(3)
    for M in mats:
        result = mat_mul(M, result)
    return result


def about_point(M, cx, cy):
    """Apply M about the point (cx, cy): move it to the origin, apply M, move back."""
    return compose([translation(-cx, -cy), M, translation(cx, cy)])


def describe_determinant(M, tol=1e-9):
    """Explain what the determinant of the transformation does to the image."""
    d = determinant(M)
    if abs(d) < tol:
        return {"det": d, "area_factor": 0.0, "orientation": "collapsed",
                "text": "det = 0: the image is flattened onto a line/point. "
                        "Information is lost and the matrix is not invertible."}
    flipped = d < 0
    mag = abs(d)
    size = ("enlarged" if mag > 1 + tol else "shrunk" if mag < 1 - tol
            else "same area")
    return {"det": d, "area_factor": mag,
            "orientation": "flipped" if flipped else "preserved",
            "text": f"det = {d:.4f}: area is multiplied by {mag:.4f} ({size}); "
                    + ("the image is mirrored (orientation reversed)."
                       if flipped else "orientation is preserved.")}
